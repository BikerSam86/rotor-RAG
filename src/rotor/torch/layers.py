"""
PyTorch ternary layers with straight-through estimator.

Key insight: Ternary multiply = NO MULTIPLY!
- Weight = +1 → keep activation
- Weight = -1 → negate activation
- Weight =  0 → zero activation

Just bit ops + adds. No expensive hardware needed!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TernaryQuantize(torch.autograd.Function):
    """
    Straight-through estimator for ternary quantization.

    Forward: Quantize to {-1, 0, +1}
    Backward: Pass gradient straight through (pretend no quantization)

    This is the KEY to training - gradients flow as if we didn't quantize!
    """

    @staticmethod
    def forward(ctx, input, threshold=0.0):
        """
        Quantize to ternary.

        Args:
            input: Float tensor
            threshold: Dead zone around zero

        Returns:
            Ternary tensor {-1, 0, +1}
        """
        # Quantize
        output = input.clone()
        output[torch.abs(input) <= threshold] = 0
        output[input > threshold] = 1
        output[input < -threshold] = -1

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Straight-through: gradient passes unchanged.

        This is the magic - we pretend quantization didn't happen!
        """
        # Gradient flows straight through
        grad_input = grad_output.clone()

        return grad_input, None  # None for threshold (not learnable)


class TernaryLinear(nn.Module):
    """
    Linear layer with ternary weights.

    During training:
    - Weights stored as float32 (shadow weights)
    - Forward pass: quantize to ternary
    - Backward pass: update float weights
    - Gradients flow via straight-through estimator

    After training:
    - Discard float weights
    - Store only 2-bit ternary encoding
    - Inference uses popcount + adds (NO MULTIPLY!)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        threshold: float = 0.3
    ):
        """
        Args:
            in_features: Input dimension
            out_features: Output dimension
            bias: Whether to use bias
            threshold: Quantization threshold (values in [-t, t] → 0)
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.threshold = threshold

        # Shadow weights (full precision, trainable)
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        # Initialize
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights to encourage ternary distribution."""
        # Use Kaiming initialization but with larger scale
        # This ensures most weights START outside the dead zone
        fan_in = self.in_features
        std = (2.0 / fan_in) ** 0.5
        bound = std * 2.0  # Larger bound to avoid threshold
        nn.init.uniform_(self.weight, -bound, bound)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input):
        """
        Forward pass with ternary weights.

        This is where the magic happens:
        1. Quantize weights to {-1, 0, +1}
        2. Matrix multiply (but it's really just adds/subtracts!)
        3. Add bias

        NO EXPENSIVE MULTIPLIES NEEDED!
        """
        # Quantize weights using straight-through estimator
        weight_ternary = TernaryQuantize.apply(self.weight, self.threshold)

        # Linear operation
        # In reality, this becomes: popcount + adds (see below)
        output = F.linear(input, weight_ternary, self.bias)

        return output

    def get_weight_stats(self):
        """Get statistics about quantized weights."""
        with torch.no_grad():
            weight_ternary = TernaryQuantize.apply(self.weight, self.threshold)

            total = weight_ternary.numel()
            zeros = (weight_ternary == 0).sum().item()
            positives = (weight_ternary == 1).sum().item()
            negatives = (weight_ternary == -1).sum().item()

            return {
                'total': total,
                'zeros': zeros,
                'positives': positives,
                'negatives': negatives,
                'sparsity': zeros / total,
            }


class TernaryMLP(nn.Module):
    """
    Simple MLP with ternary weights.

    Great for MNIST and other simple tasks.
    Proves that ternary networks can learn!
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: list = [256, 128],
        num_classes: int = 10,
        threshold: float = 0.3
    ):
        """
        Args:
            input_dim: Input dimension (e.g., 784 for MNIST)
            hidden_dims: Hidden layer sizes
            num_classes: Number of output classes
            threshold: Quantization threshold
        """
        super().__init__()

        layers = []

        # Input layer
        layers.append(TernaryLinear(input_dim, hidden_dims[0], threshold=threshold))
        layers.append(nn.ReLU())

        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(TernaryLinear(hidden_dims[i], hidden_dims[i+1], threshold=threshold))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(TernaryLinear(hidden_dims[-1], num_classes, threshold=threshold))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass."""
        return self.network(x)

    def get_stats(self):
        """Get weight statistics for all ternary layers."""
        stats = {}
        layer_idx = 0

        for name, module in self.named_modules():
            if isinstance(module, TernaryLinear):
                stats[f'layer_{layer_idx}'] = module.get_weight_stats()
                layer_idx += 1

        return stats


class ActivationQuantize(torch.autograd.Function):
    """
    Quantize activations to int8 (for W2A8 mode).

    This is optional - can also keep activations as float.
    But quantizing activations too makes it REALLY fast.
    """

    @staticmethod
    def forward(ctx, input, bits=8):
        """Quantize activations."""
        # Symmetric quantization
        max_val = torch.abs(input).max()
        qmax = 2 ** (bits - 1) - 1

        scale = max_val / qmax if max_val > 0 else 1.0
        ctx.scale = scale

        quantized = torch.clamp(
            torch.round(input / scale),
            -qmax - 1,
            qmax
        )

        return quantized * scale  # Dequantize for next layer

    @staticmethod
    def backward(ctx, grad_output):
        """Straight through."""
        return grad_output, None


def quantize_model_weights(model):
    """
    Convert trained model to pure ternary (for deployment).

    After training, we can:
    1. Quantize all weights to ternary
    2. Encode as 2-bit pairs
    3. Throw away float weights
    4. Model is now 16× smaller!
    """
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, TernaryLinear):
                # Quantize permanently
                weight_ternary = TernaryQuantize.apply(
                    module.weight,
                    module.threshold
                )
                module.weight.data = weight_ternary

    return model


# ==============================================================================
# THE KEY INSIGHT: What ternary multiply actually does
# ==============================================================================

def ternary_multiply_explained(weight, activation):
    """
    What happens when we "multiply" ternary weights × activations.

    NO ACTUAL MULTIPLY HARDWARE NEEDED!

    Args:
        weight: Ternary {-1, 0, +1}
        activation: Any value

    Returns:
        Product (but computed via simple ops)
    """
    # This is what the "multiply" actually becomes:

    if weight == 1:
        result = activation          # Just keep it
    elif weight == -1:
        result = -activation         # Just negate (flip sign bit)
    else:  # weight == 0
        result = 0                   # Just zero

    # On hardware:
    # - Case +1: Pass through (no op)
    # - Case -1: XOR sign bit (1 instruction)
    # - Case  0: Set to zero (1 instruction)

    # NO MULTIPLY INSTRUCTION EXECUTED!

    return result


def ternary_matvec_explained(W_ternary, x):
    """
    Matrix-vector multiply with ternary weights.

    What actually happens:
    1. For each output neuron:
       - Separate activations into 3 groups: weight=+1, -1, 0
       - Sum the +1 group
       - Sum the -1 group and negate
       - Ignore the 0 group
       - Final = sum_pos - sum_neg

    2. In practice with 2-bit encoding:
       - AND operations to select groups
       - Popcount to sum
       - Integer subtract

    ZERO MULTIPLIES!
    """
    m, n = W_ternary.shape
    y = torch.zeros(m)

    for i in range(m):
        # Get row of weights
        w = W_ternary[i]

        # Separate into groups (conceptually)
        pos_mask = (w == 1)
        neg_mask = (w == -1)
        # zero_mask = (w == 0)  # Don't need this, just skip

        # Sum positive group
        sum_pos = x[pos_mask].sum() if pos_mask.any() else 0

        # Sum negative group (then negate)
        sum_neg = x[neg_mask].sum() if neg_mask.any() else 0

        # Result
        y[i] = sum_pos - sum_neg

        # On actual hardware with 2-bit encoding:
        # sum_pos = sum(x[AND(bit0, NOT bit1)])  ← popcount!
        # sum_neg = sum(x[AND(NOT bit0, bit1)])  ← popcount!
        # y[i] = sum_pos - sum_neg               ← subtract!

        # NO MULTIPLIES!

    return y


# ==============================================================================
# Training utilities
# ==============================================================================

def count_parameters(model):
    """Count total parameters."""
    return sum(p.numel() for p in model.parameters())


def memory_footprint(model, ternary=True):
    """
    Calculate memory footprint.

    Args:
        model: PyTorch model
        ternary: If True, calculate for ternary encoding

    Returns:
        Memory in bytes
    """
    total_params = count_parameters(model)

    if ternary:
        # 2 bits per parameter
        bytes_used = total_params * 2 / 8
    else:
        # Float32: 4 bytes per parameter
        bytes_used = total_params * 4

    return bytes_used
