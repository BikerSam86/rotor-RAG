"""
Ternary neural network layers using 2-bit rotor encoding.
"""

import numpy as np
from typing import Optional, Tuple
from .core import RotorCore, encode_ternary, decode_ternary
from .quantization import quantize_ternary, TernaryQuantizer


class TernaryLinear:
    """
    Fully connected layer with ternary weights.

    Weights are stored in 2-bit rotor format for efficiency.
    During forward pass, inputs are quantized to ternary.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        quantizer: Optional[TernaryQuantizer] = None
    ):
        """
        Args:
            in_features: Input dimension
            out_features: Output dimension
            bias: Whether to include bias term
            quantizer: Quantizer for inputs (default: sign quantization)
        """
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        # Quantizer for inputs
        self.quantizer = quantizer or TernaryQuantizer(initial_threshold=0.0)

        # Initialize weights (float32 for training, will be quantized)
        self.weight_float = self._init_weights()

        # Ternary weights (2-bit encoded)
        self.weight_bit0, self.weight_bit1 = self._quantize_weights()

        # Bias (kept as float for now)
        if bias:
            self.bias = np.zeros(out_features, dtype=np.float32)
        else:
            self.bias = None

    def _init_weights(self) -> np.ndarray:
        """
        Initialize weights using Xavier/Glorot initialization.
        Adapted for ternary: initialize close to {-1, 0, +1}.
        """
        # Xavier init
        std = np.sqrt(2.0 / (self.in_features + self.out_features))
        weights = np.random.randn(self.out_features, self.in_features) * std

        # Push towards ternary values
        weights = np.clip(weights, -1.5, 1.5)

        return weights.astype(np.float32)

    def _quantize_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        """Quantize float weights to ternary rotor format."""
        weight_ternary = quantize_ternary(self.weight_float, threshold=0.3)
        return encode_ternary(weight_ternary)

    def update_ternary_weights(self):
        """
        Update ternary encoding from float weights.
        Call this after training updates.
        """
        self.weight_bit0, self.weight_bit1 = self._quantize_weights()

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: W @ x + b

        Args:
            x: Input [batch_size, in_features] or [in_features]

        Returns:
            Output [batch_size, out_features] or [out_features]
        """
        # Quantize input
        x_ternary = self.quantizer(x)
        x_bit0, x_bit1 = encode_ternary(x_ternary)

        # Matrix multiply using rotor operations
        if x.ndim == 1:
            # Single vector
            output = np.zeros(self.out_features, dtype=np.int32)
            for i in range(self.out_features):
                output[i] = RotorCore.dot(
                    self.weight_bit0[i], self.weight_bit1[i],
                    x_bit0, x_bit1
                )
        else:
            # Batch
            batch_size = x.shape[0]
            output = np.zeros((batch_size, self.out_features), dtype=np.int32)
            for b in range(batch_size):
                for i in range(self.out_features):
                    output[b, i] = RotorCore.dot(
                        self.weight_bit0[i], self.weight_bit1[i],
                        x_bit0[b], x_bit1[b]
                    )

        # Convert to float and add bias
        output = output.astype(np.float32)
        if self.use_bias and self.bias is not None:
            output = output + self.bias

        return output

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Alias for forward()."""
        return self.forward(x)

    def get_weight_stats(self) -> dict:
        """Get statistics about weight distribution."""
        weights = decode_ternary(self.weight_bit0, self.weight_bit1).ravel()

        return {
            'total': len(weights),
            'zeros': np.sum(weights == 0),
            'positives': np.sum(weights == 1),
            'negatives': np.sum(weights == -1),
            'sparsity': np.sum(weights == 0) / len(weights),
        }


class ReLU:
    """Ternary ReLU activation: max(0, x)."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply ReLU."""
        return np.maximum(0, x)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class SignActivation:
    """Sign activation: returns {-1, 0, +1}."""

    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply sign activation with threshold."""
        return quantize_ternary(x, threshold=self.threshold)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


class TernarySequential:
    """
    Sequential container for ternary layers.
    """

    def __init__(self, *layers):
        """
        Args:
            *layers: Variable number of layer objects
        """
        self.layers = list(layers)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through all layers."""
        for layer in self.layers:
            x = layer(x)
        return x

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def add(self, layer):
        """Add a layer to the sequence."""
        self.layers.append(layer)


class SimpleRotorNet:
    """
    Simple 2-layer ternary network for demonstration.

    Architecture:
        input -> TernaryLinear -> ReLU -> TernaryLinear -> output
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int
    ):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Layers
        self.fc1 = TernaryLinear(input_dim, hidden_dim, bias=True)
        self.relu = ReLU()
        self.fc2 = TernaryLinear(hidden_dim, output_dim, bias=True)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Get predicted class indices (argmax)."""
        logits = self.forward(x)
        if logits.ndim == 1:
            return np.argmax(logits)
        else:
            return np.argmax(logits, axis=1)

    def get_stats(self) -> dict:
        """Get network statistics."""
        return {
            'layer1_weights': self.fc1.get_weight_stats(),
            'layer2_weights': self.fc2.get_weight_stats(),
        }


class RotorTransformer:
    """
    Placeholder for transformer architecture with rotor encoding.
    Will implement attention and feed-forward with ternary weights.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_layers: int
    ):
        """
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            d_ff: Feed-forward dimension
            n_layers: Number of transformer layers
        """
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.n_layers = n_layers

        # TODO: Implement transformer blocks
        raise NotImplementedError("Full transformer coming soon!")

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError()
