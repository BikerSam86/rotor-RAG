"""
Quantization utilities for ternary neural networks.
"""

import numpy as np
from typing import Tuple, Optional


def quantize_ternary(
    values: np.ndarray,
    threshold: float = 0.0,
    method: str = "sign"
) -> np.ndarray:
    """
    Quantize continuous values to ternary {-1, 0, +1}.

    Args:
        values: Input array (any dtype)
        threshold: Values in [-threshold, threshold] map to 0
        method: Quantization method
            - "sign": Sign function with threshold
            - "threshold": Threshold-based with configurable bands

    Returns:
        Ternary array {-1, 0, +1}

    Examples:
        >>> x = np.array([0.8, -0.5, 0.1, -0.9, 0.0])
        >>> quantize_ternary(x, threshold=0.3)
        array([ 1, -1,  0, -1,  0])
    """
    values = np.asarray(values, dtype=np.float32)

    if method == "sign":
        # Simple sign quantization with dead zone
        result = np.where(
            np.abs(values) <= threshold,
            0,
            np.sign(values)
        )
    elif method == "threshold":
        # Explicit threshold bands
        result = np.where(
            values > threshold,
            1,
            np.where(values < -threshold, -1, 0)
        )
    else:
        raise ValueError(f"Unknown quantization method: {method}")

    return result.astype(np.int8)


def dequantize_ternary(
    ternary: np.ndarray,
    scale: float = 1.0
) -> np.ndarray:
    """
    Convert ternary values back to float (for gradient computation).

    Args:
        ternary: Ternary array {-1, 0, +1}
        scale: Scaling factor for output

    Returns:
        Float array
    """
    return ternary.astype(np.float32) * scale


def straight_through_estimator(
    forward_output: np.ndarray,
    backward_gradient: np.ndarray
) -> np.ndarray:
    """
    Straight-through estimator for ternary quantization.

    In forward pass: Use quantized values
    In backward pass: Pass gradients straight through

    Args:
        forward_output: Quantized output (not used in gradient)
        backward_gradient: Gradient from next layer

    Returns:
        Gradient to pass to previous layer (unchanged)
    """
    # In numpy, we just return the gradient
    # In PyTorch/JAX, this would use custom autograd
    return backward_gradient


class TernaryQuantizer:
    """
    Stateful quantizer with learnable thresholds.
    """

    def __init__(
        self,
        initial_threshold: float = 0.0,
        learnable: bool = False
    ):
        """
        Args:
            initial_threshold: Initial quantization threshold
            learnable: Whether threshold should be learned during training
        """
        self.threshold = initial_threshold
        self.learnable = learnable

    def __call__(self, values: np.ndarray) -> np.ndarray:
        """Quantize values to ternary."""
        return quantize_ternary(values, threshold=self.threshold)

    def update_threshold(self, new_threshold: float):
        """Update quantization threshold (e.g., during training)."""
        if self.learnable:
            self.threshold = new_threshold


def quantize_activations(
    x: np.ndarray,
    bits: int = 8,
    symmetric: bool = True
) -> Tuple[np.ndarray, float]:
    """
    Quantize activations to low-bit integers (for W2A8 mode).

    Args:
        x: Input activations
        bits: Number of bits (default 8 for int8)
        symmetric: Use symmetric quantization around zero

    Returns:
        Tuple of (quantized_values, scale_factor)
    """
    if symmetric:
        # Symmetric quantization
        max_val = np.max(np.abs(x))
        qmax = 2 ** (bits - 1) - 1  # e.g., 127 for int8

        scale = max_val / qmax if max_val > 0 else 1.0
        quantized = np.clip(
            np.round(x / scale),
            -qmax - 1,
            qmax
        ).astype(np.int8)

        return quantized, scale
    else:
        # Asymmetric quantization
        min_val = np.min(x)
        max_val = np.max(x)
        qmin = -(2 ** (bits - 1))
        qmax = 2 ** (bits - 1) - 1

        scale = (max_val - min_val) / (qmax - qmin) if max_val > min_val else 1.0
        zero_point = qmin - np.round(min_val / scale)

        quantized = np.clip(
            np.round(x / scale + zero_point),
            qmin,
            qmax
        ).astype(np.int8)

        return quantized, scale


def dequantize_activations(
    quantized: np.ndarray,
    scale: float
) -> np.ndarray:
    """
    Dequantize activations back to float.

    Args:
        quantized: Quantized integer values
        scale: Scale factor from quantization

    Returns:
        Dequantized float values
    """
    return quantized.astype(np.float32) * scale
