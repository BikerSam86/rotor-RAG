"""
Rotor-RAG: 2-bit Ternary Neural Networks
Encoding ternary logic with binary stability
"""

from .core import RotorCore
from .layers import TernaryLinear, RotorTransformer
from .quantization import quantize_ternary, dequantize_ternary

__version__ = "0.1.0"
__all__ = [
    "RotorCore",
    "TernaryLinear",
    "RotorTransformer",
    "quantize_ternary",
    "dequantize_ternary"
]
