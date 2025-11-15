"""
Python bindings for native C/CUDA implementations.
Uses ctypes for loading shared libraries.
"""

import ctypes
import numpy as np
import os
import sys
from pathlib import Path
from typing import Tuple, Optional

# Find native library
def find_library():
    """Locate the compiled native library."""
    lib_dir = Path(__file__).parent.parent.parent / "native" / "build"

    if sys.platform == "win32":
        lib_name = "librotor.dll"
    elif sys.platform == "darwin":
        lib_name = "librotor.dylib"
    else:
        lib_name = "librotor.so"

    lib_path = lib_dir / lib_name

    if not lib_path.exists():
        raise FileNotFoundError(
            f"Native library not found at {lib_path}. "
            "Please run: python native/build.py"
        )

    return str(lib_path)


# Load library (lazy loading)
_lib = None

def get_lib():
    """Get or load the native library."""
    global _lib
    if _lib is None:
        try:
            lib_path = find_library()
            _lib = ctypes.CDLL(lib_path)
            _setup_functions()
        except FileNotFoundError:
            # Library not built, will fall back to NumPy implementation
            return None
    return _lib


def _setup_functions():
    """Setup function signatures for the C library."""
    lib = _lib

    # rotor_encode
    lib.rotor_encode.argtypes = [
        ctypes.POINTER(ctypes.c_int8),  # values
        ctypes.c_size_t,                # n
        ctypes.POINTER(ctypes.c_uint8), # bit0
        ctypes.POINTER(ctypes.c_uint8), # bit1
    ]
    lib.rotor_encode.restype = None

    # rotor_decode
    lib.rotor_decode.argtypes = [
        ctypes.POINTER(ctypes.c_uint8), # bit0
        ctypes.POINTER(ctypes.c_uint8), # bit1
        ctypes.c_size_t,                # n
        ctypes.POINTER(ctypes.c_int8),  # values
    ]
    lib.rotor_decode.restype = None

    # rotor_dot
    lib.rotor_dot.argtypes = [
        ctypes.POINTER(ctypes.c_uint8), # a_bit0
        ctypes.POINTER(ctypes.c_uint8), # a_bit1
        ctypes.POINTER(ctypes.c_uint8), # b_bit0
        ctypes.POINTER(ctypes.c_uint8), # b_bit1
        ctypes.c_size_t,                # n
    ]
    lib.rotor_dot.restype = ctypes.c_int32

    # rotor_matvec
    lib.rotor_matvec.argtypes = [
        ctypes.POINTER(ctypes.c_uint8), # W_bit0
        ctypes.POINTER(ctypes.c_uint8), # W_bit1
        ctypes.POINTER(ctypes.c_float), # x
        ctypes.c_size_t,                # m
        ctypes.c_size_t,                # n
        ctypes.POINTER(ctypes.c_int32), # y
    ]
    lib.rotor_matvec.restype = None

    # rotor_batch_matmul
    lib.rotor_batch_matmul.argtypes = [
        ctypes.POINTER(ctypes.c_uint8), # W_bit0
        ctypes.POINTER(ctypes.c_uint8), # W_bit1
        ctypes.POINTER(ctypes.c_float), # X
        ctypes.c_size_t,                # batch_size
        ctypes.c_size_t,                # m
        ctypes.c_size_t,                # n
        ctypes.POINTER(ctypes.c_int32), # Y
    ]
    lib.rotor_batch_matmul.restype = None

    # rotor_quantize_ternary
    lib.rotor_quantize_ternary.argtypes = [
        ctypes.POINTER(ctypes.c_float), # values
        ctypes.c_size_t,                # n
        ctypes.c_float,                 # threshold
        ctypes.POINTER(ctypes.c_int8),  # output
    ]
    lib.rotor_quantize_ternary.restype = None


# ============================================================================
# High-level Python API
# ============================================================================

class NativeBackend:
    """Native C/CUDA backend for rotor operations."""

    def __init__(self, use_cuda: bool = False):
        """
        Args:
            use_cuda: Whether to use CUDA implementation (if available)
        """
        self.lib = get_lib()
        self.use_cuda = use_cuda and self._has_cuda()
        self.available = self.lib is not None

    def _has_cuda(self) -> bool:
        """Check if CUDA support is available."""
        if self.lib is None:
            return False
        try:
            # Check if CUDA functions exist
            return hasattr(self.lib, 'rotor_dot_cuda')
        except:
            return False

    def encode(self, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Encode ternary values to 2-bit rotor format."""
        if not self.available:
            from .core import RotorCore
            return RotorCore.encode(values)

        values = np.asarray(values, dtype=np.int8).ravel()
        n = len(values)

        bit0 = np.zeros(n, dtype=np.uint8)
        bit1 = np.zeros(n, dtype=np.uint8)

        self.lib.rotor_encode(
            values.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            n,
            bit0.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            bit1.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        )

        return bit0, bit1

    def decode(self, bit0: np.ndarray, bit1: np.ndarray) -> np.ndarray:
        """Decode 2-bit rotor format to ternary values."""
        if not self.available:
            from .core import RotorCore
            return RotorCore.decode(bit0, bit1)

        bit0 = np.asarray(bit0, dtype=np.uint8).ravel()
        bit1 = np.asarray(bit1, dtype=np.uint8).ravel()
        n = len(bit0)

        values = np.zeros(n, dtype=np.int8)

        self.lib.rotor_decode(
            bit0.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            bit1.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            n,
            values.ctypes.data_as(ctypes.POINTER(ctypes.c_int8))
        )

        return values

    def dot(
        self,
        a_bit0: np.ndarray,
        a_bit1: np.ndarray,
        b_bit0: np.ndarray,
        b_bit1: np.ndarray
    ) -> int:
        """Compute dot product of two rotor arrays."""
        if not self.available:
            from .core import RotorCore
            return RotorCore.dot(a_bit0, a_bit1, b_bit0, b_bit1)

        a_bit0 = np.asarray(a_bit0, dtype=np.uint8).ravel()
        a_bit1 = np.asarray(a_bit1, dtype=np.uint8).ravel()
        b_bit0 = np.asarray(b_bit0, dtype=np.uint8).ravel()
        b_bit1 = np.asarray(b_bit1, dtype=np.uint8).ravel()
        n = len(a_bit0)

        result = self.lib.rotor_dot(
            a_bit0.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            a_bit1.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            b_bit0.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            b_bit1.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            n
        )

        return int(result)

    def matvec(
        self,
        W_bit0: np.ndarray,
        W_bit1: np.ndarray,
        x: np.ndarray
    ) -> np.ndarray:
        """Matrix-vector multiply: y = W @ x."""
        if not self.available:
            from .core import RotorCore
            return RotorCore.matmul(W_bit0, W_bit1, x)

        W_bit0 = np.asarray(W_bit0, dtype=np.uint8)
        W_bit1 = np.asarray(W_bit1, dtype=np.uint8)
        x = np.asarray(x, dtype=np.float32).ravel()

        m, n = W_bit0.shape
        y = np.zeros(m, dtype=np.int32)

        self.lib.rotor_matvec(
            W_bit0.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            W_bit1.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            m, n,
            y.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
        )

        return y


# Global backend instance
_backend = None

def get_backend(use_cuda: bool = False) -> NativeBackend:
    """Get or create native backend."""
    global _backend
    if _backend is None:
        _backend = NativeBackend(use_cuda=use_cuda)
    return _backend
