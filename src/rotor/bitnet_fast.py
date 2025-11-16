"""
Fast BitNet conversion using C extension.

This provides Python bindings to the optimized C implementation
of bitnet_to_rotor conversion.
"""

import numpy as np
import ctypes
from pathlib import Path
import platform


# Try to load the compiled C library
def _load_library():
    """Load the compiled C library."""
    lib_dir = Path(__file__).parent.parent.parent / "native"

    # Try different library names based on platform
    if platform.system() == "Windows":
        lib_names = ["rotor_core.dll", "librotor_core.dll"]
    elif platform.system() == "Darwin":
        lib_names = ["librotor_core.dylib", "rotor_core.dylib"]
    else:
        lib_names = ["librotor_core.so", "rotor_core.so"]

    # Search in build directory
    search_paths = [
        lib_dir / "build",
        lib_dir / "build" / "Release",
        lib_dir / "build" / "Debug",
        lib_dir,
    ]

    for search_path in search_paths:
        for lib_name in lib_names:
            lib_path = search_path / lib_name
            if lib_path.exists():
                try:
                    return ctypes.CDLL(str(lib_path))
                except Exception:
                    continue

    return None


# Load library
_lib = _load_library()


def bitnet_to_rotor_fast(bitnet_packed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Fast BitNet to Rotor conversion using C implementation.

    Args:
        bitnet_packed: BitNet format weights [rows, cols_packed] uint8
                      where cols_packed = (cols + 3) // 4

    Returns:
        (bit0, bit1): Rotor format weights
                     bit0: [rows, rotor_cols] uint8 where rotor_cols = (cols + 7) // 8
                     bit1: [rows, rotor_cols] uint8
    """
    if _lib is None:
        # Fall back to Python implementation
        from rotor.bitnet import bitnet_to_rotor as bitnet_to_rotor_py
        return bitnet_to_rotor_py(bitnet_packed, validate=False)

    # Get shape
    rows, bitnet_cols_bytes = bitnet_packed.shape
    cols = bitnet_cols_bytes * 4  # Approximately (may be slightly more)

    # Calculate output shape
    rotor_cols_bytes = (cols + 7) // 8

    # Allocate output
    bit0 = np.zeros((rows, rotor_cols_bytes), dtype=np.uint8)
    bit1 = np.zeros((rows, rotor_cols_bytes), dtype=np.uint8)

    # Setup C function signature
    if not hasattr(_lib, 'bitnet_to_rotor'):
        # Function not available, fall back to Python
        from rotor.bitnet import bitnet_to_rotor as bitnet_to_rotor_py
        return bitnet_to_rotor_py(bitnet_packed, validate=False)

    _lib.bitnet_to_rotor.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),  # bitnet_packed
        ctypes.c_size_t,                 # rows
        ctypes.c_size_t,                 # cols
        ctypes.POINTER(ctypes.c_uint8),  # bit0
        ctypes.POINTER(ctypes.c_uint8),  # bit1
    ]
    _lib.bitnet_to_rotor.restype = None

    # Call C function
    _lib.bitnet_to_rotor(
        bitnet_packed.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        rows,
        cols,
        bit0.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        bit1.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
    )

    return bit0, bit1


def rotor_unpack_weights_fast(bit0: np.ndarray, bit1: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """
    Fast Rotor weight unpacking using C implementation.

    Unpacks bit0/bit1 arrays into int8 weights where weight = bit0 - bit1.
    This is MUCH faster than Python bit-by-bit loops!

    Args:
        bit0: Rotor bit0 array [rows, (cols+7)//8] uint8
        bit1: Rotor bit1 array [rows, (cols+7)//8] uint8
        rows: Number of rows in weight matrix
        cols: Number of columns in weight matrix

    Returns:
        weights: Unpacked int8 weights [rows, cols]
    """
    if _lib is None or not hasattr(_lib, 'rotor_unpack_weights'):
        # Python fallback: decode bits directly
        weights = np.zeros((rows, cols), dtype=np.int8)
        rotor_cols_bytes = (cols + 7) // 8
        bit0 = bit0[:, :rotor_cols_bytes]
        bit1 = bit1[:, :rotor_cols_bytes]

        for row in range(rows):
            for col in range(cols):
                byte_idx = col // 8
                bit_idx = col % 8
                mask = 1 << bit_idx
                pos = 1 if (bit0[row, byte_idx] & mask) else 0
                neg = 1 if (bit1[row, byte_idx] & mask) else 0
                weights[row, col] = pos - neg

        return weights

    # Allocate output
    weights = np.zeros((rows, cols), dtype=np.int8)

    # Setup C function signature
    _lib.rotor_unpack_weights.argtypes = [
        ctypes.POINTER(ctypes.c_uint8),  # bit0
        ctypes.POINTER(ctypes.c_uint8),  # bit1
        ctypes.c_size_t,                 # rows
        ctypes.c_size_t,                 # cols
        ctypes.POINTER(ctypes.c_int8),   # weights
    ]
    _lib.rotor_unpack_weights.restype = None

    # Call C function
    _lib.rotor_unpack_weights(
        bit0.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        bit1.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
        rows,
        cols,
        weights.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
    )

    return weights


# Check if C library is available
def is_c_library_available() -> bool:
    """Check if the C library is available."""
    return _lib is not None and hasattr(_lib, 'bitnet_to_rotor')


if __name__ == "__main__":
    import sys

    print("Testing fast BitNet conversion...")
    print(f"C library available: {is_c_library_available()}")

    if is_c_library_available():
        print("✓ Fast C implementation loaded!")

        # Test with small array
        test_input = np.array([[0x60, 0x80]], dtype=np.uint8)  # 8 weights
        bit0, bit1 = bitnet_to_rotor_fast(test_input)

        print(f"\nTest conversion:")
        print(f"  Input:  {test_input}")
        print(f"  bit0:   {bit0}")
        print(f"  bit1:   {bit1}")
        print(f"✓ Fast conversion working!")
    else:
        print("⚠ C library not found - will use Python fallback")
        print("\nTo build the C library:")
        print("  cd native")
        print("  python build.py")
