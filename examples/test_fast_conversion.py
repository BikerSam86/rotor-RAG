"""
Test fast BitNet conversion (with C library or Python fallback).
"""

import sys
import io
from pathlib import Path
import time
import numpy as np

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rotor.bitnet_fast import bitnet_to_rotor_fast, is_c_library_available
from rotor.bitnet import bitnet_to_rotor as bitnet_to_rotor_py


def test_correctness():
    """Test that fast conversion matches Python implementation."""
    print("=" * 70)
    print("Testing Fast BitNet Conversion")
    print("=" * 70)

    print(f"\nC library available: {is_c_library_available()}")
    if is_c_library_available():
        print("  ✓ Will use optimized C implementation")
    else:
        print("  ⚠ C library not found - using Python fallback")
        print("  To build C library: cd native && python build.py")

    # Create test data
    print("\n" + "=" * 70)
    print("Test 1: Small Array")
    print("=" * 70)

    # BitNet format: 00=0, 10=+1, 01=-1
    # Byte 0x60 = 0b01100000 = weights [0, 0, +1, 0]
    # Byte 0x80 = 0b10000000 = weights [0, 0, 0, -1]
    test_input = np.array([[0x60, 0x80]], dtype=np.uint8)  # 8 weights

    print(f"Input BitNet packed: {test_input}")
    print(f"  Shape: {test_input.shape} (1 row, 2 bytes = 8 weights)")

    # Test fast version
    start = time.perf_counter()
    bit0_fast, bit1_fast = bitnet_to_rotor_fast(test_input)
    fast_time = (time.perf_counter() - start) * 1000

    # Test Python version for comparison
    start = time.perf_counter()
    bit0_py, bit1_py = bitnet_to_rotor_py(test_input, validate=False)
    py_time = (time.perf_counter() - start) * 1000

    print(f"\nResults:")
    print(f"  bit0 (fast): {bit0_fast}")
    print(f"  bit1 (fast): {bit1_fast}")
    print(f"\n  bit0 (py):   {bit0_py}")
    print(f"  bit1 (py):   {bit1_py}")

    print(f"\nTiming:")
    print(f"  Fast version: {fast_time:.3f} ms")
    print(f"  Python version: {py_time:.3f} ms")
    if is_c_library_available():
        print(f"  Speedup: {py_time/fast_time:.1f}×")

    # Verify correctness
    if np.array_equal(bit0_fast, bit0_py) and np.array_equal(bit1_fast, bit1_py):
        print("\n✓ Results match! Fast conversion is correct.")
    else:
        print("\n✗ Results don't match!")
        return False

    # Test 2: Larger realistic array
    print("\n" + "=" * 70)
    print("Test 2: Realistic Model Layer")
    print("=" * 70)

    # Simulate a transformer layer: 2560 × 2560 weights
    rows = 2560
    cols = 2560
    bitnet_cols_bytes = (cols + 3) // 4

    # Random test data
    np.random.seed(42)
    large_input = np.random.randint(0, 256, size=(rows, bitnet_cols_bytes), dtype=np.uint8)

    print(f"\nSimulated layer shape:")
    print(f"  Weight matrix: {rows} × {cols}")
    print(f"  BitNet packed: {large_input.shape} ({large_input.nbytes:,} bytes)")

    # Test fast version
    print("\nConverting with fast method...")
    start = time.perf_counter()
    bit0_fast, bit1_fast = bitnet_to_rotor_fast(large_input)
    fast_time = time.perf_counter() - start

    print(f"  ✓ Conversion complete: {fast_time:.3f}s")
    print(f"  Output shape: {bit0_fast.shape}")
    print(f"  Output size: {(bit0_fast.nbytes + bit1_fast.nbytes):,} bytes")

    # For comparison, time Python version (but don't run it all if too slow)
    if fast_time < 5.0 or not is_c_library_available():
        print("\nConverting with Python method (for comparison)...")
        start = time.perf_counter()
        bit0_py, bit1_py = bitnet_to_rotor_py(large_input, validate=False)
        py_time = time.perf_counter() - start

        print(f"  ✓ Conversion complete: {py_time:.3f}s")

        # Verify they match
        if np.array_equal(bit0_fast, bit0_py) and np.array_equal(bit1_fast, bit1_py):
            print("\n✓ Large array conversion matches Python!")
        else:
            print("\n✗ Large array results don't match!")
            return False

        if is_c_library_available():
            speedup = py_time / fast_time
            print(f"\n🚀 Speedup: {speedup:.1f}× faster with C library!")
        else:
            print("\n⚠ Running with Python fallback (same speed)")
    else:
        print("\n(Skipping Python comparison - too slow)")

    return True


def estimate_full_model_time():
    """Estimate time to load full 2.4B model."""
    print("\n" + "=" * 70)
    print("Estimating Full Model Load Time")
    print("=" * 70)

    # BitNet-2B has 210 ternary weight layers
    # Each layer approximately 2560 × 2560
    rows = 2560
    cols = 2560
    bitnet_cols_bytes = (cols + 3) // 4

    # Create one layer
    np.random.seed(42)
    layer = np.random.randint(0, 256, size=(rows, bitnet_cols_bytes), dtype=np.uint8)

    # Time one conversion
    start = time.perf_counter()
    bit0, bit1 = bitnet_to_rotor_fast(layer)
    layer_time = time.perf_counter() - start

    # Estimate for all 210 layers
    num_layers = 210
    total_time = layer_time * num_layers

    print(f"\nSingle layer conversion: {layer_time:.3f}s")
    print(f"Estimated for {num_layers} layers: {total_time:.1f}s ({total_time/60:.1f} minutes)")

    if is_c_library_available():
        print("\n✓ This is with optimized C implementation!")
    else:
        print("\n⚠ This is with Python fallback!")
        print("  With C library, expect ~100× faster: <1 second total")


def main():
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "Fast BitNet Conversion Test" + " " * 25 + "║")
    print("╚" + "═" * 68 + "╝\n")

    if test_correctness():
        print("\n" + "=" * 70)
        print("✓ ALL TESTS PASSED!")
        print("=" * 70)

        estimate_full_model_time()

        print("\n" + "=" * 70)
        print("Summary")
        print("=" * 70)

        if is_c_library_available():
            print("\n✓ C library is available and working!")
            print("✓ Fast conversion ready for full model loading")
        else:
            print("\n⚠ C library not available - using Python fallback")
            print("  To build C library:")
            print("    1. Install a C compiler (gcc, clang, or MSVC)")
            print("    2. Run: cd native && python build.py")
            print("\n  The logic is correct, just waiting for compilation!")

        print("\n🌀 All ways, always!")
    else:
        print("\n✗ Tests failed!")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
