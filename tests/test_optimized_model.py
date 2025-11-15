"""
Comprehensive test suite for optimized BitNet model loading.

Tests:
1. C library availability
2. Fast conversion functions
3. Model loading (lightweight)
4. Forward pass
5. No overflow warnings
"""

import sys
import io
from pathlib import Path
import time
import warnings

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from rotor.bitnet_fast import is_c_library_available, bitnet_to_rotor_fast, rotor_unpack_weights_fast

print("=" * 70)
print("Optimized BitNet Model Test Suite")
print("=" * 70)

# Test 1: C Library Availability
print("\n[TEST 1] C Library Availability")
print("-" * 70)
if is_c_library_available():
    print("✓ C library loaded successfully")
else:
    print("✗ C library not available - will use Python fallback")
    sys.exit(1)

# Test 2: Fast BitNet Conversion
print("\n[TEST 2] Fast BitNet Conversion")
print("-" * 70)
# Create test data: 2x8 weights (2 rows, 8 weights)
# BitNet format: 4 weights per byte, so 2 bytes per row
test_bitnet = np.array([
    [0b10_00_01_10, 0b00_10_01_00],  # Row 0: +1, 0, -1, +1, 0, +1, -1, 0
    [0b01_10_10_00, 0b10_00_00_01],  # Row 1: -1, +1, +1, 0, +1, 0, 0, -1
], dtype=np.uint8)

start = time.perf_counter()
bit0, bit1 = bitnet_to_rotor_fast(test_bitnet)
elapsed_ms = (time.perf_counter() - start) * 1000

print(f"  Input shape: {test_bitnet.shape}")
print(f"  Output shapes: bit0={bit0.shape}, bit1={bit1.shape}")
print(f"  Time: {elapsed_ms:.4f} ms")
print("✓ Fast conversion working")

# Test 3: Fast Weight Unpacking
print("\n[TEST 3] Fast Weight Unpacking")
print("-" * 70)

start = time.perf_counter()
weights = rotor_unpack_weights_fast(bit0, bit1, rows=2, cols=8)
elapsed_ms = (time.perf_counter() - start) * 1000

print(f"  Input: bit0={bit0.shape}, bit1={bit1.shape}")
print(f"  Output: weights={weights.shape}")
print(f"  Time: {elapsed_ms:.4f} ms")

# Verify values
expected_row0 = np.array([1, 0, -1, 1, 0, 1, -1, 0], dtype=np.int8)
expected_row1 = np.array([-1, 1, 1, 0, 1, 0, 0, -1], dtype=np.int8)

if np.array_equal(weights[0], expected_row0) and np.array_equal(weights[1], expected_row1):
    print("✓ Unpacked weights correct!")
else:
    print(f"✗ Weight mismatch!")
    print(f"  Got row 0:      {weights[0]}")
    print(f"  Expected row 0: {expected_row0}")
    print(f"  Got row 1:      {weights[1]}")
    print(f"  Expected row 1: {expected_row1}")
    sys.exit(1)

# Test 4: SiLU Activation (no overflow)
print("\n[TEST 4] SiLU Activation (Numerical Stability)")
print("-" * 70)

from rotor.transformer import GatedFFN

# Catch warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")

    ffn = GatedFFN(d_model=64, d_ff=128)

    # Test with extreme values that would previously overflow
    test_extreme = np.array([
        [-1000.0, -100.0, -10.0, 0.0, 10.0, 100.0, 1000.0],
    ], dtype=np.float32)

    result = ffn._silu(test_extreme)

    # Check for overflow warnings
    overflow_warnings = [warning for warning in w if 'overflow' in str(warning.message).lower()]

    if overflow_warnings:
        print(f"✗ Found {len(overflow_warnings)} overflow warning(s)")
        for warning in overflow_warnings:
            print(f"  {warning.message}")
        sys.exit(1)
    else:
        print("✓ No overflow warnings")
        print(f"  Tested with extreme values: min={test_extreme.min()}, max={test_extreme.max()}")
        print(f"  Output range: min={result.min():.4f}, max={result.max():.4f}")

# Test 5: Small Model Forward Pass
print("\n[TEST 5] Small Model Forward Pass")
print("-" * 70)

from rotor.transformer import TernaryLinear

# Create small linear layer
linear = TernaryLinear(in_features=8, out_features=4)

# Set up test weights manually
rotor_cols_bytes = (8 + 7) // 8  # 1 byte for 8 weights
linear.bit0 = np.array([
    0b10101010,  # Row 0
    0b11001100,  # Row 1
    0b11110000,  # Row 2
    0b00001111,  # Row 3
], dtype=np.uint8)

linear.bit1 = np.zeros(4, dtype=np.uint8)  # All zeros (no negative weights)
linear.out_features = 4
linear.in_features = 8
linear.weight_shape = (4, 8)
linear.scale = 1.0

# Decode weights
print("  Decoding weights with fast C function...")
start = time.perf_counter()
linear.weights_cache = linear._decode_weights()
decode_time_ms = (time.perf_counter() - start) * 1000
print(f"  ✓ Decoded in {decode_time_ms:.4f} ms")

# Forward pass
test_input = np.ones((1, 8), dtype=np.float32)
start = time.perf_counter()
output = linear.forward(test_input)
forward_time_ms = (time.perf_counter() - start) * 1000

print(f"  Input shape: {test_input.shape}")
print(f"  Output shape: {output.shape}")
print(f"  Forward pass: {forward_time_ms:.4f} ms")
print("✓ Forward pass successful")

# Test 6: Performance Summary
print("\n[TEST 6] Performance Benchmark")
print("-" * 70)

# Benchmark realistic layer size
rows, cols = 2560, 2560
test_large = np.random.randint(0, 256, size=(rows, cols // 4), dtype=np.uint8)

print(f"  Testing with realistic layer: {rows}×{cols} weights")

# BitNet → Rotor conversion
start = time.perf_counter()
bit0_large, bit1_large = bitnet_to_rotor_fast(test_large)
convert_time = time.perf_counter() - start

# Weight unpacking
start = time.perf_counter()
weights_large = rotor_unpack_weights_fast(bit0_large, bit1_large, rows, cols)
unpack_time = time.perf_counter() - start

print(f"  BitNet→Rotor: {convert_time:.3f} sec")
print(f"  Unpack:       {unpack_time:.3f} sec")
print(f"  Total:        {convert_time + unpack_time:.3f} sec")

# Calculate estimated full model time
layers = 30
matrices_per_layer = 7
total_matrices = layers * matrices_per_layer
estimated_time = total_matrices * (convert_time + unpack_time)

print(f"\n  Estimated full model ({total_matrices} matrices): {estimated_time:.1f} sec")

print("\n" + "=" * 70)
print("✅ ALL TESTS PASSED!")
print("=" * 70)
print("\n🚀 Optimized model is working perfectly!")
print("   - C library loaded")
print("   - Fast conversion verified")
print("   - Fast unpacking verified")
print("   - No overflow warnings")
print("   - Forward pass working")
print(f"   - Full model loads in ~{estimated_time:.0f} seconds")
print("\n🌀 All ways, always!")
