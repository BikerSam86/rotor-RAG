"""
Quick validation test for optimized model.

Focuses on:
1. No overflow warnings ✓
2. C library working
3. Performance is good
"""

import sys
import io
from pathlib import Path
import time
import warnings

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

print("=" * 70)
print("Quick Validation Test Suite")
print("=" * 70)

# Test 1: C Library
print("\n[TEST 1] C Library Availability")
print("-" * 70)
from rotor.bitnet_fast import is_c_library_available
if is_c_library_available():
    print("✓ C library loaded")
else:
    print("✗ C library not available")
    sys.exit(1)

# Test 2: No Overflow in SiLU
print("\n[TEST 2] SiLU Numerical Stability (No Overflow)")
print("-" * 70)

from rotor.transformer import GatedFFN

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")

    ffn = GatedFFN(d_model=64, d_ff=128)

    # Test with extreme values
    extreme_values = np.array([-1000.0, -100.0, 0.0, 100.0, 1000.0], dtype=np.float32)
    result = ffn._silu(extreme_values)

    overflow_warnings = [warning for warning in w if 'overflow' in str(warning.message).lower()]

    if overflow_warnings:
        print(f"✗ Found {len(overflow_warnings)} overflow warning(s)!")
        for warning in overflow_warnings:
            print(f"  {warning.message}")
        sys.exit(1)
    else:
        print("✓ No overflow warnings")
        print(f"  Input range: [{extreme_values.min():.0f}, {extreme_values.max():.0f}]")
        print(f"  Output range: [{result.min():.4f}, {result.max():.4f}]")

# Test 3: Fast Conversion Performance
print("\n[TEST 3] Fast Conversion Performance")
print("-" * 70)

from rotor.bitnet_fast import bitnet_to_rotor_fast, rotor_unpack_weights_fast

# Realistic layer size
rows, cols = 2560, 2560
test_data = np.random.randint(0, 256, size=(rows, cols // 4), dtype=np.uint8)

print(f"  Testing {rows}×{cols} matrix")

# BitNet → Rotor
start = time.perf_counter()
bit0, bit1 = bitnet_to_rotor_fast(test_data)
convert_time = time.perf_counter() - start

# Rotor → int8
start = time.perf_counter()
weights = rotor_unpack_weights_fast(bit0, bit1, rows, cols)
unpack_time = time.perf_counter() - start

total_time = convert_time + unpack_time

print(f"  BitNet→Rotor: {convert_time*1000:.1f} ms")
print(f"  Rotor→int8:   {unpack_time*1000:.1f} ms")
print(f"  Total:        {total_time*1000:.1f} ms")

# Check if reasonably fast (< 1 second for a single layer)
if total_time > 1.0:
    print(f"✗ Too slow! Expected < 1s, got {total_time:.2f}s")
    sys.exit(1)
else:
    print(f"✓ Performance good!")

# Estimate full model time
layers = 30
matrices_per_layer = 7
estimated_time = layers * matrices_per_layer * total_time

print(f"\n  Estimated full model time:")
print(f"    {layers} layers × {matrices_per_layer} matrices = {layers * matrices_per_layer} total")
print(f"    ~{estimated_time:.0f} seconds ({estimated_time/60:.1f} minutes)")

if estimated_time > 120:  # More than 2 minutes
    print(f"  ⚠ Warning: Estimated time is high")
else:
    print(f"  ✓ Expected to load in < 2 minutes")

# Test 4: Forward Pass Smoke Test
print("\n[TEST 4] Forward Pass Smoke Test")
print("-" * 70)

from rotor.transformer import TernaryLinear

linear = TernaryLinear(in_features=512, out_features=256)

# Create random weights
n_bytes_0 = ((512 * 256) + 7) // 8
linear.bit0 = np.random.randint(0, 2, size=n_bytes_0, dtype=np.uint8)
linear.bit1 = np.random.randint(0, 2, size=n_bytes_0, dtype=np.uint8)
linear.out_features = 256
linear.in_features = 512
linear.weight_shape = (256, 512)
linear.scale = 1.0

# Decode (uses fast C function)
print("  Decoding weights...")
start = time.perf_counter()
linear.weights_cache = linear._decode_weights()
decode_time = (time.perf_counter() - start) * 1000

# Forward pass
test_input = np.random.randn(1, 512).astype(np.float32)
start = time.perf_counter()
output = linear.forward(test_input)
forward_time = (time.perf_counter() - start) * 1000

print(f"  Decode:  {decode_time:.2f} ms")
print(f"  Forward: {forward_time:.2f} ms")
print(f"  Output shape: {output.shape}")
print("✓ Forward pass successful")

# Summary
print("\n" + "=" * 70)
print("✅ ALL VALIDATION TESTS PASSED!")
print("=" * 70)
print("\n Summary:")
print("   ✓ C library working")
print("   ✓ No overflow warnings")
print("   ✓ Performance excellent")
print("   ✓ Forward pass working")
print(f"   ✓ Full model should load in ~{estimated_time:.0f}s")
print("\n🚀 System is optimized and ready!")
print("🌀 All ways, always!")
