"""
Debug BitNet format to understand bit ordering.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from rotor.bitnet_fast import bitnet_to_rotor_fast, rotor_unpack_weights_fast

print("=" * 70)
print("BitNet Format Debugging")
print("=" * 70)

# Simple test: 1 row, 4 weights
# Let's encode: +1, 0, -1, 0
# BitNet encoding: 00=0, 10=+1, 01=-1
# So we want: 10, 00, 01, 00 (from LSB to MSB)
# Byte: 00_01_00_10 = 0b00010010 = 0x12

test_byte = 0b00_01_00_10
print(f"\nTest byte: 0b{test_byte:08b} (0x{test_byte:02x})")
print("Expected weights: [+1, 0, -1, 0]")
print()

# Create BitNet array
bitnet_packed = np.array([[test_byte]], dtype=np.uint8)
print(f"BitNet packed shape: {bitnet_packed.shape}")

# Convert to Rotor
bit0, bit1 = bitnet_to_rotor_fast(bitnet_packed)
print(f"Rotor bit0: {bit0}")
print(f"Rotor bit1: {bit1}")
print(f"Rotor bit0 binary: {format(bit0[0, 0], '08b')}")
print(f"Rotor bit1 binary: {format(bit1[0, 0], '08b')}")

# Unpack weights
weights = rotor_unpack_weights_fast(bit0, bit1, rows=1, cols=4)
print(f"\nUnpacked weights: {weights[0]}")

# Manual decode to understand
print("\nManual bit-by-bit decode:")
for i in range(4):
    two_bits = (test_byte >> (i * 2)) & 0b11
    bit0_val = (bit0[0, 0] >> i) & 1
    bit1_val = (bit1[0, 0] >> i) & 1
    weight_val = bit0_val - bit1_val

    print(f"  Position {i}:")
    print(f"    BitNet bits: {two_bits:02b} (value {two_bits})")
    print(f"    Rotor bit0: {bit0_val}, bit1: {bit1_val}")
    print(f"    Weight: {bit0_val} - {bit1_val} = {weight_val}")

print("\n" + "=" * 70)
