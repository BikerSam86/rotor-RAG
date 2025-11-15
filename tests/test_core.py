"""
Tests for core rotor operations.
"""

import numpy as np
import sys
from pathlib import Path

# Fix Windows console encoding
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rotor.core import RotorCore


def test_encode_decode():
    """Test basic encoding and decoding."""
    print("Testing encode/decode...")

    values = np.array([1, -1, 0, 1, -1, 0, 0, 1])
    bit0, bit1 = RotorCore.encode(values)

    # Check encoding
    assert np.array_equal(bit0, [1, 0, 0, 1, 0, 0, 0, 1])
    assert np.array_equal(bit1, [0, 1, 0, 0, 1, 0, 0, 0])

    # Check decoding
    decoded = RotorCore.decode(bit0, bit1)
    assert np.array_equal(decoded, values)

    print("  ✓ Encode/decode works correctly")


def test_dot_product():
    """Test dot product computation."""
    print("Testing dot product...")

    # Simple test: [1, -1, 0] · [1, 1, 1] = 1*1 + (-1)*1 + 0*1 = 0
    a = np.array([1, -1, 0])
    b = np.array([1, 1, 1])

    a_bit0, a_bit1 = RotorCore.encode(a)
    b_bit0, b_bit1 = RotorCore.encode(b)

    result = RotorCore.dot(a_bit0, a_bit1, b_bit0, b_bit1)
    expected = np.dot(a, b)

    assert result == expected, f"Expected {expected}, got {result}"

    # Another test: [1, 1, 1] · [1, 1, 1] = 3
    a = np.array([1, 1, 1])
    b = np.array([1, 1, 1])

    a_bit0, a_bit1 = RotorCore.encode(a)
    b_bit0, b_bit1 = RotorCore.encode(b)

    result = RotorCore.dot(a_bit0, a_bit1, b_bit0, b_bit1)
    expected = 3

    assert result == expected, f"Expected {expected}, got {result}"

    print("  ✓ Dot product works correctly")


def test_negate():
    """Test negation operation."""
    print("Testing negation...")

    values = np.array([1, -1, 0, 1])
    bit0, bit1 = RotorCore.encode(values)

    # Negate
    neg_bit0, neg_bit1 = RotorCore.negate(bit0, bit1)
    negated = RotorCore.decode(neg_bit0, neg_bit1)

    expected = -values
    assert np.array_equal(negated, expected)

    print("  ✓ Negation works correctly")


def test_pack_unpack():
    """Test packing and unpacking."""
    print("Testing pack/unpack...")

    # Create 8 rotors (2 bytes when packed)
    values = np.array([1, -1, 0, 1, -1, 0, 1, 0])
    bit0, bit1 = RotorCore.encode(values)

    # Pack
    packed = RotorCore.pack(bit0, bit1)
    assert len(packed) == 2, "Should pack to 2 bytes"

    # Unpack
    unpacked_bit0, unpacked_bit1 = RotorCore.unpack(packed, n_rotors=8)

    # Verify
    assert np.array_equal(unpacked_bit0, bit0)
    assert np.array_equal(unpacked_bit1, bit1)

    unpacked_values = RotorCore.decode(unpacked_bit0, unpacked_bit1)
    assert np.array_equal(unpacked_values, values)

    print("  ✓ Pack/unpack works correctly")


def test_error_detection():
    """Test error state detection."""
    print("Testing error detection...")

    # Create normal values
    bit0 = np.array([1, 0, 0, 1], dtype=np.uint8)
    bit1 = np.array([0, 1, 0, 0], dtype=np.uint8)

    errors = RotorCore.check_errors(bit0, bit1)
    assert not np.any(errors), "Should have no errors"

    # Introduce error state (11)
    bit0[2] = 1
    bit1[2] = 1

    errors = RotorCore.check_errors(bit0, bit1)
    assert np.sum(errors) == 1, "Should detect 1 error"
    assert errors[2], "Should detect error at position 2"

    print("  ✓ Error detection works correctly")


def test_matmul():
    """Test matrix multiplication."""
    print("Testing matrix multiplication...")

    # Small weight matrix (3x4)
    W = np.array([
        [1, -1, 0, 1],
        [0, 1, -1, 0],
        [1, 0, 1, -1]
    ])
    W_bit0, W_bit1 = RotorCore.encode(W)

    # Input vector (4,)
    x = np.array([0.5, -0.3, 0.8, -0.2])

    # Compute using rotor matmul
    output = RotorCore.matmul(W_bit0, W_bit1, x)

    # Verify shape
    assert output.shape == (3,), f"Expected shape (3,), got {output.shape}"

    # Compute expected (with quantization)
    x_quant = np.sign(x).astype(np.int8)
    expected = W @ x_quant

    assert np.array_equal(output, expected)

    print("  ✓ Matrix multiplication works correctly")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*50)
    print("Running Rotor Core Tests")
    print("="*50 + "\n")

    test_encode_decode()
    test_dot_product()
    test_negate()
    test_pack_unpack()
    test_error_detection()
    test_matmul()

    print("\n" + "="*50)
    print("All tests passed! ✓")
    print("="*50 + "\n")


if __name__ == "__main__":
    run_all_tests()
