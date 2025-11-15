"""
BitNet compatibility layer.

Provides conversion between Microsoft BitNet format and our Rotor format.
"""

import numpy as np
from typing import Tuple, Optional
import warnings


class BitNetFormat:
    """
    Microsoft BitNet weight encoding.

    Encoding (2 bits per weight):
        00 = 0
        10 = +1
        01 = -1
        11 = error/unused

    Packing: 4 weights per byte
    """

    # Encoding constants
    ZERO = 0b00
    POS = 0b10
    NEG = 0b01
    ERR = 0b11

    @staticmethod
    def decode_weight(packed_byte: int, index: int) -> int:
        """
        Extract single weight from packed byte.

        Args:
            packed_byte: Packed byte containing 4 weights
            index: Weight index (0-3)

        Returns:
            Weight value: -1, 0, or +1
        """
        bits = (packed_byte >> (index * 2)) & 0b11

        if bits == BitNetFormat.ZERO:
            return 0
        elif bits == BitNetFormat.POS:
            return 1
        elif bits == BitNetFormat.NEG:
            return -1
        else:
            warnings.warn(f"BitNet error code (0b11) at index {index}")
            return 0

    @staticmethod
    def encode_weight(value: int) -> int:
        """
        Encode weight value to 2-bit representation.

        Args:
            value: -1, 0, or +1

        Returns:
            2-bit encoding
        """
        if value == 0:
            return BitNetFormat.ZERO
        elif value > 0:
            return BitNetFormat.POS
        elif value < 0:
            return BitNetFormat.NEG
        else:
            return BitNetFormat.ZERO


def bitnet_to_rotor(
    bitnet_packed: np.ndarray,
    validate: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert BitNet packed weights to Rotor format.

    Args:
        bitnet_packed: BitNet weights, shape [m, n//4] or [n//4]
                      dtype uint8, 4 weights per byte
        validate: Whether to validate no error codes present

    Returns:
        (bit0, bit1): Rotor format bit arrays
                      bit0: +1 indicators
                      bit1: -1 indicators
                      Shape: [m, (n+7)//8] or [(n+7)//8]

    Example:
        >>> bitnet = np.array([[0b00_10_01_00]], dtype=np.uint8)  # [0, -1, +1, 0]
        >>> bit0, bit1 = bitnet_to_rotor(bitnet)
        >>> # bit0 has bit set for +1, bit1 has bit set for -1
    """
    # Handle both 1D and 2D
    is_1d = (bitnet_packed.ndim == 1)
    if is_1d:
        bitnet_packed = bitnet_packed.reshape(1, -1)

    m, n_packed = bitnet_packed.shape
    n_weights = n_packed * 4

    # Allocate output (8 weights per byte in our format)
    n_bytes_out = (n_weights + 7) // 8
    bit0 = np.zeros((m, n_bytes_out), dtype=np.uint8)
    bit1 = np.zeros((m, n_bytes_out), dtype=np.uint8)

    error_count = 0

    for row in range(m):
        for i in range(n_weights):
            # Extract from BitNet format
            byte_idx = i // 4
            if byte_idx >= n_packed:
                break

            bit_pos = (i % 4) * 2
            weight_bits = (bitnet_packed[row, byte_idx] >> bit_pos) & 0b11

            # Decode
            if weight_bits == BitNetFormat.ZERO:
                value = 0
            elif weight_bits == BitNetFormat.POS:
                value = 1
            elif weight_bits == BitNetFormat.NEG:
                value = -1
            else:
                value = 0
                error_count += 1

            # Encode to Rotor format
            out_byte = i // 8
            out_bit = i % 8

            if value == 1:
                bit0[row, out_byte] |= (1 << out_bit)
            elif value == -1:
                bit1[row, out_byte] |= (1 << out_bit)

    if validate and error_count > 0:
        warnings.warn(f"Found {error_count} error codes (0b11) in BitNet weights")

    if is_1d:
        bit0 = bit0.reshape(-1)
        bit1 = bit1.reshape(-1)

    return bit0, bit1


def rotor_to_bitnet(
    bit0: np.ndarray,
    bit1: np.ndarray
) -> np.ndarray:
    """
    Convert Rotor format to BitNet packed weights.

    Args:
        bit0: +1 indicators, shape [m, n_bytes] or [n_bytes]
        bit1: -1 indicators, same shape as bit0

    Returns:
        bitnet_packed: BitNet format, shape [m, n_weights//4] or [n_weights//4]
                      4 weights per byte

    Example:
        >>> # Create Rotor format: [+1, 0, -1, 0]
        >>> bit0 = np.array([[0b00000001]], dtype=np.uint8)  # +1 at position 0
        >>> bit1 = np.array([[0b00000100]], dtype=np.uint8)  # -1 at position 2
        >>> bitnet = rotor_to_bitnet(bit0, bit1)
        >>> # bitnet = [[0b00_01_00_10]] (4 weights in 1 byte)
    """
    assert bit0.shape == bit1.shape, "bit0 and bit1 must have same shape"

    # Handle both 1D and 2D
    is_1d = (bit0.ndim == 1)
    if is_1d:
        bit0 = bit0.reshape(1, -1)
        bit1 = bit1.reshape(1, -1)

    m, n_bytes = bit0.shape
    n_weights = n_bytes * 8

    # Output: 4 weights per byte
    n_packed = (n_weights + 3) // 4
    bitnet_packed = np.zeros((m, n_packed), dtype=np.uint8)

    for row in range(m):
        for i in range(n_weights):
            # Decode from Rotor format
            in_byte = i // 8
            in_bit = i % 8

            b0 = (bit0[row, in_byte] >> in_bit) & 1
            b1 = (bit1[row, in_byte] >> in_bit) & 1

            # Value: +1, 0, or -1
            value = int(b0) - int(b1)

            # Encode to BitNet
            if value == 0:
                weight_bits = BitNetFormat.ZERO
            elif value == 1:
                weight_bits = BitNetFormat.POS
            elif value == -1:
                weight_bits = BitNetFormat.NEG
            else:
                weight_bits = BitNetFormat.ERR

            # Pack into output
            out_byte = i // 4
            if out_byte >= n_packed:
                break

            out_pos = (i % 4) * 2
            bitnet_packed[row, out_byte] |= (weight_bits << out_pos)

    if is_1d:
        bitnet_packed = bitnet_packed.reshape(-1)

    return bitnet_packed


def decode_bitnet_array(bitnet_packed: np.ndarray) -> np.ndarray:
    """
    Fully decode BitNet weights to int8 array.

    Args:
        bitnet_packed: BitNet format [m, n//4] or [n//4]

    Returns:
        weights: Decoded weights [m, n] or [n], values in {-1, 0, +1}

    Example:
        >>> bitnet = np.array([[0b00_10_01_00]], dtype=np.uint8)
        >>> weights = decode_bitnet_array(bitnet)
        >>> # weights = [[0, -1, +1, 0]]
    """
    is_1d = (bitnet_packed.ndim == 1)
    if is_1d:
        bitnet_packed = bitnet_packed.reshape(1, -1)

    m, n_packed = bitnet_packed.shape
    n_weights = n_packed * 4

    weights = np.zeros((m, n_weights), dtype=np.int8)

    for row in range(m):
        for i in range(n_weights):
            byte_idx = i // 4
            value = BitNetFormat.decode_weight(bitnet_packed[row, byte_idx], i % 4)
            weights[row, i] = value

    if is_1d:
        weights = weights.reshape(-1)

    return weights


def encode_bitnet_array(weights: np.ndarray) -> np.ndarray:
    """
    Encode int8 weights to BitNet format.

    Args:
        weights: Ternary weights [m, n] or [n], values in {-1, 0, +1}

    Returns:
        bitnet_packed: BitNet format [m, n//4] or [n//4]

    Example:
        >>> weights = np.array([[0, -1, +1, 0]], dtype=np.int8)
        >>> bitnet = encode_bitnet_array(weights)
        >>> # bitnet = [[0b00_10_01_00]]
    """
    is_1d = (weights.ndim == 1)
    if is_1d:
        weights = weights.reshape(1, -1)

    m, n = weights.shape
    n_packed = (n + 3) // 4

    bitnet_packed = np.zeros((m, n_packed), dtype=np.uint8)

    for row in range(m):
        for i in range(n):
            value = weights[row, i]
            weight_bits = BitNetFormat.encode_weight(value)

            byte_idx = i // 4
            bit_pos = (i % 4) * 2
            bitnet_packed[row, byte_idx] |= (weight_bits << bit_pos)

    if is_1d:
        bitnet_packed = bitnet_packed.reshape(-1)

    return bitnet_packed


def validate_conversion(
    bitnet_packed: np.ndarray,
    tolerance: float = 0
) -> bool:
    """
    Validate BitNet ↔ Rotor conversion is lossless.

    Args:
        bitnet_packed: Original BitNet weights
        tolerance: Allowed difference (should be 0)

    Returns:
        True if conversion is perfect

    Example:
        >>> bitnet = np.array([[0b00_10_01_00]], dtype=np.uint8)
        >>> assert validate_conversion(bitnet)
    """
    # Convert to Rotor and back
    bit0, bit1 = bitnet_to_rotor(bitnet_packed, validate=False)
    bitnet_reconstructed = rotor_to_bitnet(bit0, bit1)

    # Compare (only compare valid bytes)
    n_valid = bitnet_packed.shape[-1]
    match = np.allclose(
        bitnet_packed[..., :n_valid],
        bitnet_reconstructed[..., :n_valid],
        atol=tolerance
    )

    if not match:
        diff = np.sum(bitnet_packed != bitnet_reconstructed)
        warnings.warn(f"Conversion mismatch: {diff} bytes differ")

    return match


# Example usage and tests
if __name__ == "__main__":
    print("Testing BitNet ↔ Rotor conversion...")

    # Test 1: Simple conversion
    print("\nTest 1: Simple 4-weight conversion")
    bitnet = np.array([[0b00_10_01_00]], dtype=np.uint8)  # [0, -1, +1, 0]
    print(f"BitNet: {bin(bitnet[0, 0])}")

    bit0, bit1 = bitnet_to_rotor(bitnet)
    print(f"Rotor bit0: {bin(bit0[0, 0])}")  # +1 at position 2
    print(f"Rotor bit1: {bin(bit1[0, 0])}")  # -1 at position 1

    # Decode to verify
    decoded = decode_bitnet_array(bitnet)
    print(f"Decoded: {decoded[0]}")  # [0, -1, +1, 0]

    # Convert back
    bitnet_reconstructed = rotor_to_bitnet(bit0, bit1)
    print(f"Reconstructed: {bin(bitnet_reconstructed[0, 0])}")
    print(f"Original shape: {bitnet.shape}, Reconstructed shape: {bitnet_reconstructed.shape}")
    print(f"Original: {bitnet}")
    print(f"Reconstructed: {bitnet_reconstructed}")
    print(f"Equal: {np.array_equal(bitnet, bitnet_reconstructed)}")
    # Only compare the valid bytes
    assert np.array_equal(bitnet[..., :bitnet.shape[-1]], bitnet_reconstructed[..., :bitnet.shape[-1]]), "Conversion failed!"
    print("✓ Test 1 passed")

    # Test 2: Larger matrix
    print("\nTest 2: Random matrix conversion")
    np.random.seed(42)
    weights = np.random.choice([-1, 0, 1], size=(64, 256), p=[0.3, 0.4, 0.3])
    weights = weights.astype(np.int8)

    bitnet = encode_bitnet_array(weights)
    print(f"Encoded shape: {bitnet.shape}")  # (64, 64) - 4 weights per byte

    bit0, bit1 = bitnet_to_rotor(bitnet)
    print(f"Rotor shape: bit0={bit0.shape}, bit1={bit1.shape}")  # (64, 32) - 8 weights per byte

    # Verify
    assert validate_conversion(bitnet), "Large matrix conversion failed!"
    print("✓ Test 2 passed")

    # Test 3: Performance comparison
    print("\nTest 3: Performance comparison")
    import time

    n_trials = 1000
    m, n = 256, 512

    weights = np.random.choice([-1, 0, 1], size=(m, n)).astype(np.int8)
    bitnet = encode_bitnet_array(weights)

    # Time conversion
    start = time.perf_counter()
    for _ in range(n_trials):
        bit0, bit1 = bitnet_to_rotor(bitnet, validate=False)
    elapsed = (time.perf_counter() - start) / n_trials * 1000

    print(f"BitNet → Rotor conversion: {elapsed:.3f} ms")
    print(f"Memory: BitNet {bitnet.nbytes} bytes → Rotor {bit0.nbytes + bit1.nbytes} bytes")
    print(f"Ratio: {(bit0.nbytes + bit1.nbytes) / bitnet.nbytes:.2f}×")

    print("\n✓ All tests passed!")
    print("\nBitNet compatibility layer ready!")
