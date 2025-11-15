"""
Core 2-bit ternary rotor operations

Encoding scheme:
    00 → 0  (neutral/rest)
    10 → +1 (forward/push)
    01 → -1 (reverse/pull)
    11 → ∅  (error/reserved)

Decode formula: value = bit0 - bit1
"""

import numpy as np
from typing import Tuple


class RotorCore:
    """
    Core operations for 2-bit ternary rotor encoding.

    This class provides the fundamental operations for encoding,
    decoding, and computing with ternary values using a 2-bit
    binary representation.
    """

    @staticmethod
    def encode(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Encode ternary values {-1, 0, +1} into 2-bit pairs.

        Args:
            values: Array of ternary values (-1, 0, or +1)

        Returns:
            Tuple of (bit0, bit1) arrays where value = bit0 - bit1

        Examples:
            >>> values = np.array([1, -1, 0, 1])
            >>> bit0, bit1 = RotorCore.encode(values)
            >>> bit0  # [1, 0, 0, 1]
            >>> bit1  # [0, 1, 0, 0]
        """
        values = np.asarray(values, dtype=np.int8)

        # Encoding logic:
        # +1 → bit0=1, bit1=0  (10)
        # -1 → bit0=0, bit1=1  (01)
        #  0 → bit0=0, bit1=0  (00)

        bit0 = (values == 1).astype(np.uint8)
        bit1 = (values == -1).astype(np.uint8)

        return bit0, bit1

    @staticmethod
    def decode(bit0: np.ndarray, bit1: np.ndarray) -> np.ndarray:
        """
        Decode 2-bit pairs back to ternary values.

        Args:
            bit0: First bit of each rotor
            bit1: Second bit of each rotor

        Returns:
            Array of ternary values {-1, 0, +1}

        Examples:
            >>> bit0 = np.array([1, 0, 0, 1])
            >>> bit1 = np.array([0, 1, 0, 0])
            >>> RotorCore.decode(bit0, bit1)  # [1, -1, 0, 1]
        """
        return bit0.astype(np.int8) - bit1.astype(np.int8)

    @staticmethod
    def pack(bit0: np.ndarray, bit1: np.ndarray) -> np.ndarray:
        """
        Pack bit pairs into compact uint8 representation.
        4 rotors per byte: [b7 b6][b5 b4][b3 b2][b1 b0]

        Args:
            bit0: First bits (must be 1D, length multiple of 4)
            bit1: Second bits (must be 1D, length multiple of 4)

        Returns:
            Packed uint8 array (length = input_length // 4)
        """
        bit0 = bit0.astype(np.uint8)
        bit1 = bit1.astype(np.uint8)

        n = len(bit0)
        assert n % 4 == 0, "Array length must be multiple of 4 for packing"

        # Reshape into groups of 4
        bit0_groups = bit0.reshape(-1, 4)
        bit1_groups = bit1.reshape(-1, 4)

        # Pack: each group of 4 rotors into 1 byte
        # [bit0_3, bit1_3, bit0_2, bit1_2, bit0_1, bit1_1, bit0_0, bit1_0]
        packed = (
            (bit0_groups[:, 3] << 7) |
            (bit1_groups[:, 3] << 6) |
            (bit0_groups[:, 2] << 5) |
            (bit1_groups[:, 2] << 4) |
            (bit0_groups[:, 1] << 3) |
            (bit1_groups[:, 1] << 2) |
            (bit0_groups[:, 0] << 1) |
            (bit1_groups[:, 0] << 0)
        )

        return packed.astype(np.uint8)

    @staticmethod
    def unpack(packed: np.ndarray, n_rotors: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Unpack uint8 array back into bit pairs.

        Args:
            packed: Packed uint8 array
            n_rotors: Number of rotors to extract

        Returns:
            Tuple of (bit0, bit1) arrays
        """
        n_bytes = (n_rotors + 3) // 4  # Ceiling division
        packed = packed[:n_bytes]

        # Extract bits for each position
        bit0_3 = (packed >> 7) & 1
        bit1_3 = (packed >> 6) & 1
        bit0_2 = (packed >> 5) & 1
        bit1_2 = (packed >> 4) & 1
        bit0_1 = (packed >> 3) & 1
        bit1_1 = (packed >> 2) & 1
        bit0_0 = (packed >> 1) & 1
        bit1_0 = (packed >> 0) & 1

        # Interleave back
        bit0 = np.stack([bit0_0, bit0_1, bit0_2, bit0_3], axis=1).ravel()[:n_rotors]
        bit1 = np.stack([bit1_0, bit1_1, bit1_2, bit1_3], axis=1).ravel()[:n_rotors]

        return bit0, bit1

    @staticmethod
    def dot(a_bit0: np.ndarray, a_bit1: np.ndarray,
            b_bit0: np.ndarray, b_bit1: np.ndarray) -> int:
        """
        Compute dot product of two rotor arrays.

        Uses the formula:
            (a0 - a1) · (b0 - b1) = a0·b0 - a0·b1 - a1·b0 + a1·b1

        Computed via bit operations:
            result = popcount(a0 & b0) - popcount(a0 & b1)
                   - popcount(a1 & b0) + popcount(a1 & b1)

        Args:
            a_bit0, a_bit1: First rotor array bits
            b_bit0, b_bit1: Second rotor array bits

        Returns:
            Dot product (integer)
        """
        a_bit0 = a_bit0.astype(bool)
        a_bit1 = a_bit1.astype(bool)
        b_bit0 = b_bit0.astype(bool)
        b_bit1 = b_bit1.astype(bool)

        pp = np.sum(a_bit0 & b_bit0)  # a0 AND b0
        pn = np.sum(a_bit0 & b_bit1)  # a0 AND b1
        np_term = np.sum(a_bit1 & b_bit0)  # a1 AND b0
        nn = np.sum(a_bit1 & b_bit1)  # a1 AND b1

        return int(pp - pn - np_term + nn)

    @staticmethod
    def matmul(W_bit0: np.ndarray, W_bit1: np.ndarray,
               x: np.ndarray) -> np.ndarray:
        """
        Matrix multiply: W @ x where W is ternary (rotor-encoded).

        Args:
            W_bit0: Weight matrix bit0 [out_features, in_features]
            W_bit1: Weight matrix bit1 [out_features, in_features]
            x: Input vector/matrix (will be quantized to ternary)

        Returns:
            Output vector/matrix (integer values, needs quantization)
        """
        # Quantize input to ternary
        x_ternary = np.sign(x).astype(np.int8)  # Simple sign quantization
        x_bit0, x_bit1 = RotorCore.encode(x_ternary)

        # Compute each output element via dot product
        out_features = W_bit0.shape[0]

        if x.ndim == 1:
            # Vector input
            output = np.zeros(out_features, dtype=np.int32)
            for i in range(out_features):
                output[i] = RotorCore.dot(
                    W_bit0[i], W_bit1[i],
                    x_bit0, x_bit1
                )
        else:
            # Batch input [batch_size, in_features]
            batch_size = x.shape[0]
            output = np.zeros((batch_size, out_features), dtype=np.int32)
            for b in range(batch_size):
                for i in range(out_features):
                    output[b, i] = RotorCore.dot(
                        W_bit0[i], W_bit1[i],
                        x_bit0[b], x_bit1[b]
                    )

        return output

    @staticmethod
    def negate(bit0: np.ndarray, bit1: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Negate rotor values (flip sign).

        Implementation: Simply swap bit0 and bit1

        Args:
            bit0, bit1: Rotor bit arrays

        Returns:
            Tuple of (new_bit0, new_bit1) representing negated values
        """
        return bit1.copy(), bit0.copy()

    @staticmethod
    def check_errors(bit0: np.ndarray, bit1: np.ndarray) -> np.ndarray:
        """
        Check for error states (11 pattern).

        Args:
            bit0, bit1: Rotor bit arrays

        Returns:
            Boolean array indicating error positions
        """
        return (bit0 == 1) & (bit1 == 1)


# Convenience functions
def encode_ternary(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Encode ternary values to 2-bit rotor format."""
    return RotorCore.encode(values)


def decode_ternary(bit0: np.ndarray, bit1: np.ndarray) -> np.ndarray:
    """Decode 2-bit rotor format back to ternary values."""
    return RotorCore.decode(bit0, bit1)
