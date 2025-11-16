"""Unit tests for optimized BitNet helpers."""

import sys
from pathlib import Path
import warnings

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rotor.bitnet_fast import (
    bitnet_to_rotor_fast,
    rotor_unpack_weights_fast,
    is_c_library_available,
)
from rotor.transformer import GatedFFN, TernaryLinear

_TEST_BITNET = np.array([
    [0b10_00_01_10, 0b00_10_01_00],
    [0b01_10_10_00, 0b10_00_00_01],
], dtype=np.uint8)
_EXPECTED_WEIGHTS = np.array([
    [1, -1, 0, 1, 0, -1, 1, 0],
    [0, 1, 1, -1, -1, 0, 0, 1],
], dtype=np.int8)


def test_bitnet_fast_conversion_matches_expected():
    """bitnet_to_rotor_fast should decode into the expected ternary weights."""
    bit0, bit1 = bitnet_to_rotor_fast(_TEST_BITNET)

    assert bit0.shape == bit1.shape == (2, 1)

    decoded = rotor_unpack_weights_fast(bit0, bit1, rows=2, cols=8)
    np.testing.assert_array_equal(decoded, _EXPECTED_WEIGHTS)


def test_rotor_unpack_fallback_handles_non_native_build():
    """rotor_unpack_weights_fast should still work without the native backend."""
    if is_c_library_available():
        pytest.skip("Fallback path not exercised when native backend is present")

    bit0, bit1 = bitnet_to_rotor_fast(_TEST_BITNET)
    decoded = rotor_unpack_weights_fast(bit0, bit1, rows=2, cols=8)
    np.testing.assert_array_equal(decoded, _EXPECTED_WEIGHTS)


def test_gated_ffn_silu_is_stable():
    """GatedFFN._silu must not emit overflow warnings on extreme inputs."""
    ffn = GatedFFN(d_model=64, d_ff=128)
    extreme = np.array([[-1000.0, -100.0, -10.0, 0.0, 10.0, 100.0, 1000.0]], dtype=np.float32)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        output = ffn._silu(extreme)

    assert output.shape == extreme.shape
    overflow = [w for w in caught if "overflow" in str(w.message).lower()]
    assert not overflow, f"Unexpected overflow warnings: {overflow}"


def test_ternary_linear_forward_uses_decoded_weights():
    """TernaryLinear forward pass should produce finite outputs from custom weights."""
    layer = TernaryLinear(in_features=8, out_features=4)
    layer.bit0 = np.array([0b10101010, 0b11001100, 0b11110000, 0b00001111], dtype=np.uint8)
    layer.bit1 = np.zeros_like(layer.bit0)
    layer.out_features = 4
    layer.in_features = 8
    layer.weight_shape = (4, 8)
    layer.scale = 1.0
    layer.weights_cache = layer._decode_weights()

    x = np.ones((1, 8), dtype=np.float32)
    y = layer.forward(x)

    assert y.shape == (1, 4)
    assert np.isfinite(y).all()
