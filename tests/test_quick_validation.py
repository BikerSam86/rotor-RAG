"""Smoke tests for the optimized BitNet pipeline."""

import sys
from pathlib import Path
import time
import warnings

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rotor.bitnet_fast import bitnet_to_rotor_fast, rotor_unpack_weights_fast, is_c_library_available
from rotor.transformer import GatedFFN, TernaryLinear


@pytest.mark.skipif(not is_c_library_available(), reason="native backend not built")
def test_native_backend_reported_available():
    """The quick validation suite requires the compiled native backend."""
    assert is_c_library_available()


def test_silu_extreme_inputs_are_stable():
    """SiLU implementation should not overflow across a wide dynamic range."""
    ffn = GatedFFN(d_model=64, d_ff=128)
    extreme = np.array([-1000.0, -100.0, 0.0, 100.0, 1000.0], dtype=np.float32)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        output = ffn._silu(extreme)

    overflow = [w for w in caught if "overflow" in str(w.message).lower()]
    assert not overflow
    assert output.shape == extreme.shape


def test_bitnet_fast_conversion_completes_quickly():
    """Conversion and unpacking should handle a moderate matrix quickly."""
    rows, cols = 256, 256
    packed = np.random.randint(0, 256, size=(rows, cols // 4), dtype=np.uint8)

    start = time.perf_counter()
    bit0, bit1 = bitnet_to_rotor_fast(packed)
    convert_time = time.perf_counter() - start

    start = time.perf_counter()
    weights = rotor_unpack_weights_fast(bit0, bit1, rows, cols)
    unpack_time = time.perf_counter() - start

    assert bit0.shape == bit1.shape == (rows, (cols + 7) // 8)
    assert weights.shape == (rows, cols)
    assert convert_time + unpack_time < 0.5, "Conversion took unexpectedly long"


def test_forward_pass_smoke():
    """A small ternary linear layer should run end-to-end without errors."""
    layer = TernaryLinear(in_features=128, out_features=32)
    n_bytes = ((layer.in_features * layer.out_features) + 7) // 8
    layer.bit0 = np.random.randint(0, 2, size=n_bytes, dtype=np.uint8)
    layer.bit1 = np.random.randint(0, 2, size=n_bytes, dtype=np.uint8)
    layer.weight_shape = (layer.out_features, layer.in_features)
    layer.out_features = 32
    layer.in_features = 128
    layer.scale = 1.0
    layer.weights_cache = layer._decode_weights()

    x = np.random.randn(4, layer.in_features).astype(np.float32)
    y = layer.forward(x)

    assert y.shape == (4, layer.out_features)
    assert np.isfinite(y).all()
