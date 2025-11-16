"""Unit tests for the sampling helpers used by TextGenerator."""

import numpy as np
import pytest

# Ensure local imports work when running tests directly
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rotor.generation import GreedySampling, TopKSampling, TopPSampling


@pytest.fixture(autouse=True)
def reset_numpy_rng():
    """Reset the legacy RNG so sampling tests remain deterministic."""
    state = np.random.get_state()
    np.random.seed(42)
    try:
        yield
    finally:
        np.random.set_state(state)


@pytest.fixture
def sample_logits():
    """Reusable set of logits for sampling tests."""
    rng = np.random.default_rng(0)
    return rng.normal(size=128).astype(np.float32)


def test_greedy_sampling_returns_argmax(sample_logits):
    greedy = GreedySampling()
    expected = int(np.argmax(sample_logits))

    assert greedy.sample(sample_logits) == expected
    # Greedy sampling should be deterministic regardless of calls
    assert greedy.sample(sample_logits) == expected


def test_topk_sampling_respects_candidate_pool(sample_logits):
    sampler = TopKSampling(k=10)
    np.random.seed(1)
    samples = [sampler.sample(sample_logits) for _ in range(50)]

    top_k = set(np.argpartition(sample_logits, -10)[-10:])
    assert set(samples).issubset({int(idx) for idx in top_k})
    # stochastic behaviour should produce more than one distinct token
    assert len(set(samples)) > 1


def test_topk_temperature_influences_diversity(sample_logits):
    sampler = TopKSampling(k=40)

    np.random.seed(2)
    hot_samples = [sampler.sample(sample_logits, temperature=2.0) for _ in range(100)]
    np.random.seed(2)
    cold_samples = [sampler.sample(sample_logits, temperature=0.1) for _ in range(100)]

    assert len(set(hot_samples)) >= len(set(cold_samples))


def test_topp_sampling_respects_probability_mass(sample_logits):
    sampler = TopPSampling(p=0.9)
    np.random.seed(3)
    samples = [sampler.sample(sample_logits) for _ in range(25)]

    probs = TopPSampling._softmax(sample_logits)
    sorted_indices = np.argsort(probs)[::-1]
    nucleus_size = np.searchsorted(np.cumsum(probs[sorted_indices]), 0.9) + 1
    nucleus = {int(idx) for idx in sorted_indices[:nucleus_size]}

    assert set(samples).issubset(nucleus)


def test_sampling_handles_extreme_logits():
    extreme_logits = np.array([-1000.0, 0.0, 1000.0], dtype=np.float32)
    sampler = TopKSampling(k=3)
    np.random.seed(4)

    sample = sampler.sample(extreme_logits, temperature=1.0)
    assert sample in {0, 1, 2}
