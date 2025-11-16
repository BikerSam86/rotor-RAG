"""Integration-style tests for the generation helpers."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rotor.generation import TextGenerator, GreedySampling, TopKSampling, TopPSampling


class MockTokenizer:
    def __init__(self):
        self.vocab_size = 100
        self.bos_token_id = 0
        self.eos_token_id = 1
        self.pad_token_id = 2

    def encode(self, text, add_special_tokens=True, return_numpy=True):
        tokens = [self.bos_token_id] if add_special_tokens else []
        tokens.extend([10, 20, 30])
        array = np.array(tokens, dtype=np.int64)
        return array if return_numpy else tokens

    def decode(self, token_ids, skip_special_tokens=True):
        if isinstance(token_ids, np.ndarray):
            token_ids = token_ids.tolist()
        if skip_special_tokens:
            token_ids = [t for t in token_ids if t not in (self.bos_token_id, self.eos_token_id)]
        return "decoded_" + "_".join(str(t) for t in token_ids)


class MockModel:
    def __init__(self, vocab_size=100, force_argmax=True):
        self.vocab_size = vocab_size
        self.force_argmax = force_argmax
        self.call_count = 0
        self.last_kwargs = []

    def forward(
        self,
        input_ids,
        return_logits: bool = True,
        past_kv_cache=None,
        use_cache: bool = False,
    ):
        self.call_count += 1
        self.last_kwargs.append({
            "use_cache": use_cache,
            "return_logits": return_logits,
            "has_cache": past_kv_cache is not None,
        })

        batch_size, seq_len = input_ids.shape
        logits = np.random.randn(batch_size, seq_len, self.vocab_size).astype(np.float32)
        if self.force_argmax:
            logits[:, :, 42] = 10.0  # ensure deterministic greedy choice
        kv_cache = {"length": seq_len}
        return logits, kv_cache


@pytest.fixture
def tokenizer():
    return MockTokenizer()


@pytest.fixture
def mock_model():
    return MockModel()


def test_generator_runs_greedy_sampling(tokenizer, mock_model):
    generator = TextGenerator(mock_model, tokenizer, sampling_strategy=GreedySampling(), max_length=10)
    output = generator.generate("Hello", max_new_tokens=3)

    assert isinstance(output, str)
    assert mock_model.call_count == 3


def test_greedy_sampling_is_deterministic(tokenizer):
    model = MockModel()
    generator = TextGenerator(model, tokenizer, sampling_strategy=GreedySampling(), max_length=10)
    first = generator.generate("test", max_new_tokens=2)
    second = generator.generate("test", max_new_tokens=2)
    assert first == second


def test_top_k_sampling_produces_variation(tokenizer):
    model = MockModel(force_argmax=False)
    sampler = TopKSampling(k=5)
    generator = TextGenerator(model, tokenizer, sampling_strategy=sampler, temperature=1.0)
    results = set()
    for seed in range(5):
        np.random.seed(seed)
        results.add(generator.generate("prompt", max_new_tokens=2))
    assert len(results) > 1


def test_top_p_sampling_accepts_temperature(tokenizer):
    model = MockModel()
    sampler = TopPSampling(p=0.9)
    generator = TextGenerator(model, tokenizer, sampling_strategy=sampler, temperature=0.5)
    text = generator.generate("prompt", max_new_tokens=2)
    assert isinstance(text, str)


def test_on_token_callback_receives_tokens(tokenizer):
    model = MockModel()
    generator = TextGenerator(model, tokenizer, sampling_strategy=GreedySampling())
    tokens = []

    def on_token(token_text):
        tokens.append(token_text)

    generator.generate("prompt", max_new_tokens=3, callback=on_token)
    assert len(tokens) == 3
