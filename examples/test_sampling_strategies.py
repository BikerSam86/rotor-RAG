"""
Test different sampling strategies to avoid repetitive "adooadoo" outputs.
Ternary models can be sensitive to sampling parameters.
"""

import sys
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rotor.bitnet_model import load_bitnet_model
from rotor.tokenizer import BitNetTokenizer
from rotor.generation import TextGenerator, GreedySampling, SamplingStrategy
import numpy as np

print("=" * 70)
print("SAMPLING STRATEGY TEST - Fighting 'adooadoo' outputs")
print("=" * 70)

model_dir = "C:/Users/samho/Desktop/BitNet-2B-model"

print("\nLoading model and tokenizer...")
model = load_bitnet_model(model_dir, use_gpu=False)
tokenizer = BitNetTokenizer(model_dir)

# Custom sampling strategies
class TopKSampling(SamplingStrategy):
    def __init__(self, k=50, temperature=1.0):
        self.k = k
        self.temperature = temperature

    def sample(self, logits: np.ndarray) -> int:
        # Apply temperature
        logits = logits / self.temperature

        # Get top-k indices
        top_k_indices = np.argsort(logits)[-self.k:]
        top_k_logits = logits[top_k_indices]

        # Softmax
        exp_logits = np.exp(top_k_logits - np.max(top_k_logits))
        probs = exp_logits / np.sum(exp_logits)

        # Sample
        sampled_idx = np.random.choice(len(top_k_indices), p=probs)
        return int(top_k_indices[sampled_idx])


class NucleusSampling(SamplingStrategy):
    def __init__(self, p=0.9, temperature=1.0):
        self.p = p
        self.temperature = temperature

    def sample(self, logits: np.ndarray) -> int:
        # Apply temperature
        logits = logits / self.temperature

        # Softmax
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        # Sort by probability
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]

        # Find nucleus (cumulative prob >= p)
        cumsum = np.cumsum(sorted_probs)
        cutoff_idx = np.searchsorted(cumsum, self.p) + 1

        # Sample from nucleus
        nucleus_indices = sorted_indices[:cutoff_idx]
        nucleus_probs = sorted_probs[:cutoff_idx]
        nucleus_probs = nucleus_probs / nucleus_probs.sum()

        sampled_idx = np.random.choice(len(nucleus_indices), p=nucleus_probs)
        return int(nucleus_indices[sampled_idx])


prompt = "The future of AI"

print(f"\nPrompt: '{prompt}'")
print("\n" + "=" * 70)

# Test 1: Greedy (deterministic - might cause repetition)
print("TEST 1: Greedy Sampling (deterministic)")
print("-" * 70)
gen = TextGenerator(model, tokenizer, sampling_strategy=GreedySampling(), use_cache=True)
text = gen.generate(prompt, max_new_tokens=15)
print(f"Output: '{text}'")

# Test 2: Top-k with temperature
print("\n" + "=" * 70)
print("TEST 2: Top-k Sampling (k=50, temp=0.8)")
print("-" * 70)
gen = TextGenerator(model, tokenizer, sampling_strategy=TopKSampling(k=50, temperature=0.8), use_cache=True)
text = gen.generate(prompt, max_new_tokens=15)
print(f"Output: '{text}'")

# Test 3: Nucleus sampling
print("\n" + "=" * 70)
print("TEST 3: Nucleus Sampling (p=0.9, temp=0.8)")
print("-" * 70)
gen = TextGenerator(model, tokenizer, sampling_strategy=NucleusSampling(p=0.9, temperature=0.8), use_cache=True)
text = gen.generate(prompt, max_new_tokens=15)
print(f"Output: '{text}'")

# Test 4: Higher temperature
print("\n" + "=" * 70)
print("TEST 4: Top-k with Higher Temperature (k=50, temp=1.2)")
print("-" * 70)
gen = TextGenerator(model, tokenizer, sampling_strategy=TopKSampling(k=50, temperature=1.2), use_cache=True)
text = gen.generate(prompt, max_new_tokens=15)
print(f"Output: '{text}'")

print("\n" + "=" * 70)
print("RECOMMENDATIONS")
print("=" * 70)
print("""
If you see repetitive patterns (adooadoo):
  1. Use Top-K or Nucleus sampling (not Greedy)
  2. Increase temperature (0.8 - 1.2)
  3. Use top_k=40-50 or top_p=0.9

Ternary models (BitNet) characteristics:
  - More sensitive to sampling parameters
  - May need slightly higher temperature than FP32 models
  - Greedy decoding can cause loops with sparse activations

Best practices:
  - Start with Nucleus (p=0.9, temp=0.8)
  - If too random, lower temp to 0.7
  - If repetitive, raise temp to 1.0-1.2
""")

print("\n" + "=" * 70)
print("All ways, always!")
print("=" * 70)
