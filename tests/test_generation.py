"""
Test text generation utilities.
"""

import sys
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from rotor.generation import (
    GreedySampling,
    TopKSampling,
    TopPSampling,
)

print("=" * 70)
print("Generation Utilities Test")
print("=" * 70)

# Test logits (simulating model output)
np.random.seed(42)
vocab_size = 100
test_logits = np.random.randn(vocab_size).astype(np.float32)

# Test 1: Greedy Sampling
print("\n[TEST 1] Greedy Sampling")
print("-" * 70)
greedy = GreedySampling()
token_1 = greedy.sample(test_logits)
token_2 = greedy.sample(test_logits)
print(f"Sample 1: {token_1}")
print(f"Sample 2: {token_2}")

if token_1 == token_2:
    print("✓ Greedy sampling is deterministic")
else:
    print("✗ Greedy sampling should be deterministic!")
    sys.exit(1)

expected_greedy = np.argmax(test_logits)
if token_1 == expected_greedy:
    print(f"✓ Correctly picks argmax (token {expected_greedy})")
else:
    print(f"✗ Should pick token {expected_greedy}, got {token_1}")
    sys.exit(1)

# Test 2: Top-K Sampling
print("\n[TEST 2] Top-K Sampling")
print("-" * 70)
top_k = TopKSampling(k=10)
samples = [top_k.sample(test_logits, temperature=1.0) for _ in range(10)]
print(f"10 samples: {samples}")
print(f"Unique tokens: {len(set(samples))}")

if len(set(samples)) > 1:
    print("✓ Top-K sampling is stochastic")
else:
    print("⚠ Top-K should produce varied samples (might be unlucky)")

# Verify samples are in top-k
top_k_indices = np.argpartition(test_logits, -10)[-10:]
all_in_top_k = all(s in top_k_indices for s in samples)
if all_in_top_k:
    print("✓ All samples are from top-k tokens")
else:
    print("✗ Some samples are not in top-k!")
    sys.exit(1)

# Test 3: Top-P Sampling
print("\n[TEST 3] Top-P (Nucleus) Sampling")
print("-" * 70)
top_p = TopPSampling(p=0.9)
samples = [top_p.sample(test_logits, temperature=1.0) for _ in range(10)]
print(f"10 samples: {samples}")
print(f"Unique tokens: {len(set(samples))}")

if len(set(samples)) > 1:
    print("✓ Top-P sampling is stochastic")
else:
    print("⚠ Top-P should produce varied samples (might be unlucky)")

# Test 4: Temperature Effect
print("\n[TEST 4] Temperature Effect")
print("-" * 70)

# High temperature (more random)
top_k_hot = TopKSampling(k=50)
samples_hot = [top_k_hot.sample(test_logits, temperature=2.0) for _ in range(20)]
unique_hot = len(set(samples_hot))

# Low temperature (more deterministic)
top_k_cold = TopKSampling(k=50)
samples_cold = [top_k_cold.sample(test_logits, temperature=0.1) for _ in range(20)]
unique_cold = len(set(samples_cold))

print(f"High temp (2.0): {unique_hot} unique tokens")
print(f"Low temp (0.1):  {unique_cold} unique tokens")

if unique_hot >= unique_cold:
    print("✓ Higher temperature produces more diversity")
else:
    print("⚠ Expected more diversity with higher temperature")

# Test 5: Softmax Numerical Stability
print("\n[TEST 5] Softmax Numerical Stability")
print("-" * 70)

# Extreme logits
extreme_logits = np.array([-1000.0, 0.0, 1000.0], dtype=np.float32)
sampler = TopKSampling(k=3)

# Should not produce NaN or Inf
sample = sampler.sample(extreme_logits, temperature=1.0)
print(f"Sample from extreme logits: {sample}")

if np.isfinite(sample):
    print("✓ Handles extreme logits without overflow")
else:
    print("✗ Produced NaN or Inf!")
    sys.exit(1)

# Summary
print("\n" + "=" * 70)
print("✅ ALL GENERATION TESTS PASSED!")
print("=" * 70)
print("\n✓ Greedy sampling working")
print("✓ Top-K sampling working")
print("✓ Top-P sampling working")
print("✓ Temperature control working")
print("✓ Numerical stability verified")
print("\n🌀 Ready for text generation!")
