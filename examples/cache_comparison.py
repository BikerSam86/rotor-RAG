"""
Direct A/B comparison: WITH cache vs WITHOUT cache.
"""

import sys
import io
from pathlib import Path
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rotor.bitnet_model import load_bitnet_model
from rotor.tokenizer import BitNetTokenizer
from rotor.generation import GreedySampling
import numpy as np

print("=" * 70)
print("KV CACHE A/B COMPARISON")
print("=" * 70)

model_dir = "C:/Users/samho/Desktop/BitNet-2B-model"

print("\n[1] Loading model...")
model = load_bitnet_model(model_dir)

print("\n[2] Loading tokenizer...")
tokenizer = BitNetTokenizer(model_dir)

prompt = "The future of AI"
token_ids = tokenizer.encode(prompt, add_special_tokens=True, return_numpy=True)
print(f"\nPrompt: '{prompt}' ({len(token_ids)} tokens)")

sampling = GreedySampling()

# Test A: WITHOUT cache (generate token 2)
print("\n" + "=" * 70)
print("TEST A: Token 2 WITHOUT cache (recompute 6 tokens)")
print("=" * 70)

# Simulate: we already generated token 1
token_ids_extended = np.array(list(token_ids) + [78212], dtype=np.int64)  # 6 tokens total
print(f"Processing {len(token_ids_extended)} tokens (full sequence)...")

start = time.perf_counter()
input_ids = token_ids_extended.reshape(1, -1)
logits, _ = model.forward(input_ids, use_cache=False)
next_token_logits = logits[0, -1, :]
next_token = sampling.sample(next_token_logits)
time_without_cache = time.perf_counter() - start

print(f"Time WITHOUT cache: {time_without_cache:.1f}s")
print(f"Generated token: {next_token}")

# Test B: WITH cache (generate token 2)
print("\n" + "=" * 70)
print("TEST B: Token 2 WITH cache (process 1 new token)")
print("=" * 70)

# Step 1: Build cache with prompt
print("Step 1: Processing prompt to build cache...")
start_build = time.perf_counter()
input_ids = token_ids.reshape(1, -1)
logits, kv_cache = model.forward(input_ids, use_cache=True)
next_token_logits = logits[0, -1, :]
token_1 = sampling.sample(next_token_logits)
time_build = time.perf_counter() - start_build
print(f"  Cache built in {time_build:.1f}s, generated token: {token_1}")

# Step 2: Use cache for token 2
print("Step 2: Using cache to generate token 2...")
start_cached = time.perf_counter()
new_token = np.array([token_1], dtype=np.int64)
input_ids = new_token.reshape(1, -1)
logits, kv_cache = model.forward(input_ids, past_kv_cache=kv_cache, use_cache=True)
next_token_logits = logits[0, -1, :]
token_2 = sampling.sample(next_token_logits)
time_with_cache = time.perf_counter() - start_cached

print(f"Time WITH cache: {time_with_cache:.1f}s")
print(f"Generated token: {token_2}")

# Comparison
print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)
print(f"\nGenerating Token 2:")
print(f"  WITHOUT cache (6 tokens): {time_without_cache:.1f}s")
print(f"  WITH cache    (1 token):  {time_with_cache:.1f}s")

if time_without_cache > time_with_cache:
    speedup = time_without_cache / time_with_cache
    print(f"\n  ✅ Speedup: {speedup:.2f}×")
    print(f"  ✅ KV cache is {speedup:.1f}× FASTER!")
else:
    slowdown = time_with_cache / time_without_cache
    print(f"\n  ⚠️  Cache is {slowdown:.2f}× slower (unexpected!)")

print("\n" + "=" * 70)
print("ANALYSIS")
print("=" * 70)
print("Expected speedup: ~1.5-2× for short sequences")
print("(Cache saves recomputing Q,K,V for past tokens)")
print("\n🌀 All ways, always!")
