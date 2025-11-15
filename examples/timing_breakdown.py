"""
Detailed timing breakdown to see cache performance per token.
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
print("PER-TOKEN TIMING BREAKDOWN")
print("=" * 70)

model_dir = "C:/Users/samho/Desktop/BitNet-2B-model"

print("\n[1] Loading model...")
start = time.perf_counter()
model = load_bitnet_model(model_dir)
load_time = time.perf_counter() - start
print(f"Model load time: {load_time:.1f}s")

print("\n[2] Loading tokenizer...")
tokenizer = BitNetTokenizer(model_dir)
print(f"✓ Tokenizer loaded")

prompt = "The future of AI"
print(f"\n[3] Testing with prompt: '{prompt}'")
print("-" * 70)

# Encode prompt
token_ids = tokenizer.encode(prompt, add_special_tokens=True, return_numpy=True)
token_ids_list = token_ids.tolist()
print(f"Prompt tokens: {len(token_ids)} tokens")

# Manual generation loop with detailed timing
print("\n" + "=" * 70)
print("MANUAL GENERATION WITH TIMING")
print("=" * 70)

sampling = GreedySampling()
kv_cache = None
generated_tokens = []

# Token 0: Process prompt
print(f"\n[Token 0] Processing prompt ({len(token_ids)} tokens)...")
start = time.perf_counter()
input_ids = token_ids.reshape(1, -1)
logits, kv_cache = model.forward(input_ids, use_cache=True)
next_token_logits = logits[0, -1, :]
next_token = sampling.sample(next_token_logits)
elapsed = time.perf_counter() - start
print(f"  Time: {elapsed:.1f}s")
print(f"  Generated token: {next_token}")
print(f"  Cache created: {len(kv_cache)} layers")
token_ids_list.append(next_token)
generated_tokens.append(next_token)

# Token 1: Use cache (process 1 token)
print(f"\n[Token 1] Using cache (process 1 new token)...")
start = time.perf_counter()
new_token = np.array([token_ids_list[-1]], dtype=np.int64)
input_ids = new_token.reshape(1, -1)
logits, kv_cache = model.forward(input_ids, past_kv_cache=kv_cache, use_cache=True)
next_token_logits = logits[0, -1, :]
next_token = sampling.sample(next_token_logits)
elapsed = time.perf_counter() - start
print(f"  Time: {elapsed:.1f}s")
print(f"  Generated token: {next_token}")
print(f"  Cache updated: {len(kv_cache)} layers")
token_ids_list.append(next_token)
generated_tokens.append(next_token)

# Token 2: Use cache (process 1 token)
print(f"\n[Token 2] Using cache (process 1 new token)...")
start = time.perf_counter()
new_token = np.array([token_ids_list[-1]], dtype=np.int64)
input_ids = new_token.reshape(1, -1)
logits, kv_cache = model.forward(input_ids, past_kv_cache=kv_cache, use_cache=True)
next_token_logits = logits[0, -1, :]
next_token = sampling.sample(next_token_logits)
elapsed = time.perf_counter() - start
print(f"  Time: {elapsed:.1f}s")
print(f"  Generated token: {next_token}")
token_ids_list.append(next_token)
generated_tokens.append(next_token)

# Decode result
print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)
final_ids = np.array(token_ids_list, dtype=np.int64)
final_text = tokenizer.decode(final_ids, skip_special_tokens=True)
print(f"Generated: '{final_text}'")

print("\n" + "=" * 70)
print("EXPECTED PERFORMANCE")
print("=" * 70)
print("Token 0 (prompt): ~250-300s (process full prompt)")
print("Token 1 (cached): ~50-100s (still slow on Core-M)")
print("Token 2 (cached): ~50-100s")
print("\nNote: Cache IS working (only 1 token processed), but Core-M is")
print("slow at matrix operations even with cache enabled.")
print("\n🌀 All ways, always!")
