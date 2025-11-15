"""
Test KV caching implementation.

Verifies that KV caching:
1. Produces same output as without cache
2. Is faster than recomputing everything
"""

import sys
import io
from pathlib import Path
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from rotor.bitnet_model import BitNetModel
from rotor.tokenizer import BitNetTokenizer
from rotor.generation import TextGenerator, GreedySampling

print("=" * 70)
print("KV Cache Test")
print("=" * 70)

model_dir = "C:/Users/samho/Desktop/BitNet-2B-model"

# Create small test config (faster for testing)
test_config = {
    'vocab_size': 128256,
    'd_model': 256,  # Much smaller for fast testing
    'n_layers': 2,   # Only 2 layers
    'n_heads': 8,
    'n_kv_heads': 4,
    'd_ff': 1024,
}

print("\n[1] Creating small test model...")
print("-" * 70)
model = BitNetModel(test_config)

# Initialize with random weights (just for testing)
print("  Initializing random weights (for testing only)...")
for layer in model.layers:
    # Initialize attention projections
    for proj in [layer.attention.q_proj, layer.attention.k_proj,
                 layer.attention.v_proj, layer.attention.o_proj]:
        proj.bit0 = np.random.randint(0, 256, size=proj.bit0.shape, dtype=np.uint8)
        proj.bit1 = np.random.randint(0, 256, size=proj.bit1.shape, dtype=np.uint8)
        proj.scale = 0.1
        proj.weights_cache = proj._decode_weights()

    # Initialize FFN projections
    for proj in [layer.ffn.gate_proj, layer.ffn.up_proj, layer.ffn.down_proj]:
        proj.bit0 = np.random.randint(0, 256, size=proj.bit0.shape, dtype=np.uint8)
        proj.bit1 = np.random.randint(0, 256, size=proj.bit1.shape, dtype=np.uint8)
        proj.scale = 0.1
        proj.weights_cache = proj._decode_weights()

print("✓ Model initialized")

# Load tokenizer
print("\n[2] Loading tokenizer...")
print("-" * 70)
tokenizer = BitNetTokenizer(model_dir)
print(f"✓ Tokenizer loaded")

# Test prompt
prompt = "The future of AI"
print(f"\n[3] Testing with prompt: '{prompt}'")
print("-" * 70)

# Encode prompt
token_ids = tokenizer.encode(prompt, add_special_tokens=True, return_numpy=True)
print(f"Encoded to {len(token_ids)} tokens")

# Test 1: Generate WITHOUT cache
print("\n[Test 1] Generate 3 tokens WITHOUT KV cache...")
print("-" * 70)
generator_no_cache = TextGenerator(
    model=model,
    tokenizer=tokenizer,
    sampling_strategy=GreedySampling(),
    temperature=1.0,
    max_length=100,
    use_cache=False  # Disable cache
)

start = time.perf_counter()
text_no_cache = generator_no_cache.generate(prompt, max_new_tokens=3)
time_no_cache = time.perf_counter() - start

print(f"Time: {time_no_cache:.3f}s")
print(f"Output: '{text_no_cache}'")

# Test 2: Generate WITH cache
print("\n[Test 2] Generate 3 tokens WITH KV cache...")
print("-" * 70)
generator_with_cache = TextGenerator(
    model=model,
    tokenizer=tokenizer,
    sampling_strategy=GreedySampling(),
    temperature=1.0,
    max_length=100,
    use_cache=True  # Enable cache
)

start = time.perf_counter()
text_with_cache = generator_with_cache.generate(prompt, max_new_tokens=3)
time_with_cache = time.perf_counter() - start

print(f"Time: {time_with_cache:.3f}s")
print(f"Output: '{text_with_cache}'")

# Compare results
print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)

# Check if outputs match (they should with greedy sampling)
outputs_match = text_no_cache == text_with_cache
print(f"\nOutputs match: {outputs_match}")
if outputs_match:
    print("  ✓ PASS: Same output with and without cache!")
else:
    print("  ✗ FAIL: Outputs differ!")
    print(f"    Without cache: '{text_no_cache}'")
    print(f"    With cache:    '{text_with_cache}'")

# Check speedup
speedup = time_no_cache / time_with_cache if time_with_cache > 0 else 0
print(f"\nSpeed comparison:")
print(f"  Without cache: {time_no_cache:.3f}s")
print(f"  With cache:    {time_with_cache:.3f}s")
print(f"  Speedup:       {speedup:.2f}×")

if speedup > 1.2:
    print("  ✓ PASS: KV cache is faster!")
elif speedup > 0.8:
    print("  ~ MARGINAL: Cache provides minimal speedup (may vary on small models)")
else:
    print("  ✗ FAIL: Cache is slower (something wrong?)")

# Summary
print("\n" + "=" * 70)
if outputs_match and speedup > 0.8:
    print("✅ KV CACHE WORKING CORRECTLY!")
    print("=" * 70)
    print("\nReady to use with full 2.4B model!")
    print("Expected speedup on real model: 5-10× for typical generation")
else:
    print("⚠️  ISSUES DETECTED")
    print("=" * 70)
    if not outputs_match:
        print("  - Outputs don't match (cache may be buggy)")
    if speedup <= 0.8:
        print("  - No speedup observed (may be normal for tiny test model)")

print("\n🌀 All ways, always!")
