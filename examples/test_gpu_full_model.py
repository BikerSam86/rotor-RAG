"""
Test full BitNet model with GPU + KV cache acceleration.
"""

import sys
import io
from pathlib import Path
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rotor.bitnet_model import load_bitnet_model
from rotor.tokenizer import BitNetTokenizer
from rotor.generation import TextGenerator, GreedySampling

print("=" * 70)
print("GPU-ACCELERATED BitNet WITH KV CACHE")
print("=" * 70)

model_dir = "C:/Users/samho/Desktop/BitNet-2B-model"

print("\n[1] Loading GPU-accelerated model...")
start = time.perf_counter()
model = load_bitnet_model(model_dir, use_gpu=True)
load_time = time.perf_counter() - start
print(f"Model loaded in {load_time:.1f}s")

print("\n[2] Loading tokenizer...")
tokenizer = BitNetTokenizer(model_dir)

print("\n[3] Creating generator with KV cache...")
generator = TextGenerator(
    model=model,
    tokenizer=tokenizer,
    sampling_strategy=GreedySampling(),
    use_cache=True  # Enable KV cache
)

prompt = "The future of AI"
print(f"\n[4] Generating 3 tokens for: '{prompt}'")
print("=" * 70)

start = time.perf_counter()
text = generator.generate(prompt, max_new_tokens=3)
total_time = time.perf_counter() - start

print(f"\nGenerated: '{text}'")
print(f"Total time: {total_time:.1f}s")
print(f"Average per token: {total_time/3:.1f}s")

print("\n" + "=" * 70)
print("PERFORMANCE COMPARISON")
print("=" * 70)
print("\nYoga Book CPU-only (baseline):")
print("  - Per token: ~105s")
print("  - 3 tokens: ~315s")

print("\nYoga Book GPU + KV cache (expected):")
print("  - Combined speedup: ~4-5x")
print("  - Per token: ~20-30s")
print("  - 3 tokens: ~60-90s")

if total_time < 200:
    print(f"\n[SUCCESS] GPU acceleration working!")
    print(f"Actual speedup: ~{315/total_time:.1f}x faster!")
else:
    print(f"\n[INFO] Times: {total_time:.1f}s (still testing...)")

print("\n" + "=" * 70)
print("All ways, always!")
print("=" * 70)
