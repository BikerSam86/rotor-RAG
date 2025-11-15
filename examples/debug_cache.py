"""
Quick debug test to see if KV cache is being used.
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
print("DEBUG: KV Cache Investigation")
print("=" * 70)

model_dir = "C:/Users/samho/Desktop/BitNet-2B-model"

print("\n[1] Loading model...")
print("-" * 70)
model = load_bitnet_model(model_dir)

print("\n[2] Loading tokenizer...")
print("-" * 70)
tokenizer = BitNetTokenizer(model_dir)
print(f"✓ Tokenizer loaded")

prompt = "The future of AI"
print(f"\n[3] Testing with prompt: '{prompt}'")
print("-" * 70)

# Test WITH cache (use_cache=True)
print("\n[Test] Generate 2 tokens WITH KV cache...")
print("=" * 70)
generator = TextGenerator(
    model=model,
    tokenizer=tokenizer,
    sampling_strategy=GreedySampling(),
    temperature=1.0,
    max_length=100,
    use_cache=True  # Enable cache
)

print(f"\nGenerator config: use_cache={generator.use_cache}")
print("Starting generation...")

start = time.perf_counter()
text = generator.generate(prompt, max_new_tokens=2)
elapsed = time.perf_counter() - start

print(f"\nTotal time: {elapsed:.1f}s")
print(f"Output: '{text}'")

print("\n" + "=" * 70)
print("DEBUG TEST COMPLETE")
print("=" * 70)
print("\nIf you see DEBUG messages above showing:")
print("  - First step: processing N tokens")
print("  - Using cache: processing 1 new token")
print("Then cache is working!")
print("\n🌀 All ways, always!")
