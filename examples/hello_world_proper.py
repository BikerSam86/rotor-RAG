"""
Hello World with Microsoft's Proper Settings!
GPU (OpenCL) + KV Cache + Nucleus Sampling

Let's see if this potato can speak coherently!
"""

import sys
import io
from pathlib import Path
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rotor.bitnet_model import load_bitnet_model
from rotor.generation import TextGenerator, NucleusSampling
from rotor.tokenizer import BitNetTokenizer

print("=" * 70)
print("HELLO WORLD - Microsoft Settings + GPU")
print("=" * 70)
print("\nConfiguration:")
print("  - GPU: OpenCL (Intel HD 615)")
print("  - KV Cache: Enabled (2.7x speedup)")
print("  - Sampling: Nucleus (top_p=0.9)")
print("  - Temperature: 0.6 (Microsoft's setting)")
print("=" * 70)

MODEL_PATH = r"C:\Users\samho\Desktop\BitNet-2B-model"

print("\n[1/3] Loading model with GPU...")
start_load = time.perf_counter()
model = load_bitnet_model(MODEL_PATH, use_gpu=True)
load_time = time.perf_counter() - start_load
print(f"      Model loaded in {load_time:.1f}s")

print("\n[2/3] Loading tokenizer...")
tokenizer = BitNetTokenizer(MODEL_PATH)

print("\n[3/3] Creating generator with Microsoft's settings...")
generator = TextGenerator(
    model=model,
    tokenizer=tokenizer,
    sampling_strategy=NucleusSampling(p=0.9),
    temperature=0.6,
    use_cache=True
)

print("\n" + "=" * 70)
print("GENERATION")
print("=" * 70)

prompts = [
    "Hello, world! My name is",
    "The meaning of life is",
]

for prompt in prompts:
    print(f"\nPrompt: '{prompt}'")
    print("-" * 70)

    start = time.perf_counter()

    try:
        result = generator.generate(
            prompt,
            max_new_tokens=8,
            temperature=0.6
        )

        elapsed = time.perf_counter() - start

        print(f"Output: '{result}'")
        print(f"Time: {elapsed:.1f}s ({elapsed/8:.1f}s per token)")

    except Exception as e:
        print(f"Error: {e}")

print("\n" + "=" * 70)
print("POTATO STATUS")
print("=" * 70)
print("\nIf you see actual words (not 'adooadoo'), we win!")
print("This is a 2016 fanless tablet running a 2.4B LLM locally.")
print("No cloud. No subscriptions. No tethers.")
print("\nAll ways, always! 🌀")
print("=" * 70)
