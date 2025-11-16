"""
Quick comparison: Greedy vs Microsoft's Nucleus Sampling
Shows the difference in output quality.
"""

import sys
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rotor.bitnet_model import load_bitnet_model
from rotor.generation import TextGenerator, GreedySampling, NucleusSampling
from rotor.tokenizer import BitNetTokenizer
import time

print("=" * 70)
print("SAMPLING STRATEGY COMPARISON")
print("=" * 70)

MODEL_PATH = r"C:\Users\samho\Desktop\BitNet-2B-model"

print("\nLoading model (CPU + KV cache)...")
model = load_bitnet_model(MODEL_PATH, use_gpu=False)
tokenizer = BitNetTokenizer(MODEL_PATH)

prompt = "The future of AI is"
num_tokens = 5

print(f"\nPrompt: '{prompt}'")
print(f"Generating {num_tokens} tokens with each strategy...\n")

# Test 1: Greedy (what we had before)
print("=" * 70)
print("[1] GREEDY SAMPLING (old default)")
print("=" * 70)
print("Always picks highest probability token")
print("Problem: Gets stuck in loops\n")

generator_greedy = TextGenerator(
    model=model,
    tokenizer=tokenizer,
    sampling_strategy=GreedySampling(),
    use_cache=True
)

start = time.perf_counter()
result_greedy = generator_greedy.generate(prompt, max_new_tokens=num_tokens)
time_greedy = time.perf_counter() - start

print(f"Output: '{result_greedy}'")
print(f"Time: {time_greedy:.1f}s\n")

# Test 2: Nucleus (Microsoft's setting)
print("=" * 70)
print("[2] NUCLEUS SAMPLING (Microsoft's setting)")
print("=" * 70)
print("temperature=0.6, top_p=0.9")
print("Samples from top 90% probability mass\n")

generator_nucleus = TextGenerator(
    model=model,
    tokenizer=tokenizer,
    sampling_strategy=NucleusSampling(p=0.9),
    temperature=0.6,
    use_cache=True
)

start = time.perf_counter()
result_nucleus = generator_nucleus.generate(
    prompt,
    max_new_tokens=num_tokens,
    temperature=0.6
)
time_nucleus = time.perf_counter() - start

print(f"Output: '{result_nucleus}'")
print(f"Time: {time_nucleus:.1f}s\n")

# Summary
print("=" * 70)
print("VERDICT")
print("=" * 70)
print(f"\nGreedy:  '{result_greedy}'")
print(f"Nucleus: '{result_nucleus}'")
print("\nNotice the difference?")
print("- Greedy tends to repeat")
print("- Nucleus is more diverse and natural")
print("\nMicrosoft's settings (nucleus + temp 0.6) are the way to go!")
print("\n" + "=" * 70)
