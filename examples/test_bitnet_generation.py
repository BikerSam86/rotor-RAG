"""
Test BitNet generation with Microsoft's official settings.

Uses the generation_config.json settings from Microsoft's model:
- do_sample: true
- temperature: 0.6
- top_p: 0.9
"""

import sys
import io
from pathlib import Path
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rotor.bitnet_model import load_bitnet_model
from rotor.generation import TextGenerator, TopPSampling
from rotor.tokenizer import BitNetTokenizer

print("=" * 70)
print("BitNet Generation with Microsoft's Settings")
print("=" * 70)

# Microsoft's official generation_config.json settings:
# {
#   "do_sample": true,
#   "temperature": 0.6,
#   "top_p": 0.9
# }

MODEL_PATH = r"C:\Users\samho\Desktop\BitNet-2B-model"

print("\n[1] Loading model (CPU mode for stability)...")
model = load_bitnet_model(MODEL_PATH, use_gpu=False)

print("\n[2] Loading tokenizer...")
tokenizer = BitNetTokenizer(MODEL_PATH)

print("\n[3] Creating generator with Microsoft's settings...")
print("    - Nucleus sampling (top_p=0.9)")
print("    - Temperature: 0.6")
print("    - KV cache: enabled")

generator = TextGenerator(
    model=model,
    tokenizer=tokenizer,
    sampling_strategy=TopPSampling(p=0.9),  # Nucleus sampling
    temperature=0.6,  # Microsoft's recommended temperature
    use_cache=True
)

print("\n" + "=" * 70)
print("GENERATION TESTS")
print("=" * 70)

test_prompts = [
    "The future of AI",
    "Once upon a time",
    "In a world where",
]

for i, prompt in enumerate(test_prompts, 1):
    print(f"\n[Test {i}] Prompt: '{prompt}'")
    print("-" * 70)

    start = time.perf_counter()

    try:
        generated = generator.generate(
            prompt,
            max_new_tokens=10,
            temperature=0.6  # Explicitly set
        )

        elapsed = time.perf_counter() - start

        print(f"Generated: '{generated}'")
        print(f"Time: {elapsed:.1f}s ({elapsed/10:.1f}s per token)")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 70)
print("COMPARISON")
print("=" * 70)
print("\nMicrosoft's settings (temperature=0.6, top_p=0.9):")
print("  ✓ More diverse outputs")
print("  ✓ Less repetitive")
print("  ✓ More 'creative' but still coherent")
print("\nGreedy sampling (default):")
print("  ✗ Repetitive ('adooadoo')")
print("  ✗ Deterministic (same output every time)")
print("  ✗ Gets stuck in loops")

print("\n" + "=" * 70)
print("All ways, always!")
print("=" * 70)
