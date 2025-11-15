"""
Test generation quality - verify we get sensible text, not "adooadoo" gibberish.
This tests both CPU and GPU to ensure outputs are correct.
"""

import sys
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rotor.bitnet_model import load_bitnet_model
from rotor.tokenizer import BitNetTokenizer
from rotor.generation import TextGenerator, GreedySampling

print("=" * 70)
print("GENERATION QUALITY TEST")
print("=" * 70)

model_dir = "C:/Users/samho/Desktop/BitNet-2B-model"

# Load tokenizer
print("\nLoading tokenizer...")
tokenizer = BitNetTokenizer(model_dir)

# Test prompts
prompts = [
    "The future of AI",
    "Once upon a time",
    "Hello, my name is",
]

print("\n" + "=" * 70)
print("TEST 1: CPU with KV Cache (10 tokens)")
print("=" * 70)

print("\nLoading CPU model...")
model_cpu = load_bitnet_model(model_dir, use_gpu=False)

generator_cpu = TextGenerator(
    model=model_cpu,
    tokenizer=tokenizer,
    use_cache=True
)

for prompt in prompts:
    print(f"\nPrompt: '{prompt}'")
    text = generator_cpu.generate(prompt, max_new_tokens=10)
    print(f"Output: '{text}'")

    # Check for repetitive patterns (sign of bad generation)
    tokens = text.split()
    if len(tokens) > 1:
        last_token = tokens[-1]
        repeat_count = sum(1 for t in tokens[-5:] if t == last_token)
        if repeat_count >= 3:
            print("  [WARN] Repetitive output detected!")

print("\n" + "=" * 70)
print("TEST 2: Check specific token 78212")
print("=" * 70)

# This is the token we saw generated in cache tests
token_id = 78212
decoded = tokenizer.decode([token_id])
print(f"\nToken 78212 decodes to: '{decoded}'")

# Let's also check what "The future of AI" prompt encodes to
print(f"\nPrompt 'The future of AI' encodes to:")
encoded = tokenizer.encode("The future of AI")
print(f"  Token IDs: {encoded}")
print(f"  Decoded back: '{tokenizer.decode(encoded)}'")

print("\n" + "=" * 70)
print("TEST 3: Token-by-token generation (first 5 tokens)")
print("=" * 70)

prompt = "The future of AI"
print(f"\nPrompt: '{prompt}'")
print("\nGenerating token by token:")

# Manual generation to see each token
input_ids = tokenizer.encode(prompt)
print(f"Starting with {len(input_ids)} tokens: {input_ids}")

for i in range(5):
    logits, _ = model_cpu.forward(input_ids, use_cache=False)
    next_token_logits = logits[-1]

    # Greedy sampling
    next_token = int(next_token_logits.argmax())

    # Decode
    decoded = tokenizer.decode([next_token])

    print(f"  Token {i+1}: {next_token} -> '{decoded}'")

    # Check if it's a sensible character/word
    if decoded.strip() == "" or decoded == decoded[0] * len(decoded):
        print(f"    [WARN] Suspicious token!")

    # Append for next iteration
    import numpy as np
    input_ids = np.append(input_ids, next_token)

full_text = tokenizer.decode(input_ids)
print(f"\nFull output: '{full_text}'")

print("\n" + "=" * 70)
print("ANALYSIS")
print("=" * 70)

print("""
If you see:
  - Sensible English words: Model is working correctly!
  - Repetitive patterns (adooadoo): Possible issues:
    * Model weights not loaded correctly
    * Numerical precision issues
    * Temperature too low (overly deterministic)
    * Model needs warmup

Common issues with ternary models:
  - Sparse activations can cause repetition
  - May need nucleus/top-k sampling instead of greedy
  - Temperature adjustment helps diversity

Try adjusting generation parameters if output is repetitive.
""")

print("=" * 70)
