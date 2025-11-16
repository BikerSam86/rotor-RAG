"""
Test with PROPER chat format for Microsoft's BitNet model.
This model is a CHAT model, not a completion model!
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
print("PROPER CHAT FORMAT TEST")
print("=" * 70)
print("\nThis is a CHAT model - it needs proper formatting!")
print("Format: 'User: <message><|eot_id|>Assistant: '")
print("=" * 70)

MODEL_PATH = r"C:\Users\samho\Desktop\BitNet-2B-model"

print("\nLoading model (GPU + KV cache)...")
model = load_bitnet_model(MODEL_PATH, use_gpu=True)
tokenizer = BitNetTokenizer(MODEL_PATH)

print("\nCreating generator with Microsoft's settings...")
generator = TextGenerator(
    model=model,
    tokenizer=tokenizer,
    sampling_strategy=NucleusSampling(p=0.9),
    temperature=0.6,
    use_cache=True
)

print("\n" + "=" * 70)
print("GENERATION WITH PROPER CHAT FORMAT")
print("=" * 70)

# Proper chat format!
test_cases = [
    {
        "user": "Hello! How are you?",
        "formatted": "User: Hello! How are you?<|eot_id|>Assistant: "
    },
    {
        "user": "What is 2+2?",
        "formatted": "User: What is 2+2?<|eot_id|>Assistant: "
    },
]

for i, test in enumerate(test_cases, 1):
    print(f"\n[Test {i}]")
    print(f"User: {test['user']}")
    print(f"Formatted prompt: '{test['formatted']}'")
    print("-" * 70)

    start = time.perf_counter()

    try:
        result = generator.generate(
            test['formatted'],
            max_new_tokens=12,
            temperature=0.6,
            stop_on_eos=True  # Stop at <|eot_id|>
        )

        elapsed = time.perf_counter() - start

        # Extract just the assistant's response
        assistant_response = result.replace(test['formatted'], '')

        print(f"Assistant: {assistant_response}")
        print(f"Time: {elapsed:.1f}s")

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 70)
print("DIAGNOSIS")
print("=" * 70)
print("\nIf we STILL get 'adooadoo', then:")
print("  1. Model weights might be corrupted")
print("  2. Weight conversion (BitNet->Rotor) has bugs")
print("  3. Forward pass has numerical issues")
print("\nBut if we get REAL responses, then:")
print("  ✅ We just needed proper chat formatting!")
print("\n" + "=" * 70)
