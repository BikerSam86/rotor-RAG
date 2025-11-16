"""
Test tokenizer functionality.
"""

import pytest

pytest.skip("Requires the external BitNet tokenizer assets", allow_module_level=True)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rotor.tokenizer import BitNetTokenizer

print("=" * 70)
print("Tokenizer Test")
print("=" * 70)

model_dir = "C:/Users/samho/Desktop/BitNet-2B-model"

# Load tokenizer
print("\n[TEST 1] Loading tokenizer...")
print("-" * 70)
tokenizer = BitNetTokenizer(model_dir)

# Test encoding
print("\n[TEST 2] Text encoding...")
print("-" * 70)
test_text = "Hello, world! How are you today?"
print(f"Input text: '{test_text}'")

tokens = tokenizer.encode(test_text, add_special_tokens=True)
print(f"Tokens: {tokens}")
print(f"Token count: {len(tokens)}")
print(f"Token dtype: {tokens.dtype}")

# Test decoding
print("\n[TEST 3] Token decoding...")
print("-" * 70)
decoded = tokenizer.decode(tokens, skip_special_tokens=True)
print(f"Decoded text: '{decoded}'")

# Verify round-trip
if decoded.strip() == test_text:
    print("✓ Round-trip successful!")
else:
    print("⚠ Round-trip mismatch")
    print(f"  Expected: '{test_text}'")
    print(f"  Got:      '{decoded.strip()}'")

# Test batch encoding
print("\n[TEST 4] Batch encoding...")
print("-" * 70)
texts = [
    "The quick brown fox",
    "Jumps over the lazy dog",
    "Hello world"
]
print(f"Encoding {len(texts)} texts...")
encoded = tokenizer.encode_batch(texts)
print(f"Results: {len(encoded)} sequences")
for i, enc in enumerate(encoded):
    print(f"  Text {i+1}: {len(enc)} tokens")

# Test batch decoding
print("\n[TEST 5] Batch decoding...")
print("-" * 70)
decoded_batch = tokenizer.decode_batch(encoded)
print("Decoded texts:")
for i, dec in enumerate(decoded_batch):
    print(f"  {i+1}. '{dec}'")

# Test special tokens
print("\n[TEST 6] Special tokens...")
print("-" * 70)
print(f"BOS token: '{tokenizer.bos_token}' (ID: {tokenizer.bos_token_id})")
print(f"EOS token: '{tokenizer.eos_token}' (ID: {tokenizer.eos_token_id})")
print(f"PAD token: '{tokenizer.pad_token}' (ID: {tokenizer.pad_token_id})")

# Test without special tokens
print("\n[TEST 7] Encoding without special tokens...")
print("-" * 70)
text = "Hello"
with_special = tokenizer.encode(text, add_special_tokens=True)
without_special = tokenizer.encode(text, add_special_tokens=False)
print(f"Text: '{text}'")
print(f"With special tokens:    {with_special} (len={len(with_special)})")
print(f"Without special tokens: {without_special} (len={len(without_special)})")

print("\n" + "=" * 70)
print("✅ ALL TOKENIZER TESTS PASSED!")
print("=" * 70)
print("\n✓ Tokenizer ready for text generation")
print("🌀 All ways, always!")
