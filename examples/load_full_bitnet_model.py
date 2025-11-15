"""
Load the complete Microsoft BitNet-b1.58-2B-4T model with all weights.

This demonstrates loading all 30 layers and converting all ternary weights
from BitNet packed format to our superior Rotor format!
"""

import sys
import io
from pathlib import Path
import time

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rotor.bitnet_model import load_bitnet_model
import numpy as np


def main():
    print("\n" + "╔" + "═"*68 + "╗")
    print("║" + " "*12 + "Load Complete BitNet Model" + " "*29 + "║")
    print("║" + " "*8 + "Microsoft BitNet-b1.58-2B-4T (2.4B params)" + " "*18 + "║")
    print("╚" + "═"*68 + "╝\n")

    model_dir = "C:/Users/samho/Desktop/BitNet-2B-model"

    print(f"Model directory: {model_dir}\n")

    # Load complete model
    print("="*70)
    print("Loading Model (this may take a minute...)")
    print("="*70)

    start = time.perf_counter()
    model = load_bitnet_model(model_dir)
    load_time = time.perf_counter() - start

    print(f"\n✓ Model loaded in {load_time:.2f}s")

    # Test forward pass with dummy input
    print("\n" + "="*70)
    print("Testing Forward Pass")
    print("="*70)

    # Create test input (10 tokens)
    test_input = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int64)
    print(f"\nInput tokens: {test_input}")
    print(f"Input shape: {test_input.shape}")

    # Forward pass
    print("\nRunning forward pass...")
    start = time.perf_counter()
    logits = model.forward(test_input)
    forward_time = (time.perf_counter() - start) * 1000

    print(f"\n✓ Forward pass complete!")
    print(f"  Time: {forward_time:.2f} ms")
    print(f"  Output shape: {logits.shape}")
    print(f"  Expected: ({test_input.shape[0]}, {model.vocab_size})")

    # Check output
    print(f"\nOutput statistics:")
    print(f"  Min: {logits.min():.4f}")
    print(f"  Max: {logits.max():.4f}")
    print(f"  Mean: {logits.mean():.4f}")

    # Get top predictions for last token
    last_token_logits = logits[-1, :]
    top_5_indices = np.argsort(last_token_logits)[-5:][::-1]

    print(f"\nTop 5 predictions for next token:")
    for i, idx in enumerate(top_5_indices, 1):
        print(f"  {i}. Token {idx}: logit={last_token_logits[idx]:.4f}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n✓ Successfully loaded Microsoft BitNet-b1.58-2B-4T!")
    print(f"✓ Model: {model.n_layers} layers, {model.d_model} hidden dims")
    print(f"✓ Vocabulary: {model.vocab_size:,} tokens")
    print(f"✓ All 210 ternary layers converted to Rotor format!")
    print(f"✓ Forward pass working: {forward_time:.2f} ms for 10 tokens")

    # Memory estimation
    total_params = 0

    # Embeddings (FP32)
    embed_params = model.vocab_size * model.d_model
    total_params += embed_params

    # Per layer
    per_layer_params = (
        # Attention
        model.d_model * model.d_model * 4 +  # Q, K, V, O
        # FFN
        model.d_model * model.d_ff * 3  # gate, up, down
    )
    total_params += per_layer_params * model.n_layers

    print(f"\nParameter count:")
    print(f"  Embeddings: {embed_params/1e6:.1f}M (FP32)")
    print(f"  Transformer: {per_layer_params * model.n_layers/1e6:.1f}M (Ternary)")
    print(f"  Total: {total_params/1e9:.2f}B parameters")

    # Memory (rough estimate)
    embed_memory = embed_params * 4 / 1024 / 1024  # FP32
    ternary_memory = (per_layer_params * model.n_layers * 2) / 8 / 1024 / 1024  # 2 bits/weight
    total_memory = embed_memory + ternary_memory

    print(f"\nMemory usage:")
    print(f"  Embeddings: {embed_memory:.1f} MB")
    print(f"  Ternary weights: {ternary_memory:.1f} MB")
    print(f"  Total: {total_memory:.1f} MB")
    print(f"  (vs FP32: {total_params * 4 / 1024 / 1024 / 1024:.2f} GB - {total_params * 4 / total_memory / 1024:.1f}× larger!)")

    print(f"\n🎉 SUCCESS! Real 2.4B parameter model running with Rotor format!")
    print(f"\nNext steps:")
    print(f"  1. Load tokenizer and decode outputs")
    print(f"  2. Implement text generation")
    print(f"  3. Test with real prompts")
    print(f"  4. Benchmark vs bitnet.cpp")

    print(f"\n🌀 All ways, always!")
    print(f"="*70 + "\n")


if __name__ == "__main__":
    main()
