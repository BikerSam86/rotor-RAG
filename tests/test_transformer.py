"""Test transformer components."""

import sys
import io
from pathlib import Path

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from rotor.transformer import (
    RMSNorm,
    TernaryLinear,
    MultiHeadAttention,
    GatedFFN,
    TransformerBlock
)


def test_rmsnorm():
    """Test RMSNorm."""
    print("\nTesting RMSNorm...")
    norm = RMSNorm(512)
    x = np.random.randn(10, 512).astype(np.float32)
    y = norm.forward(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
    print(f"  ✓ RMSNorm working!")


def test_ternary_linear():
    """Test TernaryLinear."""
    print("\nTesting TernaryLinear...")
    linear = TernaryLinear(512, 256)

    # Test forward pass (weights are zero-initialized)
    x = np.random.randn(10, 512).astype(np.float32)
    y = linear.forward(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
    print(f"  ✓ TernaryLinear working!")


def test_attention():
    """Test MultiHeadAttention."""
    print("\nTesting MultiHeadAttention...")
    attn = MultiHeadAttention(d_model=512, n_heads=8, n_kv_heads=4)

    # Test with batch
    x = np.random.randn(2, 10, 512).astype(np.float32)
    y = attn.forward(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
    print(f"  ✓ MultiHeadAttention working!")


def test_gated_ffn():
    """Test GatedFFN."""
    print("\nTesting GatedFFN...")
    ffn = GatedFFN(d_model=512, d_ff=2048)

    x = np.random.randn(2, 10, 512).astype(np.float32)
    y = ffn.forward(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
    print(f"  ✓ GatedFFN working!")


def test_transformer_block():
    """Test full TransformerBlock."""
    print("\nTesting TransformerBlock...")
    block = TransformerBlock(d_model=512, n_heads=8, d_ff=2048, n_kv_heads=4)

    x = np.random.randn(2, 10, 512).astype(np.float32)
    y = block.forward(x)
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {y.shape}")
    print(f"  ✓ TransformerBlock working!")


def main():
    print("="*70)
    print("Testing Transformer Components")
    print("="*70)

    test_rmsnorm()
    test_ternary_linear()
    test_attention()
    test_gated_ffn()
    test_transformer_block()

    print("\n" + "="*70)
    print("✅ All transformer components working!")
    print("="*70)


if __name__ == "__main__":
    main()
