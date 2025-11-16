"""
Test PyTorch ternary layers.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("PyTorch not installed. Skipping torch tests.")


def test_ternary_quantize():
    """Test ternary quantization function."""
    if not HAS_TORCH:
        return

    from rotor.torch import TernaryQuantize

    print("Testing TernaryQuantize...")

    # Test values
    x = torch.tensor([0.9, 0.5, 0.2, 0.0, -0.2, -0.5, -0.9])

    # Quantize with threshold=0.3
    x_ternary = TernaryQuantize.apply(x, 0.3)

    expected = torch.tensor([1, 1, 0, 0, 0, -1, -1])

    assert torch.equal(x_ternary, expected), f"Expected {expected}, got {x_ternary}"
    print("  ✓ Quantization works correctly")


def test_ternary_linear():
    """Test TernaryLinear layer."""
    if not HAS_TORCH:
        return

    from rotor.torch import TernaryLinear

    print("Testing TernaryLinear...")

    # Create layer
    layer = TernaryLinear(10, 5, bias=True, threshold=0.3)

    # Forward pass
    x = torch.randn(2, 10)  # Batch of 2
    y = layer(x)

    assert y.shape == (2, 5), f"Expected shape (2, 5), got {y.shape}"
    print(f"  ✓ Forward pass works: {x.shape} → {y.shape}")

    # Check weight stats
    stats = layer.get_weight_stats()
    assert stats['total'] == 50, "Wrong total weight count"
    assert stats['zeros'] + stats['positives'] + stats['negatives'] == 50
    print(f"  ✓ Weight stats: {stats['sparsity']*100:.1f}% sparsity")


def test_ternary_mlp():
    """Test complete MLP."""
    if not HAS_TORCH:
        return

    from rotor.torch import TernaryMLP

    print("Testing TernaryMLP...")

    # Create network
    model = TernaryMLP(
        input_dim=784,
        hidden_dims=[128, 64],
        num_classes=10,
        threshold=0.3
    )

    # Forward pass
    x = torch.randn(32, 784)  # Batch of 32 MNIST images
    y = model(x)

    assert y.shape == (32, 10), f"Expected shape (32, 10), got {y.shape}"
    print(f"  ✓ MLP forward pass: {x.shape} → {y.shape}")

    # Get stats
    stats = model.get_stats()
    print(f"  ✓ Model has {len(stats)} ternary layers")
    for layer_name, layer_stats in stats.items():
        print(f"    {layer_name}: {layer_stats['sparsity']*100:.1f}% sparse")


def test_straight_through_estimator():
    """Test that gradients flow correctly."""
    if not HAS_TORCH:
        return

    from rotor.torch import TernaryLinear

    print("Testing Straight-Through Estimator...")

    # Create layer
    layer = TernaryLinear(5, 3, threshold=0.3)

    # Input
    x = torch.randn(1, 5, requires_grad=True)

    # Forward
    y = layer(x)

    # Backward
    loss = y.sum()
    loss.backward()

    # Check gradients exist
    assert x.grad is not None, "No gradient for input"
    assert layer.weight.grad is not None, "No gradient for weights"

    print("  ✓ Gradients flow through quantization")
    print(f"  ✓ Input grad shape: {x.grad.shape}")
    print(f"  ✓ Weight grad shape: {layer.weight.grad.shape}")


def run_all_tests():
    """Run all tests."""
    if not HAS_TORCH:
        print("\n⚠️  PyTorch not available. Install with:")
        print("  pip install torch torchvision")
        return

    print("\n" + "="*70)
    print("Testing PyTorch Ternary Layers")
    print("="*70 + "\n")

    test_ternary_quantize()
    test_ternary_linear()
    test_ternary_mlp()
    test_straight_through_estimator()

    print("\n" + "="*70)
    print("All PyTorch tests passed! ✓")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_all_tests()
