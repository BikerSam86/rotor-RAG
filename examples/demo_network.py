"""
Demo: Simple ternary neural network using rotor encoding.

This demonstrates:
1. Creating a ternary network
2. Running forward pass
3. Inspecting weight statistics
"""

import numpy as np
import sys
from pathlib import Path
import io

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rotor.layers import SimpleRotorNet, TernaryLinear
from rotor.quantization import quantize_ternary


def demo_simple_network():
    """Demonstrate simple rotor network."""
    print("\n" + "="*60)
    print("Demo: Simple Ternary Neural Network")
    print("="*60 + "\n")

    # Create network
    print("Creating network: 128 -> 64 -> 10")
    net = SimpleRotorNet(
        input_dim=128,
        hidden_dim=64,
        output_dim=10
    )

    # Print network stats
    print("\nNetwork Statistics:")
    stats = net.get_stats()

    print(f"\nLayer 1 (128 -> 64):")
    l1 = stats['layer1_weights']
    print(f"  Total weights: {l1['total']}")
    print(f"  +1 weights: {l1['positives']} ({l1['positives']/l1['total']*100:.1f}%)")
    print(f"  -1 weights: {l1['negatives']} ({l1['negatives']/l1['total']*100:.1f}%)")
    print(f"   0 weights: {l1['zeros']} ({l1['zeros']/l1['total']*100:.1f}%)")
    print(f"  Sparsity: {l1['sparsity']*100:.1f}%")

    print(f"\nLayer 2 (64 -> 10):")
    l2 = stats['layer2_weights']
    print(f"  Total weights: {l2['total']}")
    print(f"  +1 weights: {l2['positives']} ({l2['positives']/l2['total']*100:.1f}%)")
    print(f"  -1 weights: {l2['negatives']} ({l2['negatives']/l2['total']*100:.1f}%)")
    print(f"   0 weights: {l2['zeros']} ({l2['zeros']/l2['total']*100:.1f}%)")
    print(f"  Sparsity: {l2['sparsity']*100:.1f}%")

    # Calculate memory usage
    total_weights = l1['total'] + l2['total']
    memory_bytes = total_weights * 2 / 8  # 2 bits per weight
    memory_fp16 = total_weights * 2  # 2 bytes per fp16 weight

    print(f"\nMemory Comparison:")
    print(f"  Ternary (2-bit): {memory_bytes:.0f} bytes ({memory_bytes/1024:.2f} KB)")
    print(f"  FP16: {memory_fp16:.0f} bytes ({memory_fp16/1024:.2f} KB)")
    print(f"  Compression: {memory_fp16/memory_bytes:.1f}x smaller")

    # Test forward pass
    print("\n" + "-"*60)
    print("Testing Forward Pass")
    print("-"*60)

    # Single input
    print("\n1. Single input vector:")
    x_single = np.random.randn(128).astype(np.float32)
    output_single = net.forward(x_single)
    print(f"   Input shape: {x_single.shape}")
    print(f"   Output shape: {output_single.shape}")
    print(f"   Output logits: {output_single}")
    print(f"   Predicted class: {net.predict(x_single)}")

    # Batch input
    print("\n2. Batch input (32 samples):")
    x_batch = np.random.randn(32, 128).astype(np.float32)
    output_batch = net.forward(x_batch)
    predictions = net.predict(x_batch)
    print(f"   Input shape: {x_batch.shape}")
    print(f"   Output shape: {output_batch.shape}")
    print(f"   First 5 predictions: {predictions[:5]}")

    print("\n" + "="*60)
    print("Demo completed successfully! ✓")
    print("="*60 + "\n")


def demo_single_layer():
    """Demonstrate single ternary layer."""
    print("\n" + "="*60)
    print("Demo: Single Ternary Layer")
    print("="*60 + "\n")

    # Create layer
    layer = TernaryLinear(in_features=10, out_features=5, bias=True)

    print("Layer: 10 -> 5")
    print(f"Weight stats: {layer.get_weight_stats()}")

    # Test with different inputs
    print("\nTesting with different inputs:")

    # Test 1: Positive input
    x1 = np.ones(10)
    y1 = layer(x1)
    print(f"\n1. All +1 input: {y1}")

    # Test 2: Negative input
    x2 = -np.ones(10)
    y2 = layer(x2)
    print(f"2. All -1 input: {y2}")

    # Test 3: Mixed input
    x3 = np.array([1, -1, 0, 1, -1, 0, 1, -1, 0, 1], dtype=np.float32)
    y3 = layer(x3)
    print(f"3. Mixed input: {y3}")

    # Test 4: Small random values (will quantize to mostly zeros)
    x4 = np.random.randn(10) * 0.1
    y4 = layer(x4)
    print(f"4. Small random: {y4}")

    print("\n" + "="*60)


def demo_quantization():
    """Demonstrate quantization behavior."""
    print("\n" + "="*60)
    print("Demo: Ternary Quantization")
    print("="*60 + "\n")

    # Original values
    x = np.array([
        0.9, 0.5, 0.2, 0.05, 0.0,
        -0.05, -0.2, -0.5, -0.9, -1.5
    ])

    print("Original values:")
    print(x)

    # Quantize with different thresholds
    print("\nQuantized with threshold=0.0:")
    q0 = quantize_ternary(x, threshold=0.0)
    print(q0)

    print("\nQuantized with threshold=0.1:")
    q1 = quantize_ternary(x, threshold=0.1)
    print(q1)

    print("\nQuantized with threshold=0.3:")
    q3 = quantize_ternary(x, threshold=0.3)
    print(q3)

    print("\n" + "="*60)


if __name__ == "__main__":
    demo_simple_network()
    demo_single_layer()
    demo_quantization()
