"""
Load REAL BitNet ternary weights and convert to Rotor format.

This focuses on the actual uint8 packed ternary weights from Microsoft BitNet.
"""

import sys
import io
from pathlib import Path
import time

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from safetensors import safe_open
from rotor.bitnet import bitnet_to_rotor
import torch


def load_bitnet_weights(model_path):
    """Load only the uint8 ternary weight tensors"""
    print(f"Loading BitNet model from: {model_path}")

    weight_tensors = {}
    scale_tensors = {}

    with safe_open(model_path, framework="pt") as f:
        keys = f.keys()
        print(f"\nFound {len(keys)} tensors")

        # Separate weights and scales
        for key in keys:
            if key.endswith('.weight') and not key.endswith('_scale'):
                tensor = f.get_tensor(key)

                # Only keep uint8 tensors (these are the ternary weights!)
                if tensor.dtype == torch.uint8:
                    weight_tensors[key] = tensor.cpu().numpy()

            elif key.endswith('.weight_scale'):
                tensor = f.get_tensor(key)
                if tensor.dtype == torch.bfloat16:
                    tensor = tensor.float()
                scale_tensors[key] = tensor.cpu().numpy()

    print(f"\nFound {len(weight_tensors)} ternary weight layers")
    print(f"Found {len(scale_tensors)} scale parameters")

    return weight_tensors, scale_tensors


def analyze_bitnet_packed_weights(weight_tensors):
    """Analyze the packed BitNet weights"""
    print("\n" + "="*70)
    print("Analyzing REAL BitNet Packed Weights")
    print("="*70)

    # Pick first few layers
    sample_names = list(weight_tensors.keys())[:5]

    print(f"\nSample layers:")
    for i, name in enumerate(sample_names, 1):
        tensor = weight_tensors[name]
        print(f"\n{i}. {name}")
        print(f"   Shape: {tensor.shape}")
        print(f"   Dtype: {tensor.dtype}")
        print(f"   Size: {tensor.nbytes / 1024:.2f} KB")
        print(f"   Total weights: {np.prod(tensor.shape):,}")

        # Check value distribution (these are PACKED, so each byte contains 4 weights)
        unique_bytes = np.unique(tensor)
        print(f"   Unique byte values: {len(unique_bytes)}")

        # Sample some bytes
        flat = tensor.flatten()
        sample_bytes = flat[:10]
        print(f"   Sample bytes: {sample_bytes}")

        # Decode a few weights to see the pattern
        print(f"   Decoding first 16 weights:")
        decoded = []
        for byte_val in sample_bytes[:4]:  # First 4 bytes = 16 weights
            # Each byte contains 4 weights (2 bits each)
            for shift in [0, 2, 4, 6]:
                two_bits = (byte_val >> shift) & 0b11
                # BitNet encoding: 00=0, 10=+1, 01=-1, 11=error
                if two_bits == 0b00:
                    decoded.append(0)
                elif two_bits == 0b10:
                    decoded.append(+1)
                elif two_bits == 0b01:
                    decoded.append(-1)
                else:
                    decoded.append('?')

        print(f"   Decoded weights: {decoded}")

        # Count ternary distribution (approximate from bytes)
        zeros = np.sum(tensor == 0b00000000)
        print(f"   All-zero bytes: {zeros} ({zeros/tensor.size*100:.1f}%)")


def convert_bitnet_layer_to_rotor(packed_weights, layer_name):
    """Convert a BitNet packed layer to Rotor format"""
    print(f"\n" + "="*70)
    print(f"Converting: {layer_name}")
    print("="*70)

    print(f"\nInput:")
    print(f"  Shape: {packed_weights.shape}")
    print(f"  Dtype: {packed_weights.dtype}")
    print(f"  Size: {packed_weights.nbytes / 1024:.2f} KB")

    # The weights are ALREADY in BitNet packed format!
    # We just need to convert to our Rotor format

    start = time.perf_counter()
    bit0, bit1 = bitnet_to_rotor(packed_weights, validate=False)
    convert_time = (time.perf_counter() - start) * 1000

    print(f"\nOutput (Rotor format):")
    print(f"  bit0 shape: {bit0.shape}")
    print(f"  bit1 shape: {bit1.shape}")
    print(f"  Total size: {(bit0.nbytes + bit1.nbytes) / 1024:.2f} KB")
    print(f"  Conversion time: {convert_time:.2f} ms")

    return bit0, bit1


def main():
    print("\n" + "╔" + "═"*68 + "╗")
    print("║" + " "*12 + "REAL BitNet Weight Extractor" + " "*27 + "║")
    print("║" + " "*8 + "Microsoft BitNet-b1.58-2B-4T Model" + " "*25 + "║")
    print("╚" + "═"*68 + "╝\n")

    # Model path
    model_path = Path("C:/Users/samho/Desktop/BitNet-2B-model/model.safetensors")

    if not model_path.exists():
        print(f"❌ Model not found at: {model_path}")
        return

    print(f"✅ Found model: {model_path}")
    print(f"   Size: {model_path.stat().st_size / 1024 / 1024 / 1024:.2f} GB")

    # Load model
    start = time.perf_counter()
    weight_tensors, scale_tensors = load_bitnet_weights(str(model_path))
    load_time = time.perf_counter() - start

    print(f"\n✅ Loaded {len(weight_tensors)} weight layers in {load_time:.2f}s")

    # Analyze weights
    analyze_bitnet_packed_weights(weight_tensors)

    # Convert a sample layer
    if weight_tensors:
        sample_name = list(weight_tensors.keys())[0]
        sample_weights = weight_tensors[sample_name]

        bit0, bit1 = convert_bitnet_layer_to_rotor(sample_weights, sample_name)

        print(f"\n✅ Successfully converted REAL BitNet weights to Rotor format!")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY - REAL BitNet Ternary Weights")
    print("="*70)
    print(f"\n✅ Loaded Microsoft BitNet-b1.58-2B-4T model")
    print(f"✅ Extracted {len(weight_tensors)} ternary weight layers")
    print(f"✅ Each layer is ALREADY in BitNet packed format (uint8)")

    total_weight_bytes = sum(t.nbytes for t in weight_tensors.values())
    total_rotor_bytes = total_weight_bytes * 2  # Rotor uses 2 arrays

    print(f"\nMemory:")
    print(f"  BitNet format: {total_weight_bytes / 1024 / 1024:.1f} MB")
    print(f"  Rotor format: {total_rotor_bytes / 1024 / 1024:.1f} MB")
    print(f"  Overhead: {total_rotor_bytes / total_weight_bytes:.1f}×")

    print(f"\n✅ Converted sample layer: {sample_name}")
    print(f"   Shape: {sample_weights.shape}")
    print(f"   BitNet size: {sample_weights.nbytes / 1024:.2f} KB")
    print(f"   Rotor size: {(bit0.nbytes + bit1.nbytes) / 1024:.2f} KB")

    print(f"\n🎉 SUCCESS! Real Microsoft BitNet weights ready for Rotor!")
    print(f"\nNext steps:")
    print(f"  1. Convert all {len(weight_tensors)} layers to Rotor format")
    print(f"  2. Implement full BitNet inference with our kernels")
    print(f"  3. Benchmark inference speed vs bitnet.cpp")
    print(f"  4. Run actual language model generation!")

    print(f"\n🌀 All ways, always!")
    print(f"="*70 + "\n")


if __name__ == "__main__":
    main()
