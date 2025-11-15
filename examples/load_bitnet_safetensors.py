"""
Load Microsoft BitNet model from safetensors format.

This is the REAL deal - loading actual BitNet weights!
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
from rotor.bitnet import encode_bitnet_array, bitnet_to_rotor
import torch


def load_bitnet_safetensors(model_path):
    """Load BitNet model from safetensors file"""
    print(f"Loading BitNet model from: {model_path}")

    tensors = {}
    with safe_open(model_path, framework="pt") as f:  # Use PyTorch for bfloat16 support
        # Get all tensor names
        keys = f.keys()
        print(f"\nFound {len(keys)} tensors")

        # Load tensors
        print("\nLoading tensors...")
        for key in keys:
            tensor = f.get_tensor(key)
            # Convert to numpy (and to float32 if bfloat16)
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.float()  # Convert to float32
            tensors[key] = tensor.cpu().numpy()

            # Print first few
            if len(tensors) <= 10:
                print(f"  {key}: shape={tensors[key].shape}, dtype={tensors[key].dtype}")

        if len(tensors) > 10:
            print(f"  ... and {len(tensors) - 10} more tensors")

    return tensors


def analyze_bitnet_weights(tensors):
    """Analyze the weight distribution"""
    print("\n" + "="*70)
    print("Analyzing BitNet Weights")
    print("="*70)

    # Find typical transformer layers
    attn_layers = [k for k in tensors.keys() if 'attn' in k.lower() and 'weight' in k.lower()]
    ffn_layers = [k for k in tensors.keys() if ('ffn' in k.lower() or 'mlp' in k.lower()) and 'weight' in k.lower()]

    print(f"\nLayer types:")
    print(f"  Attention layers: {len(attn_layers)}")
    print(f"  FFN/MLP layers: {len(ffn_layers)}")

    # Analyze a sample layer
    if attn_layers:
        sample_name = attn_layers[0]
        sample_tensor = tensors[sample_name]

        print(f"\nSample layer: {sample_name}")
        print(f"  Shape: {sample_tensor.shape}")
        print(f"  Dtype: {sample_tensor.dtype}")
        print(f"  Min: {sample_tensor.min()}")
        print(f"  Max: {sample_tensor.max()}")
        print(f"  Mean: {sample_tensor.mean():.4f}")

        # Check if already ternary
        unique_values = np.unique(sample_tensor)
        print(f"  Unique values: {unique_values[:10]}")

        # Count ternary values
        if len(unique_values) <= 10:
            print(f"\n  Value distribution:")
            for val in unique_values:
                count = (sample_tensor == val).sum()
                pct = count / sample_tensor.size * 100
                print(f"    {val}: {count:,} ({pct:.2f}%)")

    return attn_layers, ffn_layers


def convert_layer_to_rotor(tensor_data, layer_name):
    """Convert a single layer to Rotor format"""
    print(f"\nConverting {layer_name}...")
    print(f"  Input shape: {tensor_data.shape}")
    print(f"  Input dtype: {tensor_data.dtype}")

    # Check if already ternary
    unique_vals = np.unique(tensor_data)
    print(f"  Unique values: {len(unique_vals)}")

    # Quantize to ternary if needed
    if len(unique_vals) > 3 or not np.all(np.isin(unique_vals, [-1, 0, 1])):
        print(f"  ⚠️  Not ternary! Quantizing...")

        # Simple ternary quantization
        # threshold = np.percentile(np.abs(tensor_data), 20)  # Keep top 80%
        threshold = 0.0

        ternary_data = np.zeros_like(tensor_data, dtype=np.int8)
        ternary_data[tensor_data > threshold] = 1
        ternary_data[tensor_data < -threshold] = -1

        # Check distribution
        zeros = (ternary_data == 0).sum()
        pos = (ternary_data == 1).sum()
        neg = (ternary_data == -1).sum()

        print(f"  After quantization:")
        print(f"    Zeros: {zeros:,} ({zeros/ternary_data.size*100:.1f}%)")
        print(f"    +1:    {pos:,} ({pos/ternary_data.size*100:.1f}%)")
        print(f"    -1:    {neg:,} ({neg/ternary_data.size*100:.1f}%)")

        tensor_data = ternary_data

    # Encode to BitNet format
    start = time.perf_counter()
    bitnet_packed = encode_bitnet_array(tensor_data.astype(np.int8))
    encode_time = (time.perf_counter() - start) * 1000

    print(f"  Encoded to BitNet: {bitnet_packed.shape}, {bitnet_packed.nbytes/1024:.2f} KB")
    print(f"    Time: {encode_time:.2f} ms")

    # Convert to Rotor format
    start = time.perf_counter()
    bit0, bit1 = bitnet_to_rotor(bitnet_packed, validate=False)
    convert_time = (time.perf_counter() - start) * 1000

    print(f"  Converted to Rotor: bit0={bit0.shape}, bit1={bit1.shape}")
    print(f"    Memory: {(bit0.nbytes + bit1.nbytes)/1024:.2f} KB")
    print(f"    Time: {convert_time:.2f} ms")

    return bit0, bit1


def main():
    print("\n" + "╔" + "═"*68 + "╗")
    print("║" + " "*12 + "REAL BitNet Model Loader" + " "*31 + "║")
    print("║" + " "*8 + "Microsoft BitNet-b1.58-2B-4T Model" + " "*25 + "║")
    print("╚" + "═"*68 + "╝\n")

    # Model path
    model_path = Path("C:/Users/samho/Desktop/BitNet-2B-model/model.safetensors")

    if not model_path.exists():
        print(f"❌ Model not found at: {model_path}")
        print(f"\n📥 Downloading BitNet model (~2.4GB)...")
        print(f"\nRun this command:")
        print(f'  python -c "from huggingface_hub import hf_hub_download; '
              f'hf_hub_download(repo_id=\'microsoft/BitNet-b1.58-2B-4T\', '
              f'filename=\'model.safetensors\', local_dir=\'BitNet-2B-model\')"')
        print(f"\nThis will take a few minutes depending on your internet speed...")
        return

    print(f"✅ Found model: {model_path}")
    print(f"   Size: {model_path.stat().st_size / 1024 / 1024 / 1024:.2f} GB")

    # Load model
    print("\n" + "="*70)
    print("Step 1: Loading SafeTensors File")
    print("="*70)

    start = time.perf_counter()
    tensors = load_bitnet_safetensors(str(model_path))
    load_time = time.perf_counter() - start

    print(f"\n✅ Loaded {len(tensors)} tensors in {load_time:.2f}s")

    # Analyze weights
    attn_layers, ffn_layers = analyze_bitnet_weights(tensors)

    # Convert a sample layer
    if attn_layers or ffn_layers:
        print("\n" + "="*70)
        print("Step 2: Converting Sample Layer to Rotor Format")
        print("="*70)

        # Pick first attention layer
        if attn_layers:
            sample_name = attn_layers[0]
        else:
            sample_name = ffn_layers[0]

        sample_tensor = tensors[sample_name]

        bit0, bit1 = convert_layer_to_rotor(sample_tensor, sample_name)

        print(f"\n✅ Successfully converted REAL BitNet weights to Rotor format!")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY - REAL BitNet Model")
    print("="*70)
    print(f"\n✅ Loaded Microsoft BitNet-b1.58-2B-4T model")
    print(f"✅ Model has {len(tensors)} tensors")
    print(f"✅ Total parameters: ~2.4 billion")

    total_memory_fp32 = sum(t.nbytes for t in tensors.values()) / 1024 / 1024
    total_memory_ternary = total_memory_fp32 / 16  # 2 bits vs 32 bits

    print(f"\nMemory:")
    print(f"  FP32 (original): {total_memory_fp32:.1f} MB")
    print(f"  Ternary (2-bit): {total_memory_ternary:.1f} MB")
    print(f"  Compression: {total_memory_fp32/total_memory_ternary:.1f}×")

    if attn_layers:
        print(f"\n✅ Converted sample layer to Rotor format")
        print(f"   Layer: {sample_name}")
        print(f"   Shape: {sample_tensor.shape}")
        print(f"   Rotor format: {(bit0.nbytes + bit1.nbytes)/1024:.2f} KB")

    print(f"\n🎉 SUCCESS! Real BitNet weights loaded and converted!")
    print(f"\nNext steps:")
    print(f"  1. Convert all layers to Rotor format")
    print(f"  2. Run full model inference")
    print(f"  3. Benchmark vs Microsoft bitnet.cpp")
    print(f"  4. Deploy to edge devices!")

    print(f"\n🌀 All ways, always!")
    print(f"="*70 + "\n")


if __name__ == "__main__":
    main()
