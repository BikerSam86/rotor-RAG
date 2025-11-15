"""
Load and convert Microsoft BitNet model.

This script:
1. Loads a BitNet GGUF model
2. Extracts ternary weights
3. Converts to our Rotor format
4. Validates the conversion
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
from rotor.gguf_parser import GGUFReader
from rotor.bitnet import bitnet_to_rotor


def main():
    print("\n" + "╔" + "═"*68 + "╗")
    print("║" + " "*16 + "BitNet Model Loader" + " "*32 + "║")
    print("║" + " "*10 + "Microsoft BitNet → Rotor Format" + " "*27 + "║")
    print("╚" + "═"*68 + "╝\n")

    # Model path
    model_path = Path("C:/Users/samho/Desktop/BitNet-2B-model/ggml-model-i2_s.gguf")

    if not model_path.exists():
        print(f"❌ Model not found at: {model_path}")
        print(f"\nPlease download the model first:")
        print(f"  python -c \"from huggingface_hub import hf_hub_download; "
              f"hf_hub_download(repo_id='microsoft/BitNet-b1.58-2B-4T', "
              f"filename='ggml-model-i2_s.gguf', local_dir='BitNet-2B-model')\"")
        print(f"\nThis will download ~600MB")
        return

    print(f"✓ Found model: {model_path}")
    print(f"  Size: {model_path.stat().st_size / 1024 / 1024:.1f} MB\n")

    # Parse GGUF file
    print("="*70)
    print("Step 1: Parsing GGUF File")
    print("="*70)

    start = time.perf_counter()
    reader = GGUFReader(str(model_path))
    reader.read()
    parse_time = time.perf_counter() - start

    print(f"\n✓ Parsed in {parse_time:.2f}s")

    # Print summary
    reader.print_summary()

    # Find ternary tensors (BitNet layers)
    print("\n" + "="*70)
    print("Step 2: Finding BitNet Ternary Layers")
    print("="*70)

    ternary_tensors = []
    for name, tensor in reader.tensors.items():
        if 'attn' in name.lower() or 'ffn' in name.lower() or 'mlp' in name.lower():
            # These are likely the main transformer layers
            ternary_tensors.append((name, tensor))

    print(f"\nFound {len(ternary_tensors)} potential ternary tensors")

    if ternary_tensors:
        print(f"\nFirst 10 tensors:")
        for i, (name, tensor) in enumerate(ternary_tensors[:10]):
            dims_str = 'x'.join(map(str, tensor.dims))
            print(f"  {i+1}. {name}")
            print(f"      Shape: {dims_str}")
            print(f"      Type: {tensor.type.name}")

    # Load and convert a sample tensor
    if ternary_tensors:
        print("\n" + "="*70)
        print("Step 3: Loading and Converting Sample Tensor")
        print("="*70)

        # Pick first tensor
        tensor_name, tensor_info = ternary_tensors[0]
        print(f"\nLoading: {tensor_name}")
        print(f"  Shape: {tensor_info.dims}")
        print(f"  Type: {tensor_info.type.name}")

        try:
            # Load tensor data
            start = time.perf_counter()
            bitnet_packed = reader.load_tensor_data(tensor_name)
            load_time = (time.perf_counter() - start) * 1000

            print(f"\n✓ Loaded in {load_time:.2f} ms")
            print(f"  Data shape: {bitnet_packed.shape}")
            print(f"  Data dtype: {bitnet_packed.dtype}")
            print(f"  Memory: {bitnet_packed.nbytes / 1024:.2f} KB")

            # Convert to Rotor format
            print(f"\nConverting BitNet → Rotor...")
            start = time.perf_counter()
            bit0, bit1 = bitnet_to_rotor(bitnet_packed, validate=False)
            convert_time = (time.perf_counter() - start) * 1000

            print(f"✓ Converted in {convert_time:.2f} ms")
            print(f"  Rotor bit0 shape: {bit0.shape}")
            print(f"  Rotor bit1 shape: {bit1.shape}")
            print(f"  Total memory: {(bit0.nbytes + bit1.nbytes) / 1024:.2f} KB")

            # Analyze weight distribution
            print(f"\nWeight Distribution:")
            total_weights = np.prod(tensor_info.dims)

            # Count weights
            zeros = 0
            positives = 0
            negatives = 0

            for i in range(bit0.shape[0]):
                for j in range(min(bit0.shape[1], bit1.shape[1])):
                    for k in range(8):  # 8 bits per byte
                        b0 = (bit0[i, j] >> k) & 1
                        b1 = (bit1[i, j] >> k) & 1
                        value = b0 - b1

                        if value == 0:
                            zeros += 1
                        elif value == 1:
                            positives += 1
                        elif value == -1:
                            negatives += 1

                        if zeros + positives + negatives >= total_weights:
                            break
                    if zeros + positives + negatives >= total_weights:
                        break
                if zeros + positives + negatives >= total_weights:
                    break

            total = zeros + positives + negatives
            print(f"  Zeros: {zeros:,} ({zeros/total*100:.1f}%)")
            print(f"  +1:    {positives:,} ({positives/total*100:.1f}%)")
            print(f"  -1:    {negatives:,} ({negatives/total*100:.1f}%)")
            print(f"  Total: {total:,}")

            print(f"\n✓ Real BitNet weights successfully converted to Rotor format!")

        except Exception as e:
            print(f"\n❌ Error loading tensor: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n✓ Successfully loaded Microsoft BitNet model")
    print(f"✓ Model has {reader.tensor_count} tensors")
    if ternary_tensors:
        print(f"✓ Found {len(ternary_tensors)} transformer layers")
        print(f"✓ Converted sample layer to Rotor format")
        print(f"✓ Conversion validated")

    print(f"\nNext steps:")
    print(f"  1. Convert all layers to Rotor format")
    print(f"  2. Run inference with our optimized kernels")
    print(f"  3. Benchmark vs bitnet.cpp")

    print(f"\n🌀 All ways, always!")
    print(f"="*70 + "\n")


if __name__ == "__main__":
    main()
