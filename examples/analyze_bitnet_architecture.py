"""
Analyze the BitNet model architecture to understand its structure.

This will help us implement the transformer layers correctly.
"""

import sys
import io
from pathlib import Path
import json

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
from safetensors import safe_open
from collections import defaultdict


def analyze_model_structure(model_path):
    """Analyze the model structure from safetensors."""

    print("="*70)
    print("Analyzing BitNet Model Architecture")
    print("="*70)

    # Load model metadata
    with safe_open(model_path, framework="pt") as f:
        keys = f.keys()

        # Group tensors by layer
        layer_groups = defaultdict(list)
        config_params = {}

        for key in keys:
            if key.startswith("model.layers."):
                # Extract layer number
                parts = key.split(".")
                layer_num = int(parts[2])
                layer_groups[layer_num].append(key)
            elif key.startswith("model."):
                # Other model params
                if "embed_tokens" in key or "norm" in key:
                    config_params[key] = f.get_tensor(key).shape

        print(f"\nModel Overview:")
        print(f"  Total tensors: {len(keys)}")
        print(f"  Number of layers: {len(layer_groups)}")

        # Analyze first layer to understand structure
        if 0 in layer_groups:
            print(f"\n" + "="*70)
            print(f"Layer 0 Structure (Representative):")
            print("="*70)

            layer_0_tensors = sorted(layer_groups[0])
            for key in layer_0_tensors:
                tensor = f.get_tensor(key)
                shape_str = "x".join(map(str, tensor.shape))
                print(f"  {key}")
                print(f"    Shape: {shape_str}")
                print(f"    Dtype: {tensor.dtype}")

        # Analyze embedding and norm layers
        print(f"\n" + "="*70)
        print(f"Special Layers:")
        print("="*70)

        for key, shape in sorted(config_params.items()):
            shape_str = "x".join(map(str, shape))
            print(f"  {key}: {shape_str}")

        # Count different layer types
        print(f"\n" + "="*70)
        print(f"Layer Type Counts:")
        print("="*70)

        layer_types = defaultdict(int)
        for key in keys:
            if "self_attn" in key:
                if "weight" in key and not "weight_scale" in key:
                    if key.endswith("weight"):
                        layer_types["attention_weight"] += 1
            elif "mlp" in key:
                if "weight" in key and not "weight_scale" in key:
                    if key.endswith("weight"):
                        layer_types["mlp_weight"] += 1
            elif "norm" in key:
                layer_types["norm"] += 1

        for ltype, count in sorted(layer_types.items()):
            print(f"  {ltype}: {count}")

        # Extract config from model structure
        print(f"\n" + "="*70)
        print(f"Inferred Model Config:")
        print("="*70)

        # Get dimensions from embeddings
        embed_weight = f.get_tensor("model.embed_tokens.weight")
        vocab_size, d_model = embed_weight.shape

        print(f"  vocab_size: {vocab_size}")
        print(f"  d_model (hidden_size): {d_model}")
        print(f"  n_layers: {len(layer_groups)}")

        # Try to infer other params from layer 0
        if 0 in layer_groups:
            # Check attention dimensions
            for key in layer_groups[0]:
                if "q_proj.weight" in key:
                    tensor = f.get_tensor(key)
                    # BitNet packs 4 weights per byte (2 bits each)
                    # So actual dims are 4x the stored dims
                    if tensor.dtype == torch.uint8:
                        stored_shape = tensor.shape
                        actual_rows = stored_shape[0] * 4  # Unpacked
                        print(f"  attention_heads (approx): {actual_rows // 128}")  # Guess

                if "mlp.gate_proj.weight" in key:
                    tensor = f.get_tensor(key)
                    if tensor.dtype == torch.uint8:
                        stored_shape = tensor.shape
                        actual_rows = stored_shape[0] * 4
                        print(f"  intermediate_size (approx): {actual_rows}")

        # Save config
        config = {
            "vocab_size": vocab_size,
            "d_model": d_model,
            "n_layers": len(layer_groups),
            "model_type": "BitNet-b1.58"
        }

        return config, layer_groups


def main():
    print("\n" + "╔" + "═"*68 + "╗")
    print("║" + " "*14 + "BitNet Architecture Analyzer" + " "*25 + "║")
    print("╚" + "═"*68 + "╝\n")

    model_path = Path("C:/Users/samho/Desktop/BitNet-2B-model/model.safetensors")

    if not model_path.exists():
        print(f"❌ Model not found at: {model_path}")
        return

    print(f"✅ Found model: {model_path}")
    print(f"   Size: {model_path.stat().st_size / 1024 / 1024 / 1024:.2f} GB\n")

    config, layer_groups = analyze_model_structure(str(model_path))

    # Save config
    config_path = Path(__file__).parent.parent / "models" / "bitnet_config.json"
    config_path.parent.mkdir(exist_ok=True)

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n✅ Config saved to: {config_path}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n✅ Analyzed Microsoft BitNet-b1.58-2B-4T model")
    print(f"✅ {config['n_layers']} transformer layers")
    print(f"✅ {config['d_model']} hidden dimensions")
    print(f"✅ {config['vocab_size']:,} vocabulary size")
    print(f"\nNext: Implement transformer layers with this architecture!")

    print("\n🌀 All ways, always!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
