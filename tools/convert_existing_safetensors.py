#!/usr/bin/env python3
"""
Convert existing safetensors to rotor format
Using the safetensors file you already have
"""
import sys
import io
from pathlib import Path
import time

sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
from safetensors import safe_open
import torch

def convert_safetensors_to_rotor():
    """Convert your existing safetensors file to rotor format"""
    
    model_path = r"C:\Users\Samuel Howells\Google Drive\GitHub TriStar Personal\rotor-RAG\models\model.safetensors"
    
    print("=" * 80)
    print("🔄 CONVERTING SAFETENSORS TO ROTOR FORMAT")
    print("=" * 80)
    print(f"Source: {model_path}")
    print(f"File size: {Path(model_path).stat().st_size / (1024**3):.2f} GB")
    print()
    
    # Load and examine the safetensors
    print("[1] Loading safetensors file...")
    tensors = {}
    
    with safe_open(model_path, framework="pt") as f:
        keys = f.keys()
        print(f"    Found {len(keys)} tensors")
        
        print("\n[2] Analyzing tensor structure...")
        key_list = list(keys)
        for i, key in enumerate(key_list[:10]):  # Show first 10
            tensor = f.get_tensor(key)
            print(f"    {key}: {tensor.shape} {tensor.dtype}")
        if len(key_list) > 10:
            print(f"    ... and {len(key_list) - 10} more tensors")
        
        print(f"\n[3] Converting BFloat16 to Float32...")
        conversion_count = 0
        for key in keys:
            tensor = f.get_tensor(key)
            # Convert bfloat16 to float32 for processing
            if tensor.dtype == torch.bfloat16:
                tensor = tensor.float()
                conversion_count += 1
            tensors[key] = tensor.cpu().numpy()
        
        print(f"    Converted {conversion_count} BFloat16 tensors to Float32")
    
    print(f"\n[4] Applying ternary quantization...")
    # This is where your BitNet conversion would happen
    # Converting float weights to {-1, 0, +1} format
    
    ternary_tensors = {}
    total_params = 0
    ternary_params = 0
    
    for key, tensor in tensors.items():
        if 'weight' in key and tensor.ndim >= 2:  # Only quantize weight matrices
            # Simple threshold-based ternary quantization
            # This mimics what BitNet does
            threshold = 0.1  # Adjust based on your needs
            
            ternary = np.zeros_like(tensor)
            ternary[tensor > threshold] = 1
            ternary[tensor < -threshold] = -1
            # Values between -threshold and threshold become 0
            
            ternary_tensors[key] = ternary.astype(np.int8)
            ternary_params += tensor.size
            total_params += tensor.size
            
            sparsity = np.sum(ternary == 0) / tensor.size
            print(f"    {key}: {tensor.shape} -> {sparsity:.1%} sparse")
        else:
            # Keep non-weight tensors as-is (embeddings, norms, etc.)
            ternary_tensors[key] = tensor
            total_params += tensor.size
    
    print(f"\n[5] Conversion summary:")
    print(f"    Total parameters: {total_params:,}")
    print(f"    Ternary parameters: {ternary_params:,}")
    print(f"    Compression ratio: {ternary_params/total_params:.1%} quantized")
    
    # Calculate memory savings
    original_size = sum(t.nbytes for t in tensors.values()) / (1024**3)
    ternary_size = sum(t.nbytes for t in ternary_tensors.values()) / (1024**3)
    
    print(f"    Memory usage:")
    print(f"      Original: {original_size:.2f} GB")
    print(f"      Ternary: {ternary_size:.2f} GB")
    print(f"      Savings: {((original_size - ternary_size) / original_size * 100):.1f}%")
    
    print(f"\n✅ Conversion complete!")
    print(f"🎯 Ready for rotor-RAG SIMD bit operations!")
    
    return ternary_tensors

if __name__ == "__main__":
    convert_safetensors_to_rotor()