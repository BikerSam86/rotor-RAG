#!/usr/bin/env python3
"""
Test rotor-RAG with the real BitNet 2B model from LM Studio
"""
import pytest

pytest.skip("Requires access to local BitNet 2B assets", allow_module_level=True)

import sys
import os
sys.path.append('src')

import time
import numpy as np
from rotor.gguf_parser import GGUFReader
from rotor.bitnet_model import load_bitnet_model

def test_real_bitnet_model():
    print("=" * 70)
    print("ROTOR-RAG WITH REAL BITNET 2B MODEL TEST")
    print("=" * 70)
    
    model_dir = r"C:\Users\Samuel Howells\.lmstudio\models\microsoft\bitnet-b1.58-2B-4T-gguf"
    gguf_file = os.path.join(model_dir, "ggml-model-i2_s.gguf")
    
    print(f"Model directory: {model_dir}")
    print(f"GGUF file: {gguf_file}")
    print(f"File size: {os.path.getsize(gguf_file) / (1024**3):.2f} GB")
    
    # Test GGUF parsing
    print("\n[1] Testing GGUF Parser...")
    try:
        reader = GGUFReader(gguf_file)
        metadata = reader.metadata
        
        print(f"  ✓ GGUF metadata parsed")
        print(f"  Model type: {metadata.get('general.name', 'Unknown')}")
        print(f"  Architecture: {metadata.get('general.architecture', 'Unknown')}")
        
        if 'llama.embedding_length' in metadata:
            print(f"  Embedding length: {metadata['llama.embedding_length']}")
        if 'llama.block_count' in metadata:
            print(f"  Block count: {metadata['llama.block_count']}")
            
    except Exception as e:
        print(f"  ✗ GGUF parsing failed: {e}")
        print("  This might be expected - let's try direct model loading...")
    
    # Create a minimal config for testing
    print("\n[2] Creating test configuration...")
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        config = {
            "vocab_size": 128256,
            "d_model": 2560, 
            "n_layers": 30,
            "model_type": "BitNet-b1.58"
        }
        import json
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"  ✓ Created config: {config_path}")
    else:
        print(f"  ✓ Config exists: {config_path}")
    
    # Test model loading
    print("\n[3] Testing model loading...")
    try:
        start_time = time.time()
        model = load_bitnet_model(model_dir, use_gpu=False)  # Start with CPU
        load_time = time.time() - start_time
        
        print(f"  ✓ Model loaded successfully!")
        print(f"  Load time: {load_time:.2f} seconds")
        print(f"  Model layers: {len(model.layers) if hasattr(model, 'layers') else 'Unknown'}")
        
    except Exception as e:
        print(f"  ✗ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    # Test basic inference
    print("\n[4] Testing basic inference...")
    try:
        # Create a simple input
        input_ids = np.array([1, 2, 3, 4, 5])  # Simple token sequence
        
        start_time = time.time()
        output = model.forward(input_ids, use_cache=False)
        inference_time = time.time() - start_time
        
        print(f"  ✓ Forward pass successful!")
        print(f"  Input shape: {input_ids.shape}")
        print(f"  Output shape: {output.shape if hasattr(output, 'shape') else type(output)}")
        print(f"  Inference time: {inference_time*1000:.2f} ms")
        
    except Exception as e:
        print(f"  ✗ Basic inference failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 70)
    print("🎯 REAL BITNET MODEL TEST COMPLETE!")
    print("=" * 70)
    print(f"✅ Successfully loaded 2.4B parameter BitNet model")
    print(f"✅ Model size: {os.path.getsize(gguf_file) / (1024**3):.2f} GB")
    print(f"✅ Load time: {load_time:.2f} seconds")
    print(f"✅ Ready for performance testing!")
    
    return True

if __name__ == "__main__":
    success = test_real_bitnet_model()
    if not success:
        print("\n💥 Test failed. Check the errors above.")
        sys.exit(1)