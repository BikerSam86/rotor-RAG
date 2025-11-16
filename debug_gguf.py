#!/usr/bin/env python3
"""
Debug GGUF parsing
"""
import sys
sys.path.append('src')

from rotor.gguf_parser import GGUFReader

def debug_gguf():
    gguf_file = r"C:\Users\Samuel Howells\.lmstudio\models\microsoft\bitnet-b1.58-2B-4T-gguf\ggml-model-i2_s.gguf"
    
    print("Debugging GGUF file...")
    
    try:
        reader = GGUFReader(gguf_file)
        print(f"Reader created: {type(reader)}")
        
        # Actually read the file
        print("Reading GGUF file...")
        try:
            reader.read()
        except ValueError as e:
            print(f"Tensor type error (expected for BitNet): {e}")
            print("Continuing with metadata only...")
        
        print(f"Version: {reader.version}")
        print(f"Tensor count: {reader.tensor_count}")
        print(f"Metadata count: {reader.metadata_count}")
        
        if reader.metadata:
            print(f"Metadata keys ({len(reader.metadata)}):")
            for key in sorted(reader.metadata.keys()):
                value = reader.metadata[key]
                if isinstance(value, (str, int, float, bool)):
                    print(f"  {key}: {value}")
                else:
                    print(f"  {key}: {type(value)} (length: {len(value) if hasattr(value, '__len__') else 'N/A'})")
        else:
            print("Metadata is empty")
        
        if reader.tensors:
            print(f"Tensors ({len(reader.tensors)}):")
            for i, tensor in enumerate(reader.tensors[:5]):  # Show first 5
                print(f"  {i}: {tensor.name} {tensor.shape} {tensor.tensor_type}")
            if len(reader.tensors) > 5:
                print(f"  ... and {len(reader.tensors) - 5} more")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_gguf()