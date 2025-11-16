#!/usr/bin/env python3
"""
Extract proper config from GGUF file for rotor-RAG
"""
import sys
import os
sys.path.append('src')

import json
from rotor.gguf_parser import GGUFReader

def extract_config_from_gguf():
    gguf_file = r"C:\Users\Samuel Howells\.lmstudio\models\microsoft\bitnet-b1.58-2B-4T-gguf\ggml-model-i2_s.gguf"
    model_dir = os.path.dirname(gguf_file)
    
    print("=" * 70)
    print("EXTRACTING CONFIG FROM GGUF")
    print("=" * 70)
    
    # Parse GGUF metadata
    reader = GGUFReader(gguf_file)
    metadata = reader.metadata
    
    print("Available metadata keys:")
    for key in sorted(metadata.keys()):
        print(f"  {key}: {metadata[key]}")
    
    # Extract config values needed by rotor-RAG
    config = {}
    
    # Map GGUF metadata to rotor-RAG config
    if 'llama.vocab_size' in metadata:
        config['vocab_size'] = metadata['llama.vocab_size']
    if 'llama.embedding_length' in metadata:
        config['hidden_size'] = metadata['llama.embedding_length']
    if 'llama.block_count' in metadata:
        config['num_hidden_layers'] = metadata['llama.block_count']
    if 'llama.attention.head_count' in metadata:
        config['num_attention_heads'] = metadata['llama.attention.head_count']
    if 'llama.attention.head_count_kv' in metadata:
        config['num_key_value_heads'] = metadata['llama.attention.head_count_kv']
    if 'llama.feed_forward_length' in metadata:
        config['intermediate_size'] = metadata['llama.feed_forward_length']
    
    print("\nExtracted config:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Save config
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✅ Config saved to: {config_path}")
    
    return config

if __name__ == "__main__":
    extract_config_from_gguf()