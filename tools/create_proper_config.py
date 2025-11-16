#!/usr/bin/env python3
"""
Create proper config.json from GGUF metadata for BitNet model
"""
import json
import os

def create_config():
    model_dir = r"C:\Users\Samuel Howells\.lmstudio\models\microsoft\bitnet-b1.58-2B-4T-gguf"
    
    # Based on the GGUF metadata we extracted
    config = {
        "vocab_size": 128256,
        "hidden_size": 2560,  # bitnet-b1.58.embedding_length
        "num_hidden_layers": 30,  # bitnet-b1.58.block_count  
        "num_attention_heads": 20,  # bitnet-b1.58.attention.head_count
        "num_key_value_heads": 5,  # bitnet-b1.58.attention.head_count_kv
        "intermediate_size": 6912,  # bitnet-b1.58.feed_forward_length
        "rms_norm_eps": 9.999999747378752e-06,  # bitnet-b1.58.attention.layer_norm_rms_epsilon
        "max_position_embeddings": 4096,  # bitnet-b1.58.context_length
        "rope_theta": 500000.0,  # bitnet-b1.58.rope.freq_base
        "model_type": "bitnet-b1.58",
        "architectures": ["BitNetForCausalLM"],
        "bos_token_id": 128000,
        "eos_token_id": 128001,
        "pad_token_id": 128001
    }
    
    config_path = os.path.join(model_dir, "config.json")
    
    print("Creating proper BitNet config...")
    print("Config values:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✅ Config saved to: {config_path}")
    print("✅ Ready for rotor-RAG model loading!")

if __name__ == "__main__":
    create_config()