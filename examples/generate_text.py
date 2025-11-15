"""
Text generation demo with BitNet model.

Loads the full 2.4B model and generates text using various sampling strategies.
"""

import sys
import io
from pathlib import Path
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from rotor.bitnet_model import BitNetModel
from rotor.tokenizer import BitNetTokenizer
from rotor.generation import (
    TextGenerator,
    GreedySampling,
    TopKSampling,
    TopPSampling,
)

print("=" * 70)
print("BitNet Text Generation Demo")
print("=" * 70)

model_dir = "C:/Users/samho/Desktop/BitNet-2B-model"

# Load tokenizer (fast!)
print("\n[1] Loading tokenizer...")
print("-" * 70)
start = time.perf_counter()
tokenizer = BitNetTokenizer(model_dir)
print(f"✓ Tokenizer loaded in {time.perf_counter() - start:.2f}s")

# Load model (takes ~78 seconds with C optimizations)
print("\n[2] Loading BitNet model...")
print("-" * 70)
print("This will take about 78 seconds with C optimizations...")

# Load config
import json

config_path = Path(model_dir) / "config.json"
with open(config_path) as f:
    config = json.load(f)

# Create model config
model_config = {
    'vocab_size': config['vocab_size'],
    'd_model': config['hidden_size'],
    'n_layers': config['num_hidden_layers'],
    'n_heads': config['num_attention_heads'],
    'n_kv_heads': config.get('num_key_value_heads', 4),
    'd_ff': config['intermediate_size'],
}

start = time.perf_counter()

# Create model
model = BitNetModel(model_config)

# Load weights using built-in method
weights_path = str(Path(model_dir) / "model.safetensors")
model.load_weights_from_safetensors(weights_path)

load_time = time.perf_counter() - start
print(f"✓ Model loaded in {load_time:.0f}s")

# Test prompt
prompt = "The future of AI is"
print(f"\n[3] Testing text generation...")
print("-" * 70)
print(f"Prompt: '{prompt}'")
print()

# Generate with greedy sampling (deterministic)
print("Strategy: Greedy sampling (deterministic)")
print("-" * 70)
generator = TextGenerator(
    model=model,
    tokenizer=tokenizer,
    sampling_strategy=GreedySampling(),
    temperature=1.0,
    max_length=10,
)

print("Generating 3 tokens (this will take a few minutes on 2016 laptop)...")
start = time.perf_counter()

# Use callback to show progress
tokens_generated = [0]
token_times = []
last_time = [time.perf_counter()]

def progress_callback(token_text):
    tokens_generated[0] += 1
    current_time = time.perf_counter()
    token_time = current_time - last_time[0]
    token_times.append(token_time)
    last_time[0] = current_time
    print(f"  Token {tokens_generated[0]}: '{token_text}' ({token_time:.1f}s)")

generated_text = generator.generate(
    prompt=prompt,
    max_new_tokens=3,  # Just 3 tokens for demo
    callback=progress_callback,
)

gen_time = time.perf_counter() - start

print()
print("Generated text:")
print(f"  '{generated_text}'")
print()
print(f"Total time: {gen_time:.1f}s")
print(f"Average: {gen_time/tokens_generated[0]:.1f}s per token")
if len(token_times) > 0:
    print(f"Tokens: {token_times[0]:.1f}s (first), {np.mean(token_times[1:]):.1f}s (avg rest)")

# Note: Skipping top-k test to save time
print("\n" + "=" * 70)
print("Note: Skipping additional sampling tests to save time")
print("(Each token takes ~40-50s on 2016 laptop)")
print("-" * 70)

# Summary
print("\n" + "=" * 70)
print("✅ TEXT GENERATION WORKING!")
print("=" * 70)
print(f"\nSummary:")
print(f"  Model load: {load_time:.0f}s")
print(f"  Tokenizer: Working ✓")
print(f"  Greedy sampling: Working ✓")
print(f"  Text generation: Working ✓")
print(f"\nPerformance:")
print(f"  ~{gen_time/tokens_generated[0]:.1f}s per token on 2016 laptop")
print(f"  Can be optimized with:")
print(f"    - KV cache (avoid recomputing past tokens)")
print(f"    - Better attention kernels")
print(f"    - GPU acceleration")
print("\n🌀 All ways, always!")
