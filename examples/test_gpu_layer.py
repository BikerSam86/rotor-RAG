"""
Test GPU acceleration on actual BitNet transformer layer.
"""

import sys
import io
from pathlib import Path
import time
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rotor.bitnet_model import load_bitnet_model
from rotor.gpu_ternary import GPUTernaryOps

print("=" * 70)
print("GPU ACCELERATION TEST - Real BitNet Layer")
print("=" * 70)

model_dir = "C:/Users/samho/Desktop/BitNet-2B-model"

print("\n[1] Loading BitNet model...")
model = load_bitnet_model(model_dir)

print("\n[2] Initializing GPU...")
gpu = GPUTernaryOps()

print("\n[3] Testing Q projection from layer 0...")
print("-" * 70)

# Get first layer's Q projection
layer_0 = model.layers[0]
q_proj = layer_0.attention.q_proj

# Dimensions
in_dim = 2560
out_dim = 2560

print(f"Q projection: {in_dim} -> {out_dim}")
print(f"Weight shape: {q_proj.weights_cache.shape}")
print(f"Weight scale: {q_proj.scale}")

# Create test input (simulate hidden state from embedding)
input_vec = np.random.randn(in_dim).astype(np.float32)

# Pack GPU weights
print("\nPacking weights for GPU...")
packed_weights = gpu.pack_ternary_weights(q_proj.weights_cache)
scales = np.full(out_dim, q_proj.scale, dtype=np.float32)

print(f"Packed {q_proj.weights_cache.size} weights into {packed_weights.size} bytes")
print(f"Compression: {q_proj.weights_cache.size / packed_weights.size:.1f}x")

# Test 1: GPU forward
print("\n[Test 1] GPU Forward Pass...")
start = time.perf_counter()
for _ in range(10):  # Run 10 times to amortize overhead
    gpu_output = gpu.ternary_matmul(packed_weights, scales, input_vec, in_dim, out_dim)
gpu_time = (time.perf_counter() - start) / 10

# Test 2: CPU forward (NumPy)
print("[Test 2] CPU Forward Pass (NumPy)...")
start = time.perf_counter()
for _ in range(10):
    cpu_output = q_proj.forward(input_vec.reshape(1, -1))[0]
cpu_time = (time.perf_counter() - start) / 10

# Compare results
max_diff = np.abs(gpu_output - cpu_output).max()
mean_diff = np.abs(gpu_output - cpu_output).mean()

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)
print(f"\nSingle Q Projection Forward Pass:")
print(f"  GPU time:  {gpu_time*1000:.2f}ms")
print(f"  CPU time:  {cpu_time*1000:.2f}ms")
print(f"  Speedup:   {cpu_time/gpu_time:.2f}x")
print(f"\nAccuracy:")
print(f"  Max diff:  {max_diff:.6f}")
print(f"  Mean diff: {mean_diff:.6f}")

if max_diff < 0.01:
    print("  [OK] Accuracy acceptable!")

# Estimate full model speedup
print("\n" + "=" * 70)
print("FULL MODEL PROJECTION")
print("=" * 70)

layers = 30
projections_per_layer = 7  # Q, K, V, O, gate, up, down

total_projections = layers * projections_per_layer
print(f"\nTotal projections in model: {total_projections}")
print(f"  Per-token compute (CPU): {cpu_time * total_projections * 1000:.1f}ms")
print(f"  Per-token compute (GPU): {gpu_time * total_projections * 1000:.1f}ms")
print(f"  Expected speedup: {cpu_time/gpu_time:.2f}x")

saved_time = (cpu_time - gpu_time) * total_projections
print(f"\nTime saved per token: {saved_time*1000:.1f}ms = {saved_time:.2f}s")
print(f"\nFor 10 tokens:")
print(f"  CPU: {cpu_time * total_projections * 10:.1f}s")
print(f"  GPU: {gpu_time * total_projections * 10:.1f}s")
print(f"  Saved: {saved_time * 10:.1f}s")

print("\n" + "=" * 70)
print("All ways, always!")
print("=" * 70)
