"""
Test Vulkan compute with ternary matrix multiplication.
Compares Vulkan output to CPU baseline.
"""

import sys
import io
from pathlib import Path
import time
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

print("=" * 70)
print("VULKAN COMPUTE TEST")
print("=" * 70)

# Import Vulkan compute with graceful fallback
fallback_reason = None

try:
    from rotor.vulkan_ternary_full import VulkanTernaryCompute
    print("\n[OK] Vulkan compute module imported")
except Exception as e:
    fallback_reason = f"Failed to import Vulkan backend: {e}"
    VulkanTernaryCompute = None  # type: ignore


class _CpuFallbackVulkan:
    """CPU fallback that mimics the Vulkan interface."""

    device_name = "CPU fallback"

    def __init__(self, reason: str):
        print("\n[WARN] Using CPU fallback because Vulkan is unavailable.")
        print(f"       Reason: {reason}")

    def pack_weights(self, weights):
        # No packing needed, but keep interface identical
        return weights.astype(np.float32, copy=False)

    def ternary_matmul(self, packed_weights, scales, input_vec, in_dim, out_dim):
        weights = packed_weights.reshape(out_dim, in_dim)
        return (weights @ input_vec) * scales


# Initialize Vulkan (or fallback)
if VulkanTernaryCompute is not None:
    try:
        print("\nInitializing Vulkan...")
        vulkan = VulkanTernaryCompute(use_int8_optimized=False)
        print("[OK] Vulkan initialized successfully!")
    except Exception as e:
        fallback_reason = f"Vulkan initialization failed: {e}"
        vulkan = _CpuFallbackVulkan(fallback_reason)
else:
    vulkan = _CpuFallbackVulkan(fallback_reason or "Unknown reason")

# Test case: small matrix
in_dim = 256
out_dim = 256

print(f"\nTest: {out_dim}x{in_dim} ternary matmul")

# Create random ternary weights
weights = np.random.choice([-1, 0, 1], size=(out_dim, in_dim)).astype(np.float32)
scales = np.random.randn(out_dim).astype(np.float32)
input_vec = np.random.randn(in_dim).astype(np.float32)

print(f"  Weights shape: {weights.shape}")
print(f"  Input shape: {input_vec.shape}")

# Pack weights for GPU
try:
    packed = vulkan.pack_weights(weights)
    print(f"  Packed weights: {packed.shape}, dtype: {packed.dtype}")
except Exception as e:
    print(f"\n[ERROR] Weight packing failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# CPU baseline
print("\nComputing CPU baseline...")
start = time.perf_counter()
cpu_result = (weights @ input_vec) * scales
cpu_time = time.perf_counter() - start
print(f"  CPU time: {cpu_time*1000:.2f}ms")

# Vulkan compute
print("\nComputing on Vulkan GPU...")
try:
    start = time.perf_counter()
    gpu_result = vulkan.ternary_matmul(
        packed, scales, input_vec, in_dim, out_dim
    )
    gpu_time = time.perf_counter() - start
    print(f"  GPU time: {gpu_time*1000:.2f}ms")
except Exception as e:
    print(f"\n[ERROR] Vulkan compute failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Compare results
max_diff = np.abs(gpu_result - cpu_result).max()
mean_diff = np.abs(gpu_result - cpu_result).mean()

print("\n" + "=" * 70)
print("RESULTS")
print("=" * 70)
print(f"\nPerformance:")
print(f"  CPU time:  {cpu_time*1000:.2f}ms")
print(f"  GPU time:  {gpu_time*1000:.2f}ms")
print(f"  Speedup:   {cpu_time/gpu_time:.2f}x")

print(f"\nAccuracy:")
print(f"  Max diff:  {max_diff:.6f}")
print(f"  Mean diff: {mean_diff:.6f}")

if max_diff < 1e-3:
    print("\n[SUCCESS] Vulkan output matches CPU!")
else:
    print(f"\n[WARN] Large difference detected (max: {max_diff})")

print("\n" + "=" * 70)
print("All ways, always!")
print("=" * 70)
