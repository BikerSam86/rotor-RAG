#!/usr/bin/env python3
"""
Simple RTX 2060 Performance Test
Use working test patterns to measure RTX 2060 vs documented performance
"""
import sys
sys.path.append('src')

import numpy as np
import time
from rotor.vulkan_ternary_full import VulkanTernaryCompute

def simple_benchmark():
    print("=" * 70)
    print("RTX 2060 SIMPLE PERFORMANCE TEST")
    print("=" * 70)
    
    # Test sizes that work well
    test_sizes = [128, 256, 512]
    
    # Initialize Vulkan once
    vulkan = VulkanTernaryCompute()
    print(f"GPU: {vulkan.device_name}")
    
    for size in test_sizes:
        print(f"\nTesting {size}×{size} matrix:")
        
        # Generate test data - small sparsity for realistic workload
        weights = np.random.choice([-1, 0, 1], size=(size, size), p=[0.3, 0.4, 0.3])
        input_vec = np.random.randn(size).astype(np.float32)
        scales = np.ones(size, dtype=np.float32)
        
        # CPU baseline (multiple runs for stable timing)
        cpu_times = []
        for _ in range(10):
            start = time.perf_counter()
            cpu_result = (weights @ input_vec) * scales
            cpu_times.append(time.perf_counter() - start)
        cpu_time = np.mean(cpu_times) * 1000  # ms
        
        # GPU test 
        try:
            packed = vulkan.pack_weights(weights)
            
            # Warmup
            gpu_result = vulkan.ternary_matmul(packed, scales, input_vec, size, size)
            
            # Benchmark
            gpu_times = []
            for _ in range(10):
                start = time.perf_counter()
                gpu_result = vulkan.ternary_matmul(packed, scales, input_vec, size, size)
                gpu_times.append(time.perf_counter() - start)
            gpu_time = np.mean(gpu_times) * 1000  # ms
            
            # Accuracy check
            max_diff = np.max(np.abs(cpu_result - gpu_result))
            speedup = cpu_time / gpu_time
            
            print(f"  CPU time:     {cpu_time:.3f}ms ± {np.std(cpu_times)*1000:.3f}ms")
            print(f"  GPU time:     {gpu_time:.3f}ms ± {np.std(gpu_times)*1000:.3f}ms")
            print(f"  Speedup:      {speedup:.2f}×")
            print(f"  Max diff:     {max_diff:.6f}")
            print(f"  Status:       {'✓' if max_diff < 0.001 else '✗ ACCURACY ISSUE'}")
            
        except Exception as e:
            print(f"  GPU failed: {e}")
    
    # Compare to documented Intel HD 615 performance
    print("\n" + "=" * 70)
    print("COMPARISON TO DOCUMENTED RESULTS")
    print("=" * 70)
    
    # Re-run 256×256 for comparison
    size = 256
    weights = np.random.choice([-1, 0, 1], size=(size, size), p=[0.3, 0.4, 0.3])
    input_vec = np.random.randn(size).astype(np.float32)
    scales = np.ones(size, dtype=np.float32)
    
    # Quick timing
    start = time.perf_counter()
    cpu_result = (weights @ input_vec) * scales
    cpu_time = (time.perf_counter() - start) * 1000
    
    packed = vulkan.pack_weights(weights)
    start = time.perf_counter()
    gpu_result = vulkan.ternary_matmul(packed, scales, input_vec, size, size)
    gpu_time = (time.perf_counter() - start) * 1000
    
    intel_hd615_time = 8.20  # ms documented for 256×256
    
    print(f"256×256 Matrix Performance:")
    print(f"  Intel HD 615 (documented):  {intel_hd615_time:.2f}ms")
    print(f"  RTX 2060 (this machine):    {gpu_time:.2f}ms")
    print(f"  RTX improvement:            {intel_hd615_time/gpu_time:.1f}× faster")
    print(f"  CPU baseline:               {cpu_time:.2f}ms")
    print(f"  RTX speedup over CPU:       {cpu_time/gpu_time:.1f}×")
    
    print("\n" + "=" * 70)
    print("🚀 RTX 2060 TEST COMPLETE!")
    print("=" * 70)

if __name__ == "__main__":
    simple_benchmark()