#!/usr/bin/env python3
"""
OpenCL RTX 2060 Performance Test
Test OpenCL GPU acceleration which should work better than Vulkan
"""
import sys
sys.path.append('src')

import numpy as np
import time
from rotor.gpu_ternary import GPUTernaryOps
import pyopencl as cl

def test_opencl_performance():
    print("=" * 70)
    print("RTX 2060 OPENCL PERFORMANCE TEST")
    print("=" * 70)
    
    # Initialize OpenCL
    try:
        gpu_ops = GPUTernaryOps()
        print(f"OpenCL Device: {gpu_ops.device.name}")
        print(f"Compute Units: {gpu_ops.device.max_compute_units}")
        print(f"Global Memory: {gpu_ops.device.global_mem_size // (1024**3)} GB")
    except Exception as e:
        print(f"OpenCL initialization failed: {e}")
        return
    
    # Test various matrix sizes
    test_sizes = [128, 256, 512, 1024]
    
    for size in test_sizes:
        print(f"\nTesting {size}×{size} matrix:")
        
        # Generate ternary weights (realistic sparsity)
        weights = np.random.choice([-1, 0, 1], size=(size, size), p=[0.3, 0.4, 0.3])
        input_vec = np.random.randn(size).astype(np.float32) 
        
        # CPU baseline
        cpu_times = []
        for _ in range(5):
            start = time.perf_counter()
            cpu_result = weights @ input_vec
            cpu_times.append(time.perf_counter() - start)
        cpu_time = np.mean(cpu_times) * 1000  # ms
        
        try:
            # Pack weights for GPU
            packed_weights = gpu_ops.pack_ternary_weights(weights)
            
            # GPU benchmark
            gpu_times = []
            for _ in range(5):
                start = time.perf_counter()
                gpu_result = gpu_ops.ternary_matmul(
                    packed_weights, input_vec, output_dim=size
                )
                gpu_times.append(time.perf_counter() - start)
            gpu_time = np.mean(gpu_times) * 1000  # ms
            
            # Check accuracy
            max_diff = np.max(np.abs(cpu_result - gpu_result))
            speedup = cpu_time / gpu_time
            
            print(f"  CPU time:     {cpu_time:.3f}ms")
            print(f"  GPU time:     {gpu_time:.3f}ms") 
            print(f"  Speedup:      {speedup:.2f}×")
            print(f"  Max diff:     {max_diff:.6f}")
            print(f"  Status:       {'✓' if max_diff < 0.001 else '✗'}")
            
        except Exception as e:
            print(f"  OpenCL failed: {e}")
    
    # Compare to documented performance
    print("\n" + "=" * 70)
    print("RTX 2060 vs DOCUMENTED INTEL HD 615")
    print("=" * 70)
    
    # Test 256×256 like the documentation
    size = 256
    weights = np.random.choice([-1, 0, 1], size=(size, size), p=[0.3, 0.4, 0.3])
    input_vec = np.random.randn(size).astype(np.float32)
    
    # Quick timing
    start = time.perf_counter()
    cpu_result = weights @ input_vec
    cpu_time = (time.perf_counter() - start) * 1000
    
    try:
        packed = gpu_ops.pack_ternary_weights(weights)
        start = time.perf_counter()
        gpu_result = gpu_ops.ternary_matmul(packed, input_vec, output_dim=size)
        gpu_time = (time.perf_counter() - start) * 1000
        
        # Documented Intel HD 615 performance
        intel_hd615_time = 8.20  # ms
        cpu_intel_baseline = 0.78  # ms (from docs)
        
        print(f"256×256 Matrix Results:")
        print(f"  Intel HD 615 GPU (docs):    {intel_hd615_time:.2f}ms")
        print(f"  Intel CPU baseline (docs):  {cpu_intel_baseline:.2f}ms")
        print(f"  RTX 2060 GPU (this test):   {gpu_time:.2f}ms")
        print(f"  Our CPU (i7-10750H):        {cpu_time:.2f}ms")
        print()
        print(f"GPU Comparison:")
        print(f"  RTX 2060 vs Intel HD 615:  {intel_hd615_time/gpu_time:.1f}× faster")
        print()
        print(f"Speedup Analysis:")
        print(f"  RTX 2060 speedup:           {cpu_time/gpu_time:.1f}×")
        print(f"  Intel HD 615 speedup:       {cpu_intel_baseline/intel_hd615_time:.1f}×")
        
    except Exception as e:
        print(f"Comparison test failed: {e}")
    
    print("\n" + "=" * 70)
    print("🚀 OPENCL RTX 2060 TEST COMPLETE!")
    print("=" * 70)

if __name__ == "__main__":
    test_opencl_performance()