#!/usr/bin/env python3
"""
RTX 2060 Benchmark Test
Test GPU acceleration with larger matrices that benefit from parallel processing
"""
import sys
import os
sys.path.append('src')

import numpy as np
import time
from rotor.vulkan_ternary_full import VulkanTernaryCompute

def generate_ternary_weights(shape, sparsity=0.6):
    """Generate random ternary weights with given sparsity"""
    total = np.prod(shape)
    weights = np.random.choice([-1, 0, 1], size=total, p=[0.2, sparsity, 0.2])
    return weights.reshape(shape)

def benchmark_matrix_sizes():
    """Test different matrix sizes to find GPU sweet spot"""
    print("=" * 70)
    print("RTX 2060 TERNARY MATRIX BENCHMARK")
    print("=" * 70)
    
    # Test progressively larger sizes
    sizes = [
        (256, 256),     # Small - baseline
        (512, 512),     # Medium
        (1024, 1024),   # Large 
        (2048, 2048),   # Very large
        (2560, 2560),   # BitNet layer size
    ]
    
    results = []
    
    for rows, cols in sizes:
        print(f"\nTesting {rows}×{cols} matrix ({rows*cols:,} elements)")
        
        # Generate test data
        weights = generate_ternary_weights((rows, cols), sparsity=0.6)
        input_vec = np.random.randn(cols).astype(np.float32)
        
        # CPU baseline
        start = time.time()
        for _ in range(5):  # Multiple runs for stability
            cpu_result = np.dot(weights, input_vec)
        cpu_time = (time.time() - start) / 5
        
        try:
            # Test Vulkan GPU
            vulkan = VulkanTernaryCompute()
            packed = vulkan.pack_weights(weights)
            scales = np.ones(rows, dtype=np.float32)
            
            # Warmup
            gpu_result = vulkan.ternary_matmul(packed, scales, input_vec, cols, rows)
            
            # Benchmark
            start = time.time()
            for _ in range(5):
                gpu_result = vulkan.ternary_matmul(packed, scales, input_vec, cols, rows)
            gpu_time = (time.time() - start) / 5
            
            # Verify accuracy
            max_diff = np.max(np.abs(cpu_result - gpu_result))
            speedup = cpu_time / gpu_time
            
            results.append({
                'size': f"{rows}×{cols}",
                'elements': rows * cols,
                'cpu_time': cpu_time * 1000,  # ms
                'gpu_time': gpu_time * 1000,  # ms
                'speedup': speedup,
                'max_diff': max_diff,
                'accurate': max_diff < 0.001
            })
            
            print(f"  CPU time: {cpu_time*1000:.2f}ms")
            print(f"  GPU time: {gpu_time*1000:.2f}ms")
            print(f"  Speedup: {speedup:.2f}×")
            print(f"  Max diff: {max_diff:.6f}")
            print(f"  Status: {'✓' if max_diff < 0.001 else '✗'}")
            
        except Exception as e:
            print(f"  GPU test failed: {e}")
            results.append({
                'size': f"{rows}×{cols}",
                'elements': rows * cols,
                'cpu_time': cpu_time * 1000,
                'gpu_time': 'ERROR',
                'speedup': 'N/A',
                'max_diff': 'N/A',
                'accurate': False
            })
    
    # Summary table
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"{'Size':<12} {'Elements':<10} {'CPU (ms)':<10} {'GPU (ms)':<10} {'Speedup':<8} {'Status':<8}")
    print("-" * 70)
    
    for r in results:
        status = "✓" if r['accurate'] else "✗"
        gpu_time_str = f"{r['gpu_time']:.2f}" if isinstance(r['gpu_time'], float) else str(r['gpu_time'])
        speedup_str = f"{r['speedup']:.2f}×" if isinstance(r['speedup'], float) else str(r['speedup'])
        
        print(f"{r['size']:<12} {r['elements']:<10,} {r['cpu_time']:<10.2f} {gpu_time_str:<10} {speedup_str:<8} {status:<8}")

def benchmark_vs_documented():
    """Compare performance vs documented Intel HD 615 results"""
    print("\n" + "=" * 70)
    print("RTX 2060 vs DOCUMENTED PERFORMANCE")
    print("=" * 70)
    
    # Documented Intel HD 615 results (from project docs)
    intel_hd615_single = 8.20  # ms for 256×256
    intel_hd615_batch = 3.25   # speedup for batched operations
    
    # Test same size as documented
    size = 256
    weights = generate_ternary_weights((size, size), sparsity=0.6) 
    input_vec = np.random.randn(size).astype(np.float32)
    
    # CPU baseline
    start = time.time()
    for _ in range(10):
        cpu_result = np.dot(weights, input_vec)
    cpu_time = (time.time() - start) / 10
    
    # RTX 2060 test
    vulkan = VulkanTernaryCompute()
    packed = vulkan.pack_weights(weights)
    scales = np.ones(size, dtype=np.float32)
    
    # Warmup + benchmark
    vulkan.ternary_matmul(packed, scales, input_vec, size, size)
    start = time.time()
    for _ in range(10):
        gpu_result = vulkan.ternary_matmul(packed, scales, input_vec, size, size) 
    rtx_time = (time.time() - start) / 10
    
    print(f"256×256 Matrix Comparison:")
    print(f"  Intel HD 615 (documented): {intel_hd615_single:.2f}ms")
    print(f"  RTX 2060 (this test):     {rtx_time*1000:.2f}ms")
    print(f"  RTX improvement:          {intel_hd615_single/(rtx_time*1000):.1f}× faster")
    print(f"  CPU baseline:             {cpu_time*1000:.2f}ms")
    print(f"  RTX speedup over CPU:     {cpu_time/rtx_time:.1f}×")

if __name__ == "__main__":
    try:
        benchmark_matrix_sizes()
        benchmark_vs_documented()
        
        print("\n" + "=" * 70)
        print("🎯 RTX 2060 BENCHMARK COMPLETE!")
        print("=" * 70)
        
    except Exception as e:
        print(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()