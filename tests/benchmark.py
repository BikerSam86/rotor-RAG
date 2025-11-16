"""
Benchmark suite for rotor operations.
Compares NumPy vs C/CUDA implementations.
"""

import numpy as np
import time
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rotor.core import RotorCore
from rotor.layers import TernaryLinear, SimpleRotorNet

try:
    from rotor.native import get_backend
    native_backend = get_backend()
    HAS_NATIVE = native_backend.available
except:
    HAS_NATIVE = False


def timer(func, *args, n_iterations=100, warmup=10):
    """Time a function over multiple iterations."""
    # Warmup
    for _ in range(warmup):
        func(*args)

    # Timed runs
    start = time.perf_counter()
    for _ in range(n_iterations):
        func(*args)
    end = time.perf_counter()

    return (end - start) / n_iterations * 1000  # ms


def benchmark_encode_decode():
    """Benchmark encode/decode operations."""
    print("\n" + "="*70)
    print("Benchmark: Encode/Decode")
    print("="*70)

    sizes = [1000, 10000, 100000, 1000000]

    print(f"\n{'Size':<15} {'NumPy (ms)':<15} {'Native (ms)':<15} {'Speedup':<15}")
    print("-" * 70)

    for n in sizes:
        values = np.random.choice([-1, 0, 1], size=n).astype(np.int8)

        # NumPy
        def numpy_impl():
            bit0, bit1 = RotorCore.encode(values)
            decoded = RotorCore.decode(bit0, bit1)

        t_numpy = timer(numpy_impl, n_iterations=100)

        # Native (if available)
        if HAS_NATIVE:
            def native_impl():
                bit0, bit1 = native_backend.encode(values)
                decoded = native_backend.decode(bit0, bit1)

            t_native = timer(native_impl, n_iterations=100)
            speedup = t_numpy / t_native
            print(f"{n:<15,} {t_numpy:<15.4f} {t_native:<15.4f} {speedup:<15.2f}x")
        else:
            print(f"{n:<15,} {t_numpy:<15.4f} {'N/A':<15} {'N/A':<15}")


def benchmark_dot_product():
    """Benchmark dot product computation."""
    print("\n" + "="*70)
    print("Benchmark: Dot Product")
    print("="*70)

    sizes = [1000, 10000, 100000, 1000000]

    print(f"\n{'Size':<15} {'NumPy (ms)':<15} {'Native (ms)':<15} {'Speedup':<15}")
    print("-" * 70)

    for n in sizes:
        a = np.random.choice([-1, 0, 1], size=n).astype(np.int8)
        b = np.random.choice([-1, 0, 1], size=n).astype(np.int8)

        a_bit0, a_bit1 = RotorCore.encode(a)
        b_bit0, b_bit1 = RotorCore.encode(b)

        # NumPy
        def numpy_impl():
            return RotorCore.dot(a_bit0, a_bit1, b_bit0, b_bit1)

        t_numpy = timer(numpy_impl, n_iterations=100)

        # Native (if available)
        if HAS_NATIVE:
            def native_impl():
                return native_backend.dot(a_bit0, a_bit1, b_bit0, b_bit1)

            t_native = timer(native_impl, n_iterations=100)
            speedup = t_numpy / t_native
            print(f"{n:<15,} {t_numpy:<15.4f} {t_native:<15.4f} {speedup:<15.2f}x")
        else:
            print(f"{n:<15,} {t_numpy:<15.4f} {'N/A':<15} {'N/A':<15}")


def benchmark_matvec():
    """Benchmark matrix-vector multiplication."""
    print("\n" + "="*70)
    print("Benchmark: Matrix-Vector Multiply")
    print("="*70)

    configs = [
        (128, 256),
        (512, 512),
        (1024, 1024),
        (2048, 2048),
    ]

    print(f"\n{'Shape (m×n)':<20} {'NumPy (ms)':<15} {'Native (ms)':<15} {'Speedup':<15}")
    print("-" * 70)

    for m, n in configs:
        W = np.random.choice([-1, 0, 1], size=(m, n)).astype(np.int8)
        x = np.random.randn(n).astype(np.float32)

        W_bit0, W_bit1 = RotorCore.encode(W)

        # NumPy
        def numpy_impl():
            return RotorCore.matmul(W_bit0, W_bit1, x)

        t_numpy = timer(numpy_impl, n_iterations=50)

        # Native (if available)
        if HAS_NATIVE:
            def native_impl():
                return native_backend.matvec(W_bit0, W_bit1, x)

            t_native = timer(native_impl, n_iterations=50)
            speedup = t_numpy / t_native
            print(f"{m}×{n:<17} {t_numpy:<15.4f} {t_native:<15.4f} {speedup:<15.2f}x")
        else:
            print(f"{m}×{n:<17} {t_numpy:<15.4f} {'N/A':<15} {'N/A':<15}")


def benchmark_network():
    """Benchmark full network forward pass."""
    print("\n" + "="*70)
    print("Benchmark: Network Forward Pass")
    print("="*70)

    configs = [
        ("Small", 128, 64, 10, 1),
        ("Medium", 512, 256, 10, 1),
        ("Large", 2048, 1024, 100, 1),
        ("Batch-32", 512, 256, 10, 32),
    ]

    print(f"\n{'Config':<15} {'Batch':<10} {'NumPy (ms)':<15} {'Memory (KB)':<15}")
    print("-" * 70)

    for name, input_dim, hidden_dim, output_dim, batch_size in configs:
        net = SimpleRotorNet(input_dim, hidden_dim, output_dim)

        if batch_size == 1:
            x = np.random.randn(input_dim).astype(np.float32)
        else:
            x = np.random.randn(batch_size, input_dim).astype(np.float32)

        # Time forward pass
        def forward():
            return net.forward(x)

        t = timer(forward, n_iterations=20)

        # Calculate memory
        stats = net.get_stats()
        total_weights = stats['layer1_weights']['total'] + stats['layer2_weights']['total']
        memory_kb = (total_weights * 2 / 8) / 1024

        print(f"{name:<15} {batch_size:<10} {t:<15.4f} {memory_kb:<15.2f}")


def memory_comparison():
    """Compare memory usage of different representations."""
    print("\n" + "="*70)
    print("Memory Comparison")
    print("="*70)

    sizes = [1_000_000, 10_000_000, 100_000_000]

    print(f"\n{'Parameters':<20} {'Ternary':<15} {'INT8':<15} {'FP16':<15} {'FP32':<15}")
    print("-" * 70)

    for n in sizes:
        ternary_bytes = n * 2 / 8  # 2 bits per weight
        int8_bytes = n * 1
        fp16_bytes = n * 2
        fp32_bytes = n * 4

        def format_size(bytes_val):
            if bytes_val < 1024:
                return f"{bytes_val:.0f} B"
            elif bytes_val < 1024**2:
                return f"{bytes_val/1024:.1f} KB"
            elif bytes_val < 1024**3:
                return f"{bytes_val/1024**2:.1f} MB"
            else:
                return f"{bytes_val/1024**3:.2f} GB"

        print(f"{n:<20,} {format_size(ternary_bytes):<15} "
              f"{format_size(int8_bytes):<15} {format_size(fp16_bytes):<15} "
              f"{format_size(fp32_bytes):<15}")


def run_all_benchmarks():
    """Run all benchmarks."""
    print("\n" + "="*70)
    print("Rotor Performance Benchmarks")
    print("="*70)

    print(f"\nPlatform: {sys.platform}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"NumPy: {np.__version__}")
    print(f"Native Backend: {'Available ✓' if HAS_NATIVE else 'Not Available ✗'}")

    if not HAS_NATIVE:
        print("\n⚠️  Native library not built.")
        print("To build: python native/build.py")
        print("Falling back to NumPy implementation.\n")

    benchmark_encode_decode()
    benchmark_dot_product()
    benchmark_matvec()
    benchmark_network()
    memory_comparison()

    print("\n" + "="*70)
    print("Benchmarks Complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_all_benchmarks()
