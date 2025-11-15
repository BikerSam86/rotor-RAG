# Rotor Optimization Guide

## Performance Optimization with C/CUDA

This document explains how to build and use the optimized native implementations.

---

## Architecture

We provide three levels of optimization:

1. **NumPy (Pure Python)** - Baseline, always available
2. **C + SIMD (CPU)** - 5-10× faster, uses AVX2/AVX-512/NEON
3. **CUDA (GPU)** - 20-100× faster for large models

---

## Building Native Libraries

### Prerequisites

**Windows:**
- Visual Studio 2022 or Clang
- For CUDA: NVIDIA CUDA Toolkit 11.0+

**macOS:**
- Xcode Command Line Tools: `xcode-select --install`
- For CUDA: Not typically available on Mac

**Linux:**
- GCC: `sudo apt-get install build-essential`
- For CUDA: NVIDIA CUDA Toolkit 11.0+

### Build Commands

```bash
cd rotor-rag-code

# Build C library (CPU, SIMD-optimized)
python native/build.py

# Build with CUDA support (GPU)
python native/build.py --with-cuda
```

###Output

Files will be created in `native/build/`:
- Windows: `librotor.dll` (and optionally `librotor_cuda.dll`)
- macOS: `librotor.dylib` (and optionally `librotor_cuda.dylib`)
- Linux: `librotor.so` (and optionally `librotor_cuda.so`)

---

## Usage

### Automatic Backend Selection

```python
from rotor.layers import SimpleRotorNet

# Automatically uses fastest available backend
net = SimpleRotorNet(128, 64, 10)
x = np.random.randn(32, 128)
output = net.forward(x)  # Will use C/CUDA if available
```

### Manual Backend Selection

```python
from rotor.native import get_backend

# Get native backend
backend = get_backend(use_cuda=False)  # CPU
# backend = get_backend(use_cuda=True)   # GPU (if available)

# Use directly
a_bit0, a_bit1 = backend.encode(values)
result = backend.dot(a_bit0, a_bit1, b_bit0, b_bit1)
```

### Checking Availability

```python
from rotor.native import get_backend

backend = get_backend()

if backend.available:
    print("Native backend ready!")
    print(f"CUDA: {backend.use_cuda}")
else:
    print("Using NumPy fallback")
```

---

## Performance Benchmarks

Run benchmarks to see speedups:

```bash
# Run with NumPy only
python tests/benchmark.py

# Build native, then benchmark again
python native/build.py
python tests/benchmark.py
```

### Baseline Results (NumPy)

From our testing on Windows (Python 3.13, NumPy 2.3.4):

| Operation | Size | Time (ms) |
|-----------|------|-----------|
| Encode/Decode | 1M values | 8.59 |
| Dot Product | 1M elements | 12.79 |
| MatVec | 2048×2048 | 180.36 |
| Network Forward | 512→256→10, batch=32 | 731.77 |

### Expected Speedups

With native implementations:

| Implementation | Typical Speedup |
|----------------|-----------------|
| C + AVX2 | 5-10× |
| C + AVX-512 | 10-15× |
| CUDA | 20-100× (depends on size) |

---

## SIMD Implementation Details

### x86/x64 (Intel/AMD)

Uses **AVX2** intrinsics for parallel bitwise operations:
- Processes 32 bytes (256 rotors) per instruction
- Automatic detection via `__AVX2__` flag
- Falls back to scalar if unavailable

**Compile flags:**
```bash
-mavx2    # Enable AVX2
-O3       # Maximum optimization
```

### ARM (Apple Silicon, Mobile)

Uses **NEON** intrinsics:
- Processes 16 bytes (128 rotors) per instruction
- Automatic detection via `__ARM_NEON` flag

**Compile flags:**
```bash
-march=native   # Use best available SIMD
-O3             # Maximum optimization
```

### Key Operations

**Dot Product (SIMD)**:
```c
// Process 32 bytes at a time
__m256i pp = _mm256_and_si256(a0, b0);  // Parallel AND
pp_acc = _mm256_add_epi8(pp_acc, pp);   // Accumulate
```

**Horizontal Sum**:
After SIMD accumulation, reduce to scalar via horizontal sum.

---

## CUDA Implementation

### Kernel Architecture

**Dot Product**:
- Each warp (32 threads) processes elements in parallel
- Shared memory for block-level reduction
- Final reduction on CPU

**Matrix-Vector**:
- One block per output row
- Threads within block compute partial sums
- Efficient for large matrices

### Memory Management

CUDA kernels handle device memory automatically:
```python
# Input on host (CPU)
x = np.array([...])

# Kernel handles device copy and computation
y = backend.matvec_cuda(W_bit0, W_bit1, x)

# Output returned to host
```

### Launch Configuration

Optimized for modern GPUs:
- Block size: 256 threads (good for most GPUs)
- Grid size: Calculated based on problem size

---

## Memory Comparison

For a 100M parameter model:

| Format | Memory | Ratio |
|--------|--------|-------|
| Ternary (2-bit) | 23.8 MB | 1.0× |
| INT8 | 95.4 MB | 4.0× |
| FP16 | 190.7 MB | 8.0× |
| FP32 | 381.5 MB | 16.0× |

**Key insight**: 2-bit ternary is **8× smaller than FP16** while maintaining good performance!

---

## Troubleshooting

### "Native library not found"

**Solution**: Build the library first:
```bash
python native/build.py
```

### "Compiler not found"

**Windows**: Install Visual Studio 2022 or Clang
**macOS**: `xcode-select --install`
**Linux**: `sudo apt-get install build-essential`

### "CUDA compiler not found"

Install CUDA Toolkit from: https://developer.nvidia.com/cuda-downloads

### Build succeeds but library doesn't load

Check that the library is in `native/build/`:
```bash
ls native/build/
```

On Linux, may need to set `LD_LIBRARY_PATH`:
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/native/build
```

---

## Profiling Tips

### Measure Your Workload

```python
import time

def benchmark(func, *args, n=100):
    start = time.perf_counter()
    for _ in range(n):
        func(*args)
    return (time.perf_counter() - start) / n * 1000  # ms

# Test
t = benchmark(net.forward, x, n=100)
print(f"Forward pass: {t:.4f} ms")
```

### CPU Profiling

```bash
# Linux
perf record -g python examples/demo_network.py
perf report

# Python profiler
python -m cProfile -s cumtime examples/demo_network.py
```

### GPU Profiling

```bash
# NVIDIA
nvprof python examples/demo_network.py

# Or use NSight Systems
nsys profile python examples/demo_network.py
```

---

## Next Optimization Steps

1. **Pack weights tighter**: Currently 4 rotors/byte, could optimize to 8
2. **Fuse operations**: Combine quantization + encode + matmul
3. **Custom CUDA kernels**: Hand-optimized for specific layer sizes
4. **Multi-GPU**: Distribute large models across GPUs
5. **INT4/INT8 activations**: Mixed precision for even more speed

---

## Contributing

Have optimization ideas? Open an issue or PR!

Areas for improvement:
- ARM NEON implementation
- AMD GPU support (ROCm)
- Apple Metal shaders
- WebAssembly for browser deployment

---

🚀 **All ways, always!**
