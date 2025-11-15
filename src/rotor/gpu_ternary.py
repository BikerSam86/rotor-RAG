"""
GPU-accelerated ternary matrix operations using OpenCL.
Target: Intel HD Graphics 615 (24 EUs, 168 threads)
"""

import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array

# OpenCL kernel for ternary matrix multiplication
TERNARY_MATMUL_KERNEL = """
__kernel void ternary_matmul(
    __global const uchar* packed_weights,  // Packed 2-bit ternary weights
    __global const float* input,            // Input vector [in_dim]
    __global const float* scales,           // Scale factors per output
    __global float* output,                 // Output vector [out_dim]
    const uint in_dim,                      // Input dimension
    const uint out_dim                      // Output dimension
) {
    uint tid = get_global_id(0);
    if (tid >= out_dim) return;

    float sum = 0.0f;

    // Each thread computes one output element
    for (uint i = 0; i < in_dim; i++) {
        // Calculate position in packed array
        // 4 weights per byte (2 bits each)
        uint weight_idx = tid * in_dim + i;
        uint byte_idx = weight_idx / 4;
        uint bit_offset = (weight_idx % 4) * 2;

        // Extract 2-bit ternary value
        uchar packed = packed_weights[byte_idx];
        uchar ternary = (packed >> bit_offset) & 0x3;

        // Decode ternary: 0=>-1, 1=>0, 2=>+1
        float weight = (float)((int)ternary - 1);

        sum += input[i] * weight;
    }

    // Apply scale factor
    output[tid] = sum * scales[tid];
}
"""

# Kernel for RMSNorm (used in transformer)
RMS_NORM_KERNEL = """
__kernel void rms_norm(
    __global const float* input,
    __global const float* weight,
    __global float* output,
    const uint dim,
    const float eps
) {
    uint tid = get_global_id(0);
    if (tid >= dim) return;

    // Compute RMS (each work group handles one token)
    float sum_sq = 0.0f;
    for (uint i = 0; i < dim; i++) {
        float val = input[i];
        sum_sq += val * val;
    }

    float rms = sqrt(sum_sq / (float)dim + eps);
    output[tid] = (input[tid] / rms) * weight[tid];
}
"""


class GPUTernaryOps:
    """
    GPU-accelerated ternary operations using OpenCL.
    """

    def __init__(self, device_type='GPU'):
        """Initialize OpenCL context and compile kernels."""
        # Get OpenCL platform and device
        platforms = cl.get_platforms()

        # Find Intel GPU
        self.device = None
        for platform in platforms:
            devices = platform.get_devices()
            for dev in devices:
                if 'Intel' in dev.name and 'Graphics' in dev.name:
                    self.device = dev
                    break
            if self.device:
                break

        if not self.device:
            raise RuntimeError("Intel GPU not found!")

        print(f"[OK] Using GPU: {self.device.name}")
        print(f"  Compute Units: {self.device.max_compute_units}")
        print(f"  Max Work Group Size: {self.device.max_work_group_size}")
        print(f"  Global Memory: {self.device.global_mem_size // (1024**2)} MB")

        # Create context and queue
        self.ctx = cl.Context([self.device])
        self.queue = cl.CommandQueue(self.ctx)

        # Compile kernels
        program = cl.Program(self.ctx, TERNARY_MATMUL_KERNEL).build()
        self.matmul_kernel = program.ternary_matmul

        print("[OK] OpenCL kernels compiled")

    def pack_ternary_weights(self, weights: np.ndarray) -> np.ndarray:
        """
        Pack ternary weights {-1, 0, +1} into 2-bit format.
        4 weights per byte.

        Args:
            weights: Float array with values in {-1, 0, +1}

        Returns:
            Packed uint8 array
        """
        # Convert to encoding: -1=>0, 0=>1, +1=>2
        encoded = (weights + 1).astype(np.uint8)

        # Pack 4 values per byte
        flat = encoded.flatten()
        num_packed = (len(flat) + 3) // 4  # Ceiling division
        packed = np.zeros(num_packed, dtype=np.uint8)

        for i in range(len(flat)):
            byte_idx = i // 4
            bit_offset = (i % 4) * 2
            packed[byte_idx] |= (flat[i] << bit_offset)

        return packed

    def ternary_matmul(
        self,
        packed_weights: np.ndarray,
        scales: np.ndarray,
        input_vec: np.ndarray,
        in_dim: int,
        out_dim: int
    ) -> np.ndarray:
        """
        GPU-accelerated ternary matrix multiplication.

        Args:
            packed_weights: Packed 2-bit weights [out_dim * in_dim / 4]
            scales: Scale factors [out_dim]
            input_vec: Input vector [in_dim]
            in_dim: Input dimension
            out_dim: Output dimension

        Returns:
            Output vector [out_dim]
        """
        # Create GPU buffers
        weights_buf = cl.Buffer(
            self.ctx,
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=packed_weights
        )

        input_buf = cl.Buffer(
            self.ctx,
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=input_vec.astype(np.float32)
        )

        scales_buf = cl.Buffer(
            self.ctx,
            cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
            hostbuf=scales.astype(np.float32)
        )

        output_buf = cl.Buffer(
            self.ctx,
            cl.mem_flags.WRITE_ONLY,
            size=out_dim * 4  # 4 bytes per float
        )

        # Launch kernel
        global_size = ((out_dim + 63) // 64) * 64  # Round up to multiple of 64
        self.matmul_kernel(
            self.queue,
            (global_size,),
            None,
            weights_buf,
            input_buf,
            scales_buf,
            output_buf,
            np.uint32(in_dim),
            np.uint32(out_dim)
        )

        # Read result
        output = np.empty(out_dim, dtype=np.float32)
        cl.enqueue_copy(self.queue, output, output_buf).wait()

        return output

    def __del__(self):
        """Cleanup OpenCL resources."""
        if hasattr(self, 'queue'):
            self.queue.finish()


# Quick test function
def test_gpu_ternary():
    """Test GPU ternary operations."""
    print("=" * 70)
    print("GPU TERNARY OPERATIONS TEST")
    print("=" * 70)

    # Initialize GPU
    gpu = GPUTernaryOps()

    # Test case: small matmul
    in_dim = 2560
    out_dim = 2560

    print(f"\nTesting {out_dim}×{in_dim} ternary matmul...")

    # Create random ternary weights
    weights = np.random.choice([-1, 0, 1], size=(out_dim, in_dim))
    scales = np.random.randn(out_dim).astype(np.float32)
    input_vec = np.random.randn(in_dim).astype(np.float32)

    # Pack weights
    packed = gpu.pack_ternary_weights(weights)
    print(f"Packed {weights.size} ternary weights into {packed.size} bytes")
    print(f"Compression: {weights.size / packed.size:.1f}× (2 bits per weight)")

    # GPU matmul
    import time
    start = time.perf_counter()
    gpu_result = gpu.ternary_matmul(packed, scales, input_vec, in_dim, out_dim)
    gpu_time = time.perf_counter() - start

    # CPU matmul (for comparison)
    start = time.perf_counter()
    cpu_result = (weights @ input_vec) * scales
    cpu_time = time.perf_counter() - start

    # Verify correctness
    max_diff = np.abs(gpu_result - cpu_result).max()

    print(f"\n{'Results:'}")
    print(f"  GPU time: {gpu_time*1000:.2f}ms")
    print(f"  CPU time: {cpu_time*1000:.2f}ms")
    print(f"  Speedup: {cpu_time/gpu_time:.2f}x")
    print(f"  Max difference: {max_diff:.6f}")

    if max_diff < 1e-4:
        print(f"  [OK] Results match!")
    else:
        print(f"  [ERROR] Results don't match (diff too large)")

    print("\n" + "=" * 70)
    print("All ways, always!")
    print("=" * 70)


if __name__ == "__main__":
    test_gpu_ternary()
