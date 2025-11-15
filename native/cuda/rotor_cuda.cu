/**
 * Rotor: 2-bit Ternary Neural Network Operations
 * CUDA implementation for GPU acceleration
 */

#include "../include/rotor.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// ============================================================================
// CUDA Kernels
// ============================================================================

__global__ void rotor_encode_kernel(
    const int8_t* values,
    uint8_t* bit0,
    uint8_t* bit1,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        bit0[idx] = (values[idx] == 1) ? 1 : 0;
        bit1[idx] = (values[idx] == -1) ? 1 : 0;
    }
}

__global__ void rotor_decode_kernel(
    const uint8_t* bit0,
    const uint8_t* bit1,
    int8_t* values,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        values[idx] = (int8_t)(bit0[idx]) - (int8_t)(bit1[idx]);
    }
}

__global__ void rotor_quantize_kernel(
    const float* values,
    int8_t* output,
    float threshold,
    size_t n
) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = values[idx];
        if (val > threshold) {
            output[idx] = 1;
        } else if (val < -threshold) {
            output[idx] = -1;
        } else {
            output[idx] = 0;
        }
    }
}

/**
 * Dot product kernel using warp-level primitives
 * Each warp computes a portion of the dot product
 */
__global__ void rotor_dot_kernel(
    const uint8_t* a_bit0,
    const uint8_t* a_bit1,
    const uint8_t* b_bit0,
    const uint8_t* b_bit1,
    int32_t* partial_sums,
    size_t n
) {
    __shared__ int32_t shared_pp[256];
    __shared__ int32_t shared_pn[256];
    __shared__ int32_t shared_np[256];
    __shared__ int32_t shared_nn[256];

    size_t tid = threadIdx.x;
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    int32_t pp = 0, pn = 0, np = 0, nn = 0;

    // Each thread processes multiple elements
    for (size_t i = idx; i < n; i += blockDim.x * gridDim.x) {
        uint8_t a0 = a_bit0[i];
        uint8_t a1 = a_bit1[i];
        uint8_t b0 = b_bit0[i];
        uint8_t b1 = b_bit1[i];

        pp += (a0 & b0);
        pn += (a0 & b1);
        np += (a1 & b0);
        nn += (a1 & b1);
    }

    // Store to shared memory
    shared_pp[tid] = pp;
    shared_pn[tid] = pn;
    shared_np[tid] = np;
    shared_nn[tid] = nn;

    __syncthreads();

    // Reduction in shared memory
    for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_pp[tid] += shared_pp[tid + s];
            shared_pn[tid] += shared_pn[tid + s];
            shared_np[tid] += shared_np[tid + s];
            shared_nn[tid] += shared_nn[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block
    if (tid == 0) {
        int32_t result = shared_pp[0] - shared_pn[0] - shared_np[0] + shared_nn[0];
        partial_sums[blockIdx.x] = result;
    }
}

/**
 * Matrix-vector multiply kernel: y = W @ x
 * Each block computes one output element
 */
__global__ void rotor_matvec_kernel(
    const uint8_t* W_bit0,
    const uint8_t* W_bit1,
    const uint8_t* x_bit0,
    const uint8_t* x_bit1,
    int32_t* y,
    size_t m,
    size_t n
) {
    size_t row = blockIdx.x;

    if (row < m) {
        __shared__ int32_t shared_pp[256];
        __shared__ int32_t shared_pn[256];
        __shared__ int32_t shared_np[256];
        __shared__ int32_t shared_nn[256];

        size_t tid = threadIdx.x;
        const uint8_t* w0 = W_bit0 + row * n;
        const uint8_t* w1 = W_bit1 + row * n;

        int32_t pp = 0, pn = 0, np = 0, nn = 0;

        // Each thread processes multiple elements
        for (size_t i = tid; i < n; i += blockDim.x) {
            uint8_t w_bit0 = w0[i];
            uint8_t w_bit1 = w1[i];
            uint8_t x_b0 = x_bit0[i];
            uint8_t x_b1 = x_bit1[i];

            pp += (w_bit0 & x_b0);
            pn += (w_bit0 & x_b1);
            np += (w_bit1 & x_b0);
            nn += (w_bit1 & x_b1);
        }

        // Store to shared memory
        shared_pp[tid] = pp;
        shared_pn[tid] = pn;
        shared_np[tid] = np;
        shared_nn[tid] = nn;

        __syncthreads();

        // Reduction
        for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                shared_pp[tid] += shared_pp[tid + s];
                shared_pn[tid] += shared_pn[tid + s];
                shared_np[tid] += shared_np[tid + s];
                shared_nn[tid] += shared_nn[tid + s];
            }
            __syncthreads();
        }

        // Write result
        if (tid == 0) {
            y[row] = shared_pp[0] - shared_pn[0] - shared_np[0] + shared_nn[0];
        }
    }
}

// ============================================================================
// Host Functions (C API)
// ============================================================================

extern "C" {

void rotor_encode_cuda(
    const int8_t* d_values,
    uint8_t* d_bit0,
    uint8_t* d_bit1,
    size_t n
) {
    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;
    rotor_encode_kernel<<<num_blocks, block_size>>>(d_values, d_bit0, d_bit1, n);
}

void rotor_decode_cuda(
    const uint8_t* d_bit0,
    const uint8_t* d_bit1,
    int8_t* d_values,
    size_t n
) {
    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;
    rotor_decode_kernel<<<num_blocks, block_size>>>(d_bit0, d_bit1, d_values, n);
}

int32_t rotor_dot_cuda(
    const uint8_t* d_a_bit0,
    const uint8_t* d_a_bit1,
    const uint8_t* d_b_bit0,
    const uint8_t* d_b_bit1,
    size_t n
) {
    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;

    // Allocate temp storage for partial sums
    int32_t* d_partial_sums;
    cudaMalloc(&d_partial_sums, num_blocks * sizeof(int32_t));

    // Launch kernel
    rotor_dot_kernel<<<num_blocks, block_size>>>(
        d_a_bit0, d_a_bit1, d_b_bit0, d_b_bit1,
        d_partial_sums, n
    );

    // Copy partial sums to host and reduce
    int32_t* h_partial_sums = (int32_t*)malloc(num_blocks * sizeof(int32_t));
    cudaMemcpy(h_partial_sums, d_partial_sums, num_blocks * sizeof(int32_t), cudaMemcpyDeviceToHost);

    int32_t result = 0;
    for (int i = 0; i < num_blocks; i++) {
        result += h_partial_sums[i];
    }

    free(h_partial_sums);
    cudaFree(d_partial_sums);

    return result;
}

void rotor_matvec_cuda(
    const uint8_t* d_W_bit0,
    const uint8_t* d_W_bit1,
    const float* d_x,
    size_t m,
    size_t n,
    int32_t* d_y
) {
    // Quantize input
    int8_t* d_x_quant;
    uint8_t* d_x_bit0;
    uint8_t* d_x_bit1;

    cudaMalloc(&d_x_quant, n * sizeof(int8_t));
    cudaMalloc(&d_x_bit0, n * sizeof(uint8_t));
    cudaMalloc(&d_x_bit1, n * sizeof(uint8_t));

    int block_size = 256;
    int num_blocks = (n + block_size - 1) / block_size;

    rotor_quantize_kernel<<<num_blocks, block_size>>>(d_x, d_x_quant, 0.0f, n);
    rotor_encode_kernel<<<num_blocks, block_size>>>(d_x_quant, d_x_bit0, d_x_bit1, n);

    // Launch matvec kernel (one block per output row)
    rotor_matvec_kernel<<<m, block_size>>>(
        d_W_bit0, d_W_bit1, d_x_bit0, d_x_bit1, d_y, m, n
    );

    cudaFree(d_x_quant);
    cudaFree(d_x_bit0);
    cudaFree(d_x_bit1);
}

} // extern "C"
