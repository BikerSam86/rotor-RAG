/**
 * Rotor: 2-bit Ternary Neural Network Operations
 * C implementation with SIMD optimization
 */

#include "../include/rotor.h"
#include <string.h>

// Detect SIMD capabilities
#if defined(__AVX2__)
    #include <immintrin.h>
    #define USE_AVX2 1
#elif defined(__SSE4_1__)
    #include <smmintrin.h>
    #define USE_SSE4 1
#elif defined(__ARM_NEON)
    #include <arm_neon.h>
    #define USE_NEON 1
#endif

// ============================================================================
// Basic Operations
// ============================================================================

void rotor_encode(const int8_t* values, size_t n, uint8_t* bit0, uint8_t* bit1) {
    for (size_t i = 0; i < n; i++) {
        bit0[i] = (values[i] == 1) ? 1 : 0;
        bit1[i] = (values[i] == -1) ? 1 : 0;
    }
}

void rotor_decode(const uint8_t* bit0, const uint8_t* bit1, size_t n, int8_t* values) {
    for (size_t i = 0; i < n; i++) {
        values[i] = (int8_t)(bit0[i]) - (int8_t)(bit1[i]);
    }
}

void rotor_pack(const uint8_t* bit0, const uint8_t* bit1, size_t n, uint8_t* packed) {
    size_t n_bytes = n / 4;

    for (size_t i = 0; i < n_bytes; i++) {
        size_t idx = i * 4;
        packed[i] =
            (bit0[idx + 3] << 7) | (bit1[idx + 3] << 6) |
            (bit0[idx + 2] << 5) | (bit1[idx + 2] << 4) |
            (bit0[idx + 1] << 3) | (bit1[idx + 1] << 2) |
            (bit0[idx + 0] << 1) | (bit1[idx + 0] << 0);
    }
}

void rotor_unpack(const uint8_t* packed, size_t n_rotors, uint8_t* bit0, uint8_t* bit1) {
    size_t n_bytes = (n_rotors + 3) / 4;

    for (size_t i = 0; i < n_bytes && i * 4 < n_rotors; i++) {
        uint8_t byte = packed[i];
        size_t idx = i * 4;

        if (idx + 0 < n_rotors) {
            bit0[idx + 0] = (byte >> 1) & 1;
            bit1[idx + 0] = (byte >> 0) & 1;
        }
        if (idx + 1 < n_rotors) {
            bit0[idx + 1] = (byte >> 3) & 1;
            bit1[idx + 1] = (byte >> 2) & 1;
        }
        if (idx + 2 < n_rotors) {
            bit0[idx + 2] = (byte >> 5) & 1;
            bit1[idx + 2] = (byte >> 4) & 1;
        }
        if (idx + 3 < n_rotors) {
            bit0[idx + 3] = (byte >> 7) & 1;
            bit1[idx + 3] = (byte >> 6) & 1;
        }
    }
}

// ============================================================================
// Dot Product - Scalar (fallback)
// ============================================================================

static int32_t rotor_dot_scalar(
    const uint8_t* a_bit0, const uint8_t* a_bit1,
    const uint8_t* b_bit0, const uint8_t* b_bit1,
    size_t n
) {
    int32_t pp = 0, pn = 0, np = 0, nn = 0;

    for (size_t i = 0; i < n; i++) {
        pp += (a_bit0[i] & b_bit0[i]);
        pn += (a_bit0[i] & b_bit1[i]);
        np += (a_bit1[i] & b_bit0[i]);
        nn += (a_bit1[i] & b_bit1[i]);
    }

    return pp - pn - np + nn;
}

// ============================================================================
// Dot Product - AVX2 (x86)
// ============================================================================

#ifdef USE_AVX2
int32_t rotor_dot_avx2(
    const uint8_t* a_bit0, const uint8_t* a_bit1,
    const uint8_t* b_bit0, const uint8_t* b_bit1,
    size_t n
) {
    __m256i pp_acc = _mm256_setzero_si256();
    __m256i pn_acc = _mm256_setzero_si256();
    __m256i np_acc = _mm256_setzero_si256();
    __m256i nn_acc = _mm256_setzero_si256();

    size_t i = 0;
    const size_t simd_width = 32;  // Process 32 bytes at a time

    // SIMD loop
    for (; i + simd_width <= n; i += simd_width) {
        __m256i a0 = _mm256_loadu_si256((const __m256i*)(a_bit0 + i));
        __m256i a1 = _mm256_loadu_si256((const __m256i*)(a_bit1 + i));
        __m256i b0 = _mm256_loadu_si256((const __m256i*)(b_bit0 + i));
        __m256i b1 = _mm256_loadu_si256((const __m256i*)(b_bit1 + i));

        // Compute AND operations
        __m256i pp = _mm256_and_si256(a0, b0);
        __m256i pn = _mm256_and_si256(a0, b1);
        __m256i np = _mm256_and_si256(a1, b0);
        __m256i nn = _mm256_and_si256(a1, b1);

        // Accumulate popcounts
        pp_acc = _mm256_add_epi8(pp_acc, pp);
        pn_acc = _mm256_add_epi8(pn_acc, pn);
        np_acc = _mm256_add_epi8(np_acc, np);
        nn_acc = _mm256_add_epi8(nn_acc, nn);
    }

    // Horizontal sum
    int32_t pp_sum = 0, pn_sum = 0, np_sum = 0, nn_sum = 0;

    uint8_t pp_arr[32], pn_arr[32], np_arr[32], nn_arr[32];
    _mm256_storeu_si256((__m256i*)pp_arr, pp_acc);
    _mm256_storeu_si256((__m256i*)pn_arr, pn_acc);
    _mm256_storeu_si256((__m256i*)np_arr, np_acc);
    _mm256_storeu_si256((__m256i*)nn_arr, nn_acc);

    for (int j = 0; j < 32; j++) {
        pp_sum += pp_arr[j];
        pn_sum += pn_arr[j];
        np_sum += np_arr[j];
        nn_sum += nn_arr[j];
    }

    // Scalar remainder
    for (; i < n; i++) {
        pp_sum += (a_bit0[i] & b_bit0[i]);
        pn_sum += (a_bit0[i] & b_bit1[i]);
        np_sum += (a_bit1[i] & b_bit0[i]);
        nn_sum += (a_bit1[i] & b_bit1[i]);
    }

    return pp_sum - pn_sum - np_sum + nn_sum;
}
#endif

// ============================================================================
// Dot Product - Dispatcher
// ============================================================================

int32_t rotor_dot(
    const uint8_t* a_bit0, const uint8_t* a_bit1,
    const uint8_t* b_bit0, const uint8_t* b_bit1,
    size_t n
) {
#ifdef USE_AVX2
    return rotor_dot_avx2(a_bit0, a_bit1, b_bit0, b_bit1, n);
#elif defined(USE_NEON)
    return rotor_dot_neon(a_bit0, a_bit1, b_bit0, b_bit1, n);
#else
    return rotor_dot_scalar(a_bit0, a_bit1, b_bit0, b_bit1, n);
#endif
}

// ============================================================================
// Quantization
// ============================================================================

void rotor_quantize_ternary(const float* values, size_t n, float threshold, int8_t* output) {
    for (size_t i = 0; i < n; i++) {
        if (values[i] > threshold) {
            output[i] = 1;
        } else if (values[i] < -threshold) {
            output[i] = -1;
        } else {
            output[i] = 0;
        }
    }
}

// ============================================================================
// BitNet Format Conversion
// ============================================================================

void bitnet_to_rotor(
    const uint8_t* bitnet_packed,
    size_t rows,
    size_t cols,
    uint8_t* bit0,
    uint8_t* bit1
) {
    /**
     * Convert BitNet packed format to Rotor format.
     *
     * BitNet: 2 bits per weight, 4 weights per byte
     *   00 = 0, 10 = +1, 01 = -1, 11 = error
     *
     * Rotor: Separate bit arrays, 8 weights per byte each
     *   bit0=1 means +1, bit1=1 means -1
     *
     * This vectorizes beautifully with SIMD!
     */

    size_t bitnet_cols_bytes = (cols + 3) / 4;  // 4 weights per byte
    size_t rotor_cols_bytes = (cols + 7) / 8;   // 8 weights per byte

    // Process each row
    for (size_t row = 0; row < rows; row++) {
        const uint8_t* src = bitnet_packed + row * bitnet_cols_bytes;
        uint8_t* dst0 = bit0 + row * rotor_cols_bytes;
        uint8_t* dst1 = bit1 + row * rotor_cols_bytes;

        // Zero output first
        memset(dst0, 0, rotor_cols_bytes);
        memset(dst1, 0, rotor_cols_bytes);

        // Process 4 weights at a time (1 BitNet byte)
        size_t weight_idx = 0;
        for (size_t i = 0; i < bitnet_cols_bytes && weight_idx < cols; i++) {
            uint8_t bitnet_byte = src[i];

            // Extract 4 weights from this byte
            for (int j = 0; j < 4 && weight_idx < cols; j++, weight_idx++) {
                // Extract 2 bits for this weight
                uint8_t two_bits = (bitnet_byte >> (j * 2)) & 0b11;

                // Decode: 00=0, 10=+1, 01=-1, 11=error
                size_t byte_idx = weight_idx / 8;
                size_t bit_pos = weight_idx % 8;

                if (two_bits == 0b10) {
                    // +1: set bit0
                    dst0[byte_idx] |= (1 << bit_pos);
                } else if (two_bits == 0b01) {
                    // -1: set bit1
                    dst1[byte_idx] |= (1 << bit_pos);
                }
                // 00 = 0: both bits stay 0 (already cleared)
                // 11 = error: treat as 0
            }
        }
    }
}

/**
 * Unpack Rotor format to int8 weights.
 *
 * Rotor format: bit0/bit1 arrays (8 weights per byte)
 * Output: int8 array where weight = bit0 - bit1
 *
 * This is MUCH faster than Python bit-by-bit loops!
 */
void rotor_unpack_weights(
    const uint8_t* bit0,
    const uint8_t* bit1,
    size_t rows,
    size_t cols,
    int8_t* weights
) {
    size_t rotor_cols_bytes = (cols + 7) / 8;  // 8 weights per byte

    // Process each row
    for (size_t row = 0; row < rows; row++) {
        const uint8_t* src0 = bit0 + row * rotor_cols_bytes;
        const uint8_t* src1 = bit1 + row * rotor_cols_bytes;
        int8_t* dst = weights + row * cols;

        // Process each byte (8 weights at a time)
        size_t weight_idx = 0;
        for (size_t byte_idx = 0; byte_idx < rotor_cols_bytes && weight_idx < cols; byte_idx++) {
            uint8_t b0 = src0[byte_idx];
            uint8_t b1 = src1[byte_idx];

            // Unpack 8 weights from this byte
            for (int bit_pos = 0; bit_pos < 8 && weight_idx < cols; bit_pos++, weight_idx++) {
                int8_t v0 = (b0 >> bit_pos) & 1;
                int8_t v1 = (b1 >> bit_pos) & 1;
                dst[weight_idx] = v0 - v1;  // +1, 0, or -1
            }
        }
    }
}

// ============================================================================
// Matrix Operations
// ============================================================================

void rotor_matvec(
    const uint8_t* W_bit0, const uint8_t* W_bit1,
    const float* x,
    size_t m, size_t n,
    int32_t* y
) {
    // Quantize input
    int8_t* x_quant = (int8_t*)malloc(n * sizeof(int8_t));
    uint8_t* x_bit0 = (uint8_t*)malloc(n * sizeof(uint8_t));
    uint8_t* x_bit1 = (uint8_t*)malloc(n * sizeof(uint8_t));

    rotor_quantize_ternary(x, n, 0.0f, x_quant);
    rotor_encode(x_quant, n, x_bit0, x_bit1);

    // Compute each output element
    for (size_t i = 0; i < m; i++) {
        y[i] = rotor_dot(
            W_bit0 + i * n, W_bit1 + i * n,
            x_bit0, x_bit1,
            n
        );
    }

    free(x_quant);
    free(x_bit0);
    free(x_bit1);
}

void rotor_batch_matmul(
    const uint8_t* W_bit0, const uint8_t* W_bit1,
    const float* X,
    size_t batch_size, size_t m, size_t n,
    int32_t* Y
) {
    // Process each sample in the batch
    for (size_t b = 0; b < batch_size; b++) {
        const float* x = X + b * n;
        int32_t* y = Y + b * m;
        rotor_matvec(W_bit0, W_bit1, x, m, n, y);
    }
}
