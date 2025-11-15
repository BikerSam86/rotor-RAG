/**
 * Rotor: 2-bit Ternary Neural Network Operations
 * Header file for C/CUDA implementations
 */

#ifndef ROTOR_H
#define ROTOR_H

#include <stdint.h>
#include <stddef.h>

// DLL export/import declarations for Windows
#ifdef _WIN32
    #ifdef ROTOR_BUILD_DLL
        #define ROTOR_API __declspec(dllexport)
    #else
        #define ROTOR_API __declspec(dllimport)
    #endif
#else
    #define ROTOR_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// Core Operations
// ============================================================================

/**
 * Encode ternary values {-1, 0, +1} to 2-bit rotor format.
 *
 * @param values Input array of int8_t values (-1, 0, or 1)
 * @param n Number of values
 * @param bit0 Output: first bit of each rotor
 * @param bit1 Output: second bit of each rotor
 */
void rotor_encode(const int8_t* values, size_t n, uint8_t* bit0, uint8_t* bit1);

/**
 * Decode 2-bit rotor format back to ternary values.
 *
 * @param bit0 First bit array
 * @param bit1 Second bit array
 * @param n Number of values
 * @param values Output: decoded ternary values
 */
void rotor_decode(const uint8_t* bit0, const uint8_t* bit1, size_t n, int8_t* values);

/**
 * Pack bit pairs into compact uint8 representation (4 rotors per byte).
 *
 * @param bit0 First bit array (length must be multiple of 4)
 * @param bit1 Second bit array
 * @param n Number of rotors (must be multiple of 4)
 * @param packed Output: packed bytes (length = n/4)
 */
void rotor_pack(const uint8_t* bit0, const uint8_t* bit1, size_t n, uint8_t* packed);

/**
 * Unpack bytes back to bit pairs.
 *
 * @param packed Packed byte array
 * @param n_rotors Number of rotors to extract
 * @param bit0 Output: first bit array
 * @param bit1 Output: second bit array
 */
void rotor_unpack(const uint8_t* packed, size_t n_rotors, uint8_t* bit0, uint8_t* bit1);

/**
 * Compute dot product of two rotor arrays.
 * Uses SIMD for acceleration on x86 (AVX2/AVX-512) and ARM (NEON).
 *
 * @param a_bit0 First array bit0
 * @param a_bit1 First array bit1
 * @param b_bit0 Second array bit0
 * @param b_bit1 Second array bit1
 * @param n Number of elements
 * @return Dot product (int32_t)
 */
int32_t rotor_dot(
    const uint8_t* a_bit0, const uint8_t* a_bit1,
    const uint8_t* b_bit0, const uint8_t* b_bit1,
    size_t n
);

/**
 * Matrix-vector multiply: y = W @ x
 * W is in rotor format (ternary weights)
 * x will be quantized to ternary
 *
 * @param W_bit0 Weight matrix bit0 [m x n]
 * @param W_bit1 Weight matrix bit1 [m x n]
 * @param x Input vector (will be quantized)
 * @param m Number of output features
 * @param n Number of input features
 * @param y Output vector [m]
 */
void rotor_matvec(
    const uint8_t* W_bit0, const uint8_t* W_bit1,
    const float* x,
    size_t m, size_t n,
    int32_t* y
);

/**
 * Batched matrix multiply: Y = X @ W^T
 * X is batch of inputs [batch_size x n]
 * W is weight matrix in rotor format [m x n]
 *
 * @param W_bit0 Weight matrix bit0 [m x n]
 * @param W_bit1 Weight matrix bit1 [m x n]
 * @param X Input batch (will be quantized) [batch_size x n]
 * @param batch_size Number of samples in batch
 * @param m Number of output features
 * @param n Number of input features
 * @param Y Output batch [batch_size x m]
 */
void rotor_batch_matmul(
    const uint8_t* W_bit0, const uint8_t* W_bit1,
    const float* X,
    size_t batch_size, size_t m, size_t n,
    int32_t* Y
);

// ============================================================================
// Quantization
// ============================================================================

/**
 * Quantize float array to ternary {-1, 0, +1}.
 *
 * @param values Input float array
 * @param n Number of values
 * @param threshold Values in [-threshold, threshold] map to 0
 * @param output Output ternary array
 */
void rotor_quantize_ternary(const float* values, size_t n, float threshold, int8_t* output);

// ============================================================================
// BitNet Format Conversion
// ============================================================================

/**
 * Convert BitNet packed format to Rotor format.
 *
 * BitNet format: 2 bits per weight, 4 weights per byte
 *   00 = 0, 10 = +1, 01 = -1, 11 = error
 *
 * Rotor format: Separate bit0 and bit1 arrays (8 weights per byte each)
 *   weight = bit0 - bit1
 *
 * This is a CRITICAL performance function - uses SIMD when available!
 *
 * @param bitnet_packed Input packed BitNet weights [rows × (cols/4)] bytes
 * @param rows Number of rows in weight matrix
 * @param cols Number of columns (must be multiple of 4)
 * @param bit0 Output: positive indicator bits [rows × (cols/8)] bytes
 * @param bit1 Output: negative indicator bits [rows × (cols/8)] bytes
 */
ROTOR_API void bitnet_to_rotor(
    const uint8_t* bitnet_packed,
    size_t rows,
    size_t cols,
    uint8_t* bit0,
    uint8_t* bit1
);

/**
 * Unpack Rotor format bit arrays to int8 weight matrix.
 *
 * Rotor format: bit0/bit1 arrays (8 weights per byte)
 * Output: int8 matrix where weight = bit0 - bit1
 *
 * This is MUCH faster than Python bit-by-bit unpacking!
 *
 * @param bit0 Input: positive indicator bits [rows × (cols/8)] bytes
 * @param bit1 Input: negative indicator bits [rows × (cols/8)] bytes
 * @param rows Number of rows in weight matrix
 * @param cols Number of columns
 * @param weights Output: unpacked int8 weights [rows × cols]
 */
ROTOR_API void rotor_unpack_weights(
    const uint8_t* bit0,
    const uint8_t* bit1,
    size_t rows,
    size_t cols,
    int8_t* weights
);

// ============================================================================
// SIMD-Optimized Functions (compile-time selection)
// ============================================================================

#ifdef __AVX2__
int32_t rotor_dot_avx2(
    const uint8_t* a_bit0, const uint8_t* a_bit1,
    const uint8_t* b_bit0, const uint8_t* b_bit1,
    size_t n
);
#endif

#ifdef __AVX512F__
int32_t rotor_dot_avx512(
    const uint8_t* a_bit0, const uint8_t* a_bit1,
    const uint8_t* b_bit0, const uint8_t* b_bit1,
    size_t n
);
#endif

#ifdef __ARM_NEON
int32_t rotor_dot_neon(
    const uint8_t* a_bit0, const uint8_t* a_bit1,
    const uint8_t* b_bit0, const uint8_t* b_bit1,
    size_t n
);
#endif

#ifdef __cplusplus
}
#endif

#endif // ROTOR_H
