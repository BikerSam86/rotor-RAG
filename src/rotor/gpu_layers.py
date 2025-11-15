"""
GPU-accelerated layers for BitNet transformer.
Uses OpenCL for Intel HD Graphics and future Vulkan for Steam Deck.
"""

import numpy as np
from typing import Optional
from rotor.gpu_ternary import GPUTernaryOps

# Global GPU instance (initialized once)
_gpu_instance = None


def get_gpu():
    """Get or create GPU instance."""
    global _gpu_instance
    if _gpu_instance is None:
        try:
            _gpu_instance = GPUTernaryOps()
            print("[GPU] OpenCL acceleration enabled")
        except Exception as e:
            print(f"[GPU] Warning: Failed to initialize GPU: {e}")
            print("[GPU] Falling back to CPU")
            _gpu_instance = False  # Mark as failed, don't retry
    return _gpu_instance if _gpu_instance else None


class GPUBitNetLinear:
    """
    GPU-accelerated BitNet linear layer.
    Falls back to CPU if GPU unavailable.
    """

    def __init__(self, in_features: int, out_features: int, scale: float = 1.0, use_gpu: bool = True):
        """
        Args:
            in_features: Input dimension
            out_features: Output dimension
            scale: Weight scale factor
            use_gpu: Whether to use GPU acceleration
        """
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.use_gpu = use_gpu

        # Storage for weights (will be set by load_from_bitnet)
        self.weights_cache = None  # CPU weights [out_features, in_features]
        self.gpu_packed_weights = None  # GPU packed weights
        self.gpu_scales = None  # GPU scale array

        # GPU instance
        self.gpu = get_gpu() if use_gpu else None

    def load_from_bitnet(self, packed_weights: np.ndarray, scale: float = 1.0):
        """
        Load weights from BitNet packed format.

        Args:
            packed_weights: BitNet format [out_features, in_features] in uint8
            scale: Scaling factor
        """
        from rotor.bitnet_fast import bitnet_to_rotor_fast

        self.scale = scale

        # Convert to Rotor format
        bit0_2d, bit1_2d = bitnet_to_rotor_fast(packed_weights)

        # Decode weights for CPU/GPU
        from rotor.rotor_ops import decode_ternary

        self.weights_cache = decode_ternary(bit0_2d.flatten(), bit1_2d.flatten())
        self.weights_cache = self.weights_cache.reshape(self.out_features, self.in_features)

        # Prepare GPU weights if available
        if self.gpu and self.use_gpu:
            self.gpu_packed_weights = self.gpu.pack_ternary_weights(self.weights_cache)
            self.gpu_scales = np.full(self.out_features, self.scale, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass.

        Args:
            x: Input [batch, seq_len, in_features] or [seq_len, in_features] or [in_features]

        Returns:
            Output [batch, seq_len, out_features] or [seq_len, out_features] or [out_features]
        """
        input_shape = x.shape
        needs_reshape = len(input_shape) > 1

        # Flatten to 2D if needed [total_tokens, in_features]
        if needs_reshape:
            x_2d = x.reshape(-1, self.in_features)
        else:
            x_2d = x.reshape(1, -1) if x.ndim == 1 else x

        # Try GPU acceleration
        if self.gpu and self.use_gpu and self.gpu_packed_weights is not None:
            try:
                # Process each token separately (could batch later)
                outputs = []
                for token_vec in x_2d:
                    out = self.gpu.ternary_matmul(
                        self.gpu_packed_weights,
                        self.gpu_scales,
                        token_vec.astype(np.float32),
                        self.in_features,
                        self.out_features
                    )
                    outputs.append(out)
                result = np.array(outputs, dtype=np.float32)
            except Exception as e:
                print(f"[GPU] Warning: GPU forward failed: {e}, falling back to CPU")
                # Fall back to CPU
                result = (x_2d @ self.weights_cache.T) * self.scale
        else:
            # CPU path
            result = (x_2d @ self.weights_cache.T) * self.scale

        # Reshape back to original shape
        if needs_reshape:
            if len(input_shape) == 3:
                result = result.reshape(input_shape[0], input_shape[1], self.out_features)
            else:  # len == 2
                result = result.reshape(input_shape[0], self.out_features)
        elif x.ndim == 1:
            result = result.flatten()

        return result


def test_gpu_layer():
    """Test GPU-accelerated layer."""
    print("=" * 70)
    print("GPU LAYER TEST")
    print("=" * 70)

    # Create layer
    in_dim = 2560
    out_dim = 2560

    layer = GPUBitNetLinear(in_dim, out_dim, scale=1.2, use_gpu=True)

    # Create fake weights
    weights = np.random.choice([-1, 0, 1], size=(out_dim, in_dim))
    layer.weights_cache = weights.astype(np.float32)

    # Prepare GPU weights
    if layer.gpu:
        layer.gpu_packed_weights = layer.gpu.pack_ternary_weights(weights)
        layer.gpu_scales = np.full(out_dim, layer.scale, dtype=np.float32)

    # Test input
    x = np.random.randn(5, in_dim).astype(np.float32)  # 5 tokens

    print(f"\nTesting {out_dim}x{in_dim} layer with 5 tokens...")

    # GPU forward
    import time
    start = time.perf_counter()
    gpu_out = layer.forward(x)
    gpu_time = time.perf_counter() - start

    # CPU forward
    start = time.perf_counter()
    cpu_out = (x @ weights.T) * layer.scale
    cpu_time = time.perf_counter() - start

    # Compare
    max_diff = np.abs(gpu_out - cpu_out).max()

    print(f"\nResults:")
    print(f"  GPU time: {gpu_time*1000:.2f}ms")
    print(f"  CPU time: {cpu_time*1000:.2f}ms")
    print(f"  Speedup: {cpu_time/gpu_time:.2f}x")
    print(f"  Max diff: {max_diff:.6f}")

    if max_diff < 0.01:
        print("  [OK] Results match!")

    print("\n" + "=" * 70)
    print("All ways, always!")
    print("=" * 70)


if __name__ == "__main__":
    test_gpu_layer()
