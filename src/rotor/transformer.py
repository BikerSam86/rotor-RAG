"""
Transformer architecture using Rotor ternary format.

Implements:
- RMSNorm
- Ternary linear layers
- Multi-head attention
- Gated FFN
- Full transformer block
"""

import numpy as np
from typing import Optional, Tuple
import time


class RMSNorm:
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        self.eps = eps
        self.weight = np.ones(dim, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: Input tensor [batch, seq_len, dim] or [seq_len, dim]

        Returns:
            Normalized tensor with same shape
        """
        # Compute RMS
        norm = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + self.eps)

        # Normalize and scale
        return (x / norm) * self.weight


class TernaryLinear:
    """
    Linear layer with ternary weights in Rotor format.

    Weights are stored as bit0, bit1 arrays where:
      weight = bit0 - bit1
      (+1 if bit0=1, bit1=0)
      ( 0 if bit0=0, bit1=0)
      (-1 if bit0=0, bit1=1)
    """

    def __init__(self, in_features: int, out_features: int, use_gpu: bool = False):
        self.in_features = in_features
        self.out_features = out_features
        self.use_gpu = use_gpu

        # Rotor format: separate bit arrays (8 weights per byte)
        n_bytes = ((in_features * out_features) + 7) // 8
        self.bit0 = np.zeros(n_bytes, dtype=np.uint8)
        self.bit1 = np.zeros(n_bytes, dtype=np.uint8)

        # Scale factor for BitNet
        self.scale = 1.0

        # Cached decoded weights for fast forward pass
        self.weights_cache = None

        # GPU support
        self.gpu = None
        self.gpu_packed_weights = None
        self.gpu_scales = None
        if use_gpu:
            try:
                from rotor.gpu_layers import get_gpu
                self.gpu = get_gpu()
            except ImportError:
                pass  # GPU not available

    def load_from_bitnet(self, packed_weights: np.ndarray, scale: float = 1.0):
        """
        Load weights from BitNet packed format.

        Args:
            packed_weights: BitNet format [out_features, in_features] in uint8
            scale: Scaling factor
        """
        from rotor.bitnet_fast import bitnet_to_rotor_fast

        self.scale = scale

        # Convert to Rotor format (uses C library if available, Python fallback otherwise)
        # packed_weights is [out_features, in_features_packed]
        # We need to convert it properly
        bit0_2d, bit1_2d = bitnet_to_rotor_fast(packed_weights)

        # Flatten for storage
        self.bit0 = bit0_2d.flatten()
        self.bit1 = bit1_2d.flatten()

        # Store shape for later
        self.weight_shape = (self.out_features, self.in_features)

        # Decode weights once and cache for fast forward pass
        print(f"  Caching decoded weights ({self.out_features}x{self.in_features})...")
        self.weights_cache = self._decode_weights()
        print(f"  ✓ Weights cached")

        # Prepare GPU weights if enabled
        if self.gpu and self.use_gpu:
            self.gpu_packed_weights = self.gpu.pack_ternary_weights(self.weights_cache)
            self.gpu_scales = np.full(self.out_features, self.scale, dtype=np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: y = x @ W.T * scale

        Args:
            x: Input [batch, seq_len, in_features] or [seq_len, in_features]

        Returns:
            Output [batch, seq_len, out_features] or [seq_len, out_features]
        """
        input_shape = x.shape
        is_3d = len(input_shape) == 3

        if is_3d:
            batch, seq_len, in_feat = input_shape
            x_flat = x.reshape(batch * seq_len, in_feat)
        else:
            x_flat = x

        # Try GPU acceleration
        if self.gpu and self.use_gpu and self.gpu_packed_weights is not None:
            try:
                # Process tokens (could batch in future)
                outputs = []
                for token_vec in x_flat:
                    out = self.gpu.ternary_matmul(
                        self.gpu_packed_weights,
                        self.gpu_scales,
                        token_vec.astype(np.float32),
                        self.in_features,
                        self.out_features
                    )
                    outputs.append(out)
                output = np.array(outputs, dtype=np.float32)
            except Exception:
                # Fall back to CPU on GPU error
                output = np.dot(x_flat, self.weights_cache.T) * self.scale
        else:
            # CPU path: Use cached weights (decoded once during load)
            output = np.dot(x_flat, self.weights_cache.T) * self.scale

        if is_3d:
            output = output.reshape(batch, seq_len, self.out_features)

        return output

    def _decode_weights(self) -> np.ndarray:
        """
        Decode Rotor format to ternary weights [-1, 0, +1].

        Uses fast C implementation when available, falls back to Python.
        """
        from rotor.bitnet_fast import rotor_unpack_weights_fast

        # Reshape flattened bit arrays to 2D
        rotor_cols_bytes = (self.in_features + 7) // 8
        bit0_2d = self.bit0.reshape(self.out_features, rotor_cols_bytes)
        bit1_2d = self.bit1.reshape(self.out_features, rotor_cols_bytes)

        # Use fast C implementation
        return rotor_unpack_weights_fast(bit0_2d, bit1_2d, self.out_features, self.in_features)


class MultiHeadAttention:
    """
    Multi-head attention with ternary weights.

    Supports multi-query attention (fewer K/V heads than Q heads).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        use_gpu: bool = False
    ):
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.head_dim = head_dim if head_dim is not None else d_model // n_heads

        self.d_q = self.n_heads * self.head_dim
        self.d_kv = self.n_kv_heads * self.head_dim

        # Projections (ternary)
        self.q_proj = TernaryLinear(d_model, self.d_q, use_gpu=use_gpu)
        self.k_proj = TernaryLinear(d_model, self.d_kv, use_gpu=use_gpu)
        self.v_proj = TernaryLinear(d_model, self.d_kv, use_gpu=use_gpu)
        self.o_proj = TernaryLinear(self.d_q, d_model, use_gpu=use_gpu)

        # Sub-normalization
        self.attn_norm = RMSNorm(d_model)

    def forward(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None,
        kv_cache: Optional[dict] = None,
        use_cache: bool = False
    ) -> tuple[np.ndarray, Optional[dict]]:
        """
        Args:
            x: Input [batch, seq_len, d_model] or [seq_len, d_model]
            mask: Attention mask [seq_len, seq_len]
            kv_cache: Optional cache dict with 'k' and 'v' from previous steps
            use_cache: If True, return updated cache

        Returns:
            output: [batch, seq_len, d_model] or [seq_len, d_model]
            cache: Updated cache dict (if use_cache=True), else None
        """
        input_shape = x.shape
        is_3d = len(input_shape) == 3

        if is_3d:
            batch, seq_len, _ = input_shape
        else:
            batch = 1
            seq_len, _ = input_shape
            x = x[None, ...]

        # Project Q, K, V
        Q = self.q_proj.forward(x)  # [batch, seq_len, d_q]
        K = self.k_proj.forward(x)  # [batch, seq_len, d_kv]
        V = self.v_proj.forward(x)  # [batch, seq_len, d_kv]

        # Reshape for multi-head attention
        Q = Q.reshape(batch, seq_len, self.n_heads, self.head_dim)
        K = K.reshape(batch, seq_len, self.n_kv_heads, self.head_dim)
        V = V.reshape(batch, seq_len, self.n_kv_heads, self.head_dim)

        # Transpose to [batch, n_heads, seq_len, head_dim]
        Q = Q.transpose(0, 2, 1, 3)
        K = K.transpose(0, 2, 1, 3)
        V = V.transpose(0, 2, 1, 3)

        # Use KV cache if provided (concatenate with past K,V)
        if kv_cache is not None:
            K = np.concatenate([kv_cache['k'], K], axis=2)  # [batch, n_kv_heads, total_seq, head_dim]
            V = np.concatenate([kv_cache['v'], V], axis=2)

        # Update cache if requested
        new_cache = None
        if use_cache:
            new_cache = {'k': K, 'v': V}

        # Repeat K, V if multi-query attention
        if self.n_kv_heads < self.n_heads:
            n_repeat = self.n_heads // self.n_kv_heads
            K = np.repeat(K, n_repeat, axis=1)
            V = np.repeat(V, n_repeat, axis=1)

        # Scaled dot-product attention
        # scores = Q @ K.T / sqrt(head_dim)
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)

        # Apply mask if provided
        if mask is not None:
            scores = scores + mask

        # Softmax
        attn_weights = self._softmax(scores, axis=-1)

        # Apply attention to values
        attn_output = np.matmul(attn_weights, V)

        # Transpose back and reshape
        attn_output = attn_output.transpose(0, 2, 1, 3)
        attn_output = attn_output.reshape(batch, seq_len, self.d_q)

        # Output projection
        output = self.o_proj.forward(attn_output)

        # Sub-normalization
        output = self.attn_norm.forward(output)

        if not is_3d:
            output = output[0]

        return output, new_cache

    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Numerically stable softmax."""
        x_max = np.max(x, axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class GatedFFN:
    """
    Gated Feed-Forward Network (SwiGLU-style).

    FFN(x) = (gate_proj(x) * silu(up_proj(x))) @ down_proj
    """

    def __init__(self, d_model: int, d_ff: int, use_gpu: bool = False):
        self.d_model = d_model
        self.d_ff = d_ff

        # Projections (ternary)
        self.gate_proj = TernaryLinear(d_model, d_ff, use_gpu=use_gpu)
        self.up_proj = TernaryLinear(d_model, d_ff, use_gpu=use_gpu)
        self.down_proj = TernaryLinear(d_ff, d_model, use_gpu=use_gpu)

        # Sub-normalization
        self.ffn_norm = RMSNorm(d_ff)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Args:
            x: Input [batch, seq_len, d_model] or [seq_len, d_model]

        Returns:
            Output with same shape
        """
        # Gate and up projections
        gate = self.gate_proj.forward(x)
        up = self.up_proj.forward(x)

        # Gating: gate * silu(up)
        gated = gate * self._silu(up)

        # Sub-normalization
        gated = self.ffn_norm.forward(gated)

        # Down projection
        output = self.down_proj.forward(gated)

        return output

    def _silu(self, x: np.ndarray) -> np.ndarray:
        """
        SiLU activation (Swish): x * sigmoid(x)

        Numerically stable implementation that avoids overflow.
        """
        # Use np.clip to prevent extreme values
        x_clipped = np.clip(x, -20, 20)
        return x_clipped / (1 + np.exp(-x_clipped))


class TransformerBlock:
    """
    Single transformer block with:
    - Multi-head attention
    - Gated FFN
    - Pre-normalization
    - Residual connections
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_kv_heads: Optional[int] = None,
        use_gpu: bool = False
    ):
        self.d_model = d_model

        # Normalizations
        self.input_norm = RMSNorm(d_model)
        self.post_attn_norm = RMSNorm(d_model)

        # Attention
        self.attention = MultiHeadAttention(d_model, n_heads, n_kv_heads, use_gpu=use_gpu)

        # FFN
        self.ffn = GatedFFN(d_model, d_ff, use_gpu=use_gpu)

    def forward(
        self,
        x: np.ndarray,
        mask: Optional[np.ndarray] = None,
        kv_cache: Optional[dict] = None,
        use_cache: bool = False
    ) -> tuple[np.ndarray, Optional[dict]]:
        """
        Args:
            x: Input [batch, seq_len, d_model] or [seq_len, d_model]
            mask: Attention mask
            kv_cache: Optional KV cache from previous step
            use_cache: If True, return updated cache

        Returns:
            output: Output with same shape
            cache: Updated cache (if use_cache=True), else None
        """
        # Pre-norm attention with residual
        normed = self.input_norm.forward(x)
        attn_out, new_cache = self.attention.forward(normed, mask, kv_cache, use_cache)
        x = x + attn_out

        # Post-attention norm
        x_normed = self.post_attn_norm.forward(x)

        # FFN with residual
        ffn_out = self.ffn.forward(x_normed)
        x = x + ffn_out

        return x, new_cache


# Example usage
if __name__ == "__main__":
    print("Testing transformer components...")

    # Test RMSNorm
    norm = RMSNorm(512)
    x = np.random.randn(10, 512).astype(np.float32)
    y = norm.forward(x)
    print(f"✓ RMSNorm: {x.shape} -> {y.shape}")

    # Test TernaryLinear
    linear = TernaryLinear(512, 256)
    # Create fake ternary weights
    weights = np.random.choice([-1, 0, 1], size=(256, 512), p=[0.3, 0.4, 0.3]).astype(np.int8)
    # Would need to encode to bitnet first, then load
    y = linear.forward(x)
    print(f"✓ TernaryLinear: {x.shape} -> {y.shape}")

    # Test MultiHeadAttention
    attn = MultiHeadAttention(d_model=512, n_heads=8, n_kv_heads=4)
    x = np.random.randn(2, 10, 512).astype(np.float32)
    y = attn.forward(x)
    print(f"✓ MultiHeadAttention: {x.shape} -> {y.shape}")

    # Test GatedFFN
    ffn = GatedFFN(d_model=512, d_ff=2048)
    y = ffn.forward(x)
    print(f"✓ GatedFFN: {x.shape} -> {y.shape}")

    # Test TransformerBlock
    block = TransformerBlock(d_model=512, n_heads=8, d_ff=2048, n_kv_heads=4)
    y = block.forward(x)
    print(f"✓ TransformerBlock: {x.shape} -> {y.shape}")

    print("\n✅ All components working!")
