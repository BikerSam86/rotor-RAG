"""
Complete BitNet language model using Rotor ternary format.

This implements the full Microsoft BitNet-b1.58-2B-4T architecture.
"""

import sys
import io

# Fix Windows encoding (only if not already wrapped)
if hasattr(sys.stdout, 'buffer') and not isinstance(sys.stdout, io.TextIOWrapper):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
from typing import Optional, Tuple, Dict
import json
from pathlib import Path
import torch
from safetensors import safe_open

from rotor.transformer import (
    RMSNorm,
    TernaryLinear,
    MultiHeadAttention,
    GatedFFN,
    TransformerBlock
)


class BitNetModel:
    """
    Complete BitNet language model.

    Architecture:
    - Embedding layer (vocab_size × d_model)
    - N transformer blocks
    - Final RMSNorm
    - LM head (d_model × vocab_size)
    """

    def __init__(self, config: Dict, use_gpu: bool = False):
        """
        Initialize BitNet model.

        Args:
            config: Model configuration dict with:
                - vocab_size: Vocabulary size
                - d_model: Hidden dimension
                - n_layers: Number of transformer layers
                - n_heads: Number of attention heads
                - n_kv_heads: Number of KV heads (for multi-query attention)
                - d_ff: FFN intermediate dimension
            use_gpu: Whether to use GPU acceleration
        """
        self.config = config
        self.vocab_size = config['vocab_size']
        self.d_model = config['d_model']
        self.n_layers = config['n_layers']
        self.n_heads = config.get('n_heads', 20)
        self.n_kv_heads = config.get('n_kv_heads', 4)
        self.d_ff = config.get('d_ff', 6912)
        self.use_gpu = use_gpu

        # Embedding layer (learned, not ternary)
        self.embed_tokens = np.zeros((self.vocab_size, self.d_model), dtype=np.float32)

        # Transformer layers
        self.layers = []
        for i in range(self.n_layers):
            layer = TransformerBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_ff=self.d_ff,
                n_kv_heads=self.n_kv_heads,
                use_gpu=use_gpu
            )
            self.layers.append(layer)

        # Final normalization
        self.norm = RMSNorm(self.d_model)

        # LM head (could be ternary, but using tied embeddings for now)
        self.lm_head_weight = None  # Will tie to embeddings

        print(f"✓ Initialized BitNet model:")
        print(f"  - Vocabulary: {self.vocab_size:,}")
        print(f"  - Hidden dim: {self.d_model}")
        print(f"  - Layers: {self.n_layers}")
        print(f"  - Heads: {self.n_heads} (KV: {self.n_kv_heads})")
        print(f"  - FFN dim: {self.d_ff}")

    def load_weights_from_safetensors(self, model_path: str):
        """
        Load weights from BitNet safetensors file.

        This converts all ternary weights from BitNet format to Rotor format!
        """
        print(f"\nLoading weights from: {model_path}")

        with safe_open(model_path, framework="pt") as f:
            keys = f.keys()

            # Load embeddings
            print("  Loading embeddings...")
            embed_weight = f.get_tensor("model.embed_tokens.weight")
            if embed_weight.dtype == torch.bfloat16:
                embed_weight = embed_weight.float()
            self.embed_tokens = embed_weight.cpu().numpy()
            print(f"    ✓ Embeddings: {self.embed_tokens.shape}")

            # Load final norm
            print("  Loading final norm...")
            norm_weight = f.get_tensor("model.norm.weight")
            if norm_weight.dtype == torch.bfloat16:
                norm_weight = norm_weight.float()
            self.norm.weight = norm_weight.cpu().numpy()
            print(f"    ✓ Final norm: {self.norm.weight.shape}")

            # Load transformer layers
            print(f"  Loading {self.n_layers} transformer layers...")
            for layer_idx in range(self.n_layers):
                self._load_layer_weights(f, layer_idx)
                if (layer_idx + 1) % 5 == 0:
                    print(f"    ✓ Loaded {layer_idx + 1}/{self.n_layers} layers...")

            print(f"  ✓ All layers loaded!")

        print(f"\n✓ Model weights loaded successfully!")
        print(f"  Total transformer layers: {self.n_layers}")
        print(f"  All ternary weights converted to Rotor format!")

    def _load_layer_weights(self, f, layer_idx: int):
        """Load weights for a single transformer layer."""
        layer = self.layers[layer_idx]
        prefix = f"model.layers.{layer_idx}"

        # Load normalization weights
        input_norm = f.get_tensor(f"{prefix}.input_layernorm.weight")
        if input_norm.dtype == torch.bfloat16:
            input_norm = input_norm.float()
        layer.input_norm.weight = input_norm.cpu().numpy()

        post_attn_norm = f.get_tensor(f"{prefix}.post_attention_layernorm.weight")
        if post_attn_norm.dtype == torch.bfloat16:
            post_attn_norm = post_attn_norm.float()
        layer.post_attn_norm.weight = post_attn_norm.cpu().numpy()

        # Load attention weights (ternary!)
        self._load_ternary_weight(f, layer.attention.q_proj, f"{prefix}.self_attn.q_proj")
        self._load_ternary_weight(f, layer.attention.k_proj, f"{prefix}.self_attn.k_proj")
        self._load_ternary_weight(f, layer.attention.v_proj, f"{prefix}.self_attn.v_proj")
        self._load_ternary_weight(f, layer.attention.o_proj, f"{prefix}.self_attn.o_proj")

        # Load attention sub-norm
        attn_norm = f.get_tensor(f"{prefix}.self_attn.attn_sub_norm.weight")
        if attn_norm.dtype == torch.bfloat16:
            attn_norm = attn_norm.float()
        layer.attention.attn_norm.weight = attn_norm.cpu().numpy()

        # Load FFN weights (ternary!)
        self._load_ternary_weight(f, layer.ffn.gate_proj, f"{prefix}.mlp.gate_proj")
        self._load_ternary_weight(f, layer.ffn.up_proj, f"{prefix}.mlp.up_proj")
        self._load_ternary_weight(f, layer.ffn.down_proj, f"{prefix}.mlp.down_proj")

        # Load FFN sub-norm
        ffn_norm = f.get_tensor(f"{prefix}.mlp.ffn_sub_norm.weight")
        if ffn_norm.dtype == torch.bfloat16:
            ffn_norm = ffn_norm.float()
        layer.ffn.ffn_norm.weight = ffn_norm.cpu().numpy()

    def _load_ternary_weight(self, f, linear_layer: TernaryLinear, weight_name: str):
        """
        Load a ternary weight and convert from BitNet to Rotor format.

        This is where the magic happens - converting Microsoft's packed format
        to our superior aligned format!
        """
        # Load packed weight (uint8)
        packed_weight = f.get_tensor(f"{weight_name}.weight").cpu().numpy()

        # Load scale
        scale_tensor = f.get_tensor(f"{weight_name}.weight_scale")
        if scale_tensor.dtype == torch.bfloat16:
            scale_tensor = scale_tensor.float()
        scale = scale_tensor.cpu().numpy().item()

        # Convert BitNet → Rotor format
        linear_layer.load_from_bitnet(packed_weight, scale)

    def forward(
        self,
        input_ids: np.ndarray,
        return_logits: bool = True,
        past_kv_cache: Optional[list] = None,
        use_cache: bool = False
    ) -> tuple[np.ndarray, Optional[list]]:
        """
        Forward pass through the model.

        Args:
            input_ids: Token IDs [batch, seq_len] or [seq_len]
            return_logits: If True, return logits. If False, return hidden states.
            past_kv_cache: Optional list of KV caches from previous forward passes
            use_cache: If True, return KV caches for future use

        Returns:
            logits: [batch, seq_len, vocab_size] or [seq_len, vocab_size]
            kv_caches: List of KV caches for each layer (if use_cache=True), else None
        """
        # Handle 1D input
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]  # Add batch dimension
            squeeze_batch = True
        else:
            squeeze_batch = False

        batch_size, seq_len = input_ids.shape

        # Embedding lookup
        hidden_states = self.embed_tokens[input_ids]  # [batch, seq_len, d_model]

        # Through transformer layers
        new_kv_caches = [] if use_cache else None
        for layer_idx, layer in enumerate(self.layers):
            layer_cache = past_kv_cache[layer_idx] if past_kv_cache is not None else None
            hidden_states, new_cache = layer.forward(hidden_states, kv_cache=layer_cache, use_cache=use_cache)
            if use_cache:
                new_kv_caches.append(new_cache)

        # Final normalization
        hidden_states = self.norm.forward(hidden_states)

        if return_logits:
            # LM head (using tied embeddings)
            logits = np.dot(hidden_states, self.embed_tokens.T)  # [batch, seq_len, vocab_size]

            if squeeze_batch:
                logits = logits[0]

            return logits, new_kv_caches
        else:
            if squeeze_batch:
                hidden_states = hidden_states[0]
            return hidden_states, new_kv_caches

    def generate(
        self,
        input_ids: np.ndarray,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> np.ndarray:
        """
        Generate text autoregressively.

        Args:
            input_ids: Prompt token IDs [seq_len]
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling

        Returns:
            generated_ids: Complete sequence [seq_len + max_new_tokens]
        """
        generated = input_ids.copy()

        for _ in range(max_new_tokens):
            # Forward pass
            logits = self.forward(generated)  # [seq_len, vocab_size]

            # Get logits for last token
            next_token_logits = logits[-1, :]  # [vocab_size]

            # Sample next token
            next_token = self._sample(
                next_token_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )

            # Append to sequence
            generated = np.append(generated, next_token)

        return generated

    def _sample(
        self,
        logits: np.ndarray,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None
    ) -> int:
        """
        Sample next token from logits.

        Args:
            logits: Logits for next token [vocab_size]
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Nucleus sampling

        Returns:
            token_id: Sampled token ID
        """
        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature

        # Convert to probabilities
        probs = self._softmax(logits)

        # Top-k sampling
        if top_k is not None:
            top_k_indices = np.argsort(probs)[-top_k:]
            mask = np.zeros_like(probs)
            mask[top_k_indices] = 1
            probs = probs * mask
            probs = probs / probs.sum()

        # Top-p (nucleus) sampling
        if top_p is not None:
            sorted_indices = np.argsort(probs)[::-1]
            sorted_probs = probs[sorted_indices]
            cumsum_probs = np.cumsum(sorted_probs)

            # Find cutoff
            cutoff_idx = np.searchsorted(cumsum_probs, top_p) + 1

            # Mask
            mask = np.zeros_like(probs)
            mask[sorted_indices[:cutoff_idx]] = 1
            probs = probs * mask
            probs = probs / probs.sum()

        # Sample
        token_id = np.random.choice(len(probs), p=probs)
        return token_id

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        x_max = np.max(x)
        exp_x = np.exp(x - x_max)
        return exp_x / np.sum(exp_x)


def load_bitnet_model(model_dir: str, use_gpu: bool = False) -> BitNetModel:
    """
    Load complete BitNet model from directory.

    Args:
        model_dir: Directory containing model.safetensors and config.json
        use_gpu: Whether to use GPU acceleration

    Returns:
        model: Loaded BitNetModel
    """
    model_dir = Path(model_dir)

    # Load config
    config_path = model_dir / "config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Extract relevant config
    model_config = {
        'vocab_size': config['vocab_size'],
        'd_model': config['hidden_size'],
        'n_layers': config['num_hidden_layers'],
        'n_heads': config['num_attention_heads'],
        'n_kv_heads': config.get('num_key_value_heads', config['num_attention_heads']),
        'd_ff': config['intermediate_size'],
    }

    # Create model
    print("="*70)
    print("Loading BitNet Model" + (" (GPU Accelerated)" if use_gpu else ""))
    print("="*70)
    model = BitNetModel(model_config, use_gpu=use_gpu)

    # Load weights
    weights_path = model_dir / "model.safetensors"
    model.load_weights_from_safetensors(str(weights_path))

    print("\n" + "="*70)
    print("✓ BitNet model loaded successfully!")
    print("="*70)
    print(f"Ready for inference with {model.n_layers} layers!")
    print(f"All ternary weights in optimized Rotor format!")
    print("="*70)

    return model


# Example usage
if __name__ == "__main__":
    import sys
    import io

    # Fix Windows encoding
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("\nTesting BitNetModel class...")

    # Create small test model
    test_config = {
        'vocab_size': 1000,
        'd_model': 256,
        'n_layers': 2,
        'n_heads': 8,
        'n_kv_heads': 4,
        'd_ff': 1024,
    }

    model = BitNetModel(test_config)

    # Test forward pass
    print("\nTesting forward pass...")
    input_ids = np.array([1, 2, 3, 4, 5])
    logits = model.forward(input_ids)

    print(f"Input: {input_ids.shape}")
    print(f"Output: {logits.shape}")
    print(f"\n✓ BitNetModel working!")
