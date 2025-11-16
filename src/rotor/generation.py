"""
Text generation utilities for BitNet models.

Provides sampling strategies and generation loops for autoregressive text generation.
"""

import numpy as np
from typing import Optional, List, Callable


class SamplingStrategy:
    """Base class for sampling strategies."""

    def sample(self, logits: np.ndarray, temperature: float = 1.0) -> int:
        """
        Sample a token from logits.

        Args:
            logits: Logit scores for each token (shape: [vocab_size])
            temperature: Temperature for sampling (higher = more random)

        Returns:
            Sampled token ID
        """
        raise NotImplementedError


class GreedySampling(SamplingStrategy):
    """Greedy sampling: always pick the highest probability token."""

    def sample(self, logits: np.ndarray, temperature: float = 1.0) -> int:
        """
        Greedy sampling (argmax).

        Args:
            logits: Logit scores for each token
            temperature: Ignored for greedy sampling

        Returns:
            Token ID with highest logit
        """
        return int(np.argmax(logits))


class TopKSampling(SamplingStrategy):
    """Top-k sampling: sample from the k highest probability tokens."""

    def __init__(self, k: int = 50):
        """
        Initialize top-k sampling.

        Args:
            k: Number of top tokens to consider
        """
        self.k = k

    def sample(self, logits: np.ndarray, temperature: float = 1.0) -> int:
        """
        Top-k sampling.

        Args:
            logits: Logit scores for each token
            temperature: Temperature for sampling

        Returns:
            Sampled token ID
        """
        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature

        # Get top-k indices
        top_k_indices = np.argpartition(logits, -self.k)[-self.k:]
        top_k_logits = logits[top_k_indices]

        # Convert to probabilities
        probs = self._softmax(top_k_logits)

        # Sample from top-k
        sampled_idx = np.random.choice(len(top_k_indices), p=probs)
        return int(top_k_indices[sampled_idx])

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Compute softmax with numerical stability."""
        x_shifted = x - np.max(x)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x)


class TopPSampling(SamplingStrategy):
    """Top-p (nucleus) sampling: sample from smallest set of tokens with cumulative probability >= p."""

    def __init__(self, p: float = 0.9):
        """
        Initialize top-p sampling.

        Args:
            p: Cumulative probability threshold (0.0 to 1.0)
        """
        self.p = p

    def sample(self, logits: np.ndarray, temperature: float = 1.0) -> int:
        """
        Top-p (nucleus) sampling.

        Args:
            logits: Logit scores for each token
            temperature: Temperature for sampling

        Returns:
            Sampled token ID
        """
        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature

        # Convert to probabilities
        probs = self._softmax(logits)

        # Sort by probability (descending)
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]

        # Find cumulative probability threshold
        cumsum_probs = np.cumsum(sorted_probs)
        nucleus_size = np.searchsorted(cumsum_probs, self.p) + 1

        # Keep only nucleus tokens
        nucleus_indices = sorted_indices[:nucleus_size]
        nucleus_probs = sorted_probs[:nucleus_size]

        # Renormalize
        nucleus_probs = nucleus_probs / np.sum(nucleus_probs)

        # Sample from nucleus
        sampled_idx = np.random.choice(len(nucleus_indices), p=nucleus_probs)
        return int(nucleus_indices[sampled_idx])

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Compute softmax with numerical stability."""
        x_shifted = x - np.max(x)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x)


class TextGenerator:
    """Text generation with various sampling strategies."""

    def __init__(
        self,
        model,
        tokenizer,
        sampling_strategy: Optional[SamplingStrategy] = None,
        temperature: float = 1.0,
        max_length: int = 100,
        use_cache: bool = True,
    ):
        """
        Initialize text generator.

        Args:
            model: BitNet model for inference
            tokenizer: Tokenizer for text encoding/decoding
            sampling_strategy: Sampling strategy (default: greedy)
            temperature: Sampling temperature
            max_length: Maximum number of tokens to generate
            use_cache: Enable KV caching for faster generation (default: True)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.sampling_strategy = sampling_strategy or GreedySampling()
        self.temperature = temperature
        self.max_length = max_length
        self.use_cache = use_cache
        self.kv_cache = None  # Will store KV cache during generation

    def generate(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_on_eos: bool = True,
        callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of new tokens to generate (default: self.max_length)
            temperature: Override default temperature
            stop_on_eos: Stop generation when EOS token is generated
            callback: Optional callback(token_text) called after each token

        Returns:
            Generated text (including prompt)
        """
        # Use defaults if not specified
        max_new_tokens = max_new_tokens or self.max_length
        temperature = temperature if temperature is not None else self.temperature

        # Encode prompt
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=True, return_numpy=True)
        token_ids_list = token_ids.tolist()

        # Calculate actual tokens to generate (respect max_length as total limit)
        current_length = len(token_ids_list)
        tokens_to_generate = min(max_new_tokens, self.max_length - current_length)

        # Don't generate if we're already at or over max_length
        if tokens_to_generate <= 0:
            return self.tokenizer.decode(token_ids, skip_special_tokens=True)

        # Reset KV cache for new generation
        self.kv_cache = None

        # Generate tokens
        for step_idx in range(tokens_to_generate):
            # Get logits for next token (uses KV cache internally)
            logits = self._get_next_token_logits(token_ids, is_first_step=(step_idx == 0))

            # Sample next token
            next_token_id = self.sampling_strategy.sample(logits, temperature)

            # Stop on EOS
            if stop_on_eos and next_token_id == self.tokenizer.eos_token_id:
                break

            # Add to sequence
            token_ids_list.append(next_token_id)
            token_ids = np.array(token_ids_list, dtype=np.int64)

            # Callback with newly generated token
            if callback:
                new_token_text = self.tokenizer.decode([next_token_id], skip_special_tokens=True)
                callback(new_token_text)

        # Decode full sequence
        generated_text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        return generated_text

    def _get_next_token_logits(self, token_ids: np.ndarray, is_first_step: bool = False) -> np.ndarray:
        """
        Get logits for next token.

        Args:
            token_ids: Input token IDs (shape: [seq_len])
            is_first_step: Whether this is the first generation step (pass full prompt)

        Returns:
            Logits for next token (shape: [vocab_size])
        """
        if self.use_cache:
            if is_first_step:
                # First step: pass full prompt, get cache
                input_ids = token_ids.reshape(1, -1)
                logits, self.kv_cache = self.model.forward(input_ids, use_cache=True)
            else:
                # Subsequent steps: only pass new token, use cached K,V
                # Only process the last token
                new_token = token_ids[-1:]
                input_ids = new_token.reshape(1, -1)
                logits, self.kv_cache = self.model.forward(
                    input_ids,
                    past_kv_cache=self.kv_cache,
                    use_cache=True
                )
        else:
            # No caching: always pass full sequence
            input_ids = token_ids.reshape(1, -1)
            logits, _ = self.model.forward(input_ids, use_cache=False)

        # Get logits for last position
        # logits shape: [batch_size, seq_len, vocab_size]
        next_token_logits = logits[0, -1, :]

        return next_token_logits

    def generate_streaming(
        self,
        prompt: str,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_on_eos: bool = True,
    ):
        """
        Generate text with streaming (yields tokens as they're generated).

        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Override default temperature
            stop_on_eos: Stop generation when EOS token is generated

        Yields:
            Generated tokens (text strings)
        """
        # Use defaults if not specified
        max_new_tokens = max_new_tokens or self.max_length
        temperature = temperature if temperature is not None else self.temperature

        # Encode prompt
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=True, return_numpy=True)
        token_ids_list = token_ids.tolist()

        # Calculate actual tokens to generate (respect max_length as total limit)
        current_length = len(token_ids_list)
        tokens_to_generate = min(max_new_tokens, self.max_length - current_length)

        # Yield prompt
        yield prompt

        # Don't generate if we're already at or over max_length
        if tokens_to_generate <= 0:
            return

        # Reset KV cache for new generation
        self.kv_cache = None

        # Generate tokens
        for step_idx in range(tokens_to_generate):
            # Get logits for next token (uses KV cache internally)
            logits = self._get_next_token_logits(token_ids, is_first_step=(step_idx == 0))

            # Sample next token
            next_token_id = self.sampling_strategy.sample(logits, temperature)

            # Stop on EOS
            if stop_on_eos and next_token_id == self.tokenizer.eos_token_id:
                break

            # Add to sequence
            token_ids_list.append(next_token_id)
            token_ids = np.array(token_ids_list, dtype=np.int64)

            # Decode and yield new token
            new_token_text = self.tokenizer.decode([next_token_id], skip_special_tokens=True)
            yield new_token_text


# Alias for clarity (Microsoft uses "top_p" terminology)
NucleusSampling = TopPSampling


# Example usage
if __name__ == "__main__":
    print("Text generation utilities loaded!")
    print("\nAvailable sampling strategies:")
    print("  - GreedySampling: Always pick highest probability token")
    print("  - TopKSampling(k=50): Sample from top k tokens")
    print("  - TopPSampling(p=0.9): Sample from nucleus (cumulative prob >= p)")
    print("  - NucleusSampling: Alias for TopPSampling")
    print("\nUse TextGenerator to generate text with a loaded model and tokenizer.")
