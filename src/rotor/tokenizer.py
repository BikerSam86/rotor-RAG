"""
Tokenizer wrapper for BitNet models.

Loads and uses HuggingFace tokenizers with the Rotor BitNet model.
"""

import json
from pathlib import Path
from typing import List, Union
import numpy as np


class BitNetTokenizer:
    """Tokenizer for BitNet models."""

    def __init__(self, model_dir: str):
        """
        Load tokenizer from model directory.

        Args:
            model_dir: Path to model directory containing tokenizer files
        """
        self.model_dir = Path(model_dir)

        # Try to load with tokenizers library (fast!)
        try:
            from tokenizers import Tokenizer
            tokenizer_path = self.model_dir / "tokenizer.json"
            self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
            self.use_fast = True
        except ImportError:
            # Fallback: will implement manual loading if needed
            raise ImportError(
                "Please install tokenizers: pip install tokenizers\n"
                "This provides fast tokenization for the model."
            )

        # Load special tokens
        special_tokens_path = self.model_dir / "special_tokens_map.json"
        with open(special_tokens_path, 'r') as f:
            self.special_tokens = json.load(f)

        # Load tokenizer config
        config_path = self.model_dir / "tokenizer_config.json"
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Get special token IDs
        self.bos_token = self.special_tokens.get("bos_token", "<s>")
        self.eos_token = self.special_tokens.get("eos_token", "</s>")
        self.pad_token = self.special_tokens.get("pad_token", "<pad>")
        self.unk_token = self.special_tokens.get("unk_token", "<unk>")

        # Encode special tokens to get their IDs
        self.bos_token_id = self.encode(self.bos_token, add_special_tokens=False)[0]
        self.eos_token_id = self.encode(self.eos_token, add_special_tokens=False)[0]
        self.pad_token_id = self.encode(self.pad_token, add_special_tokens=False)[0]

        # Vocabulary size
        self.vocab_size = self.tokenizer.get_vocab_size()

        print(f"✓ Tokenizer loaded:")
        print(f"  Vocabulary size: {self.vocab_size:,}")
        print(f"  BOS token: '{self.bos_token}' (ID: {self.bos_token_id})")
        print(f"  EOS token: '{self.eos_token}' (ID: {self.eos_token_id})")
        print(f"  PAD token: '{self.pad_token}' (ID: {self.pad_token_id})")

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        return_numpy: bool = True
    ) -> Union[List[int], np.ndarray]:
        """
        Encode text to token IDs.

        Args:
            text: Input text
            add_special_tokens: Whether to add BOS/EOS tokens
            return_numpy: Return as numpy array (default) or list

        Returns:
            Token IDs as list or numpy array
        """
        encoding = self.tokenizer.encode(text, add_special_tokens=add_special_tokens)
        token_ids = encoding.ids

        if return_numpy:
            return np.array(token_ids, dtype=np.int64)
        return token_ids

    def decode(
        self,
        token_ids: Union[List[int], np.ndarray],
        skip_special_tokens: bool = True
    ) -> str:
        """
        Decode token IDs to text.

        Args:
            token_ids: Token IDs as list or numpy array
            skip_special_tokens: Whether to skip special tokens in output

        Returns:
            Decoded text string
        """
        if isinstance(token_ids, np.ndarray):
            token_ids = token_ids.tolist()

        text = self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        return text

    def encode_batch(
        self,
        texts: List[str],
        add_special_tokens: bool = True,
        return_numpy: bool = True
    ) -> List[Union[List[int], np.ndarray]]:
        """
        Encode multiple texts to token IDs.

        Args:
            texts: List of input texts
            add_special_tokens: Whether to add BOS/EOS tokens
            return_numpy: Return as numpy arrays (default) or lists

        Returns:
            List of token ID sequences
        """
        encodings = self.tokenizer.encode_batch(texts, add_special_tokens=add_special_tokens)
        token_ids_list = [enc.ids for enc in encodings]

        if return_numpy:
            return [np.array(ids, dtype=np.int64) for ids in token_ids_list]
        return token_ids_list

    def decode_batch(
        self,
        token_ids_list: List[Union[List[int], np.ndarray]],
        skip_special_tokens: bool = True
    ) -> List[str]:
        """
        Decode multiple token ID sequences to text.

        Args:
            token_ids_list: List of token ID sequences
            skip_special_tokens: Whether to skip special tokens in output

        Returns:
            List of decoded text strings
        """
        # Convert numpy arrays to lists if needed
        ids_lists = []
        for ids in token_ids_list:
            if isinstance(ids, np.ndarray):
                ids = ids.tolist()
            ids_lists.append(ids)

        texts = self.tokenizer.decode_batch(ids_lists, skip_special_tokens=skip_special_tokens)
        return texts

    def __len__(self) -> int:
        """Return vocabulary size."""
        return self.vocab_size


# Example usage
if __name__ == "__main__":
    import sys
    import io

    # Fix Windows encoding
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("=" * 70)
    print("BitNet Tokenizer Test")
    print("=" * 70)

    model_dir = "C:/Users/samho/Desktop/BitNet-2B-model"

    # Load tokenizer
    tokenizer = BitNetTokenizer(model_dir)

    # Test encoding
    test_text = "Hello, world! How are you today?"
    print(f"\nTest text: '{test_text}'")

    tokens = tokenizer.encode(test_text)
    print(f"Tokens: {tokens}")
    print(f"Token count: {len(tokens)}")

    # Test decoding
    decoded = tokenizer.decode(tokens)
    print(f"Decoded: '{decoded}'")

    # Verify round-trip
    if decoded.strip() == test_text:
        print("✓ Round-trip successful!")
    else:
        print("⚠ Round-trip mismatch")

    print("\n" + "=" * 70)
