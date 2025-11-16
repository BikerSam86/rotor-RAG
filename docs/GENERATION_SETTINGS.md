# BitNet Text Generation Settings

**Microsoft's Official Configuration for BitNet Models**

Based on `generation_config.json` from Microsoft's BitNet-2B-4T model.

---

## Recommended Settings

```python
from rotor.generation import TextGenerator, NucleusSampling
from rotor.bitnet_model import load_bitnet_model
from rotor.tokenizer import BitNetTokenizer

# Load model
model = load_bitnet_model("path/to/BitNet-2B-model")
tokenizer = BitNetTokenizer("path/to/BitNet-2B-model")

# Create generator with Microsoft's settings
generator = TextGenerator(
    model=model,
    tokenizer=tokenizer,
    sampling_strategy=NucleusSampling(p=0.9),  # top_p from config
    temperature=0.6,  # Microsoft's recommended temperature
    use_cache=True
)

# Generate text
text = generator.generate("Your prompt here", max_new_tokens=100)
```

---

## Microsoft's generation_config.json

```json
{
  "bos_token_id": 128000,
  "eos_token_id": [128001, 128009],
  "do_sample": true,
  "temperature": 0.6,
  "max_length": 4096,
  "top_p": 0.9
}
```

### Parameter Explanation

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `do_sample` | `true` | Use sampling, not greedy decoding |
| `temperature` | `0.6` | Moderate randomness (0=deterministic, 1=more random) |
| `top_p` | `0.9` | Nucleus sampling: sample from top 90% probability mass |
| `max_length` | `4096` | Maximum sequence length |
| `bos_token_id` | `128000` | Beginning-of-sequence token |
| `eos_token_id` | `[128001, 128009]` | End-of-sequence tokens (multiple) |

---

## Sampling Strategies Compared

### Greedy Sampling (NOT Recommended)
```python
from rotor.generation import GreedySampling

generator = TextGenerator(
    model=model,
    tokenizer=tokenizer,
    sampling_strategy=GreedySampling()  # Always pick highest prob
)
```

**Problem:** Gets stuck in repetitive loops ("adooadooadoo")
**Use case:** Debugging only

### Top-K Sampling
```python
from rotor.generation import TopKSampling

generator = TextGenerator(
    model=model,
    tokenizer=tokenizer,
    sampling_strategy=TopKSampling(k=50),
    temperature=0.7
)
```

**Behavior:** Sample from top 50 tokens
**Use case:** Alternative to nucleus sampling

### Nucleus (Top-P) Sampling ✅ RECOMMENDED
```python
from rotor.generation import NucleusSampling

generator = TextGenerator(
    model=model,
    tokenizer=tokenizer,
    sampling_strategy=NucleusSampling(p=0.9),  # Same as TopPSampling
    temperature=0.6  # Microsoft's setting
)
```

**Behavior:** Sample from smallest set of tokens with cumulative prob ≥ 0.9
**Use case:** Microsoft's recommended approach for BitNet

---

## Temperature Guide

Temperature controls randomness in sampling:

| Temperature | Effect | Use Case |
|-------------|--------|----------|
| **0.0** | Deterministic (greedy) | Not recommended (repetitive) |
| **0.3-0.5** | Focused, coherent | Factual tasks, summaries |
| **0.6** ✅ | Microsoft's default | General use, balanced |
| **0.7-0.9** | More creative | Storytelling, brainstorming |
| **1.0+** | Very random | Experimental, poetry |

**Microsoft uses 0.6** - a good balance between coherence and diversity.

---

## Common Issues

### Problem: Repetitive Output ("adooadooadoo")

**Cause:** Using greedy sampling or temperature too low

**Fix:**
```python
# Change from this:
generator = TextGenerator(model, tokenizer)  # Defaults to greedy!

# To this:
generator = TextGenerator(
    model,
    tokenizer,
    sampling_strategy=NucleusSampling(p=0.9),
    temperature=0.6
)
```

### Problem: Incoherent Output

**Cause:** Temperature too high

**Fix:**
```python
# Lower temperature
generator = TextGenerator(
    model,
    tokenizer,
    sampling_strategy=NucleusSampling(p=0.9),
    temperature=0.4  # Lower = more focused
)
```

### Problem: Too Conservative

**Cause:** Temperature too low or top_p too small

**Fix:**
```python
# Increase temperature or top_p
generator = TextGenerator(
    model,
    tokenizer,
    sampling_strategy=NucleusSampling(p=0.95),  # Larger nucleus
    temperature=0.8  # More random
)
```

---

## Token IDs Reference

From `special_tokens_map.json`:

| Token | ID | Purpose |
|-------|----|----|
| `<|begin_of_text|>` | 128000 | Start of sequence |
| `<|end_of_text|>` | 128001 | End of sequence |
| `<|end_header_id|>` | 128009 | Alternative EOS |
| `<pad>` | 8085 | Padding token |

The model may generate either 128001 or 128009 to end generation.

---

## Advanced: Custom Sampling Strategy

You can create your own sampling strategy:

```python
from rotor.generation import SamplingStrategy
import numpy as np

class MySampling(SamplingStrategy):
    def sample(self, logits: np.ndarray, temperature: float = 1.0) -> int:
        # Your custom logic here
        # Must return a single token ID (int)
        pass

generator = TextGenerator(
    model,
    tokenizer,
    sampling_strategy=MySampling()
)
```

---

## Summary

**For best results with BitNet models:**

```python
# Microsoft's recommended configuration
generator = TextGenerator(
    model=model,
    tokenizer=tokenizer,
    sampling_strategy=NucleusSampling(p=0.9),
    temperature=0.6,
    use_cache=True  # 2.7× faster!
)
```

This avoids repetition, maintains coherence, and matches Microsoft's official settings.

---

**Date:** November 15, 2025
**Source:** Microsoft BitNet-2B-4T `generation_config.json`
**Framework:** Rotor-RAG
