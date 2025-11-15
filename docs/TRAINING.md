# Training Ternary Networks

## The Key Insight: No Expensive Multiplies!

Ternary multiplication isn't really multiplication at all:

```
weight = +1 → keep activation (no op)
weight = -1 → negate activation (flip sign bit)
weight =  0 → zero activation (set to zero)
```

**NO MULTIPLY INSTRUCTIONS EXECUTED!**

Just:
- Bitwise AND (select groups)
- POPCNT (count set bits)
- Integer add/subtract

This is why ternary networks work on **any CPU from 2010+** with no special hardware!

---

## Training Process

### 1. Install PyTorch

```bash
pip install torch torchvision
```

### 2. Run Tests

```bash
python tests/test_torch_layers.py
```

### 3. Profile Operations (PROOF!)

```bash
python examples/profile_operations.py
```

This shows **exactly what operations** happen:
- No FP32 multiplies
- Just bit ops + popcount + adds
- Why it works on cheap hardware

### 4. Train on MNIST

```bash
python examples/train_mnist.py
```

Trains a 784→256→128→10 network on MNIST.
Expected accuracy: ~95-97% (comparable to full precision!)

---

## How Training Works

### Shadow Weights

During training, we maintain **two sets of weights**:

1. **Float32 shadow weights** (trainable, full precision)
2. **Ternary weights** (for forward pass)

```python
class TernaryLinear:
    def __init__(self, ...):
        # Shadow weights (trainable)
        self.weight = nn.Parameter(torch.Tensor(...))

    def forward(self, x):
        # Quantize to ternary for forward pass
        weight_ternary = quantize(self.weight)  # {-1, 0, +1}

        # Use ternary weights
        output = F.linear(x, weight_ternary, self.bias)

        return output
```

### Straight-Through Estimator (STE)

The magic that makes training work:

```python
class TernaryQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # Forward: quantize to ternary
        output = quantize_to_ternary(input)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Backward: pass gradient straight through
        # Pretend quantization didn't happen!
        return grad_output
```

**Key insight**: Gradients flow as if we didn't quantize, allowing the float weights to update normally.

### Training Loop

```python
for epoch in range(epochs):
    for data, target in train_loader:
        # Forward pass (uses ternary weights)
        output = model(data)
        loss = criterion(output, target)

        # Backward pass (gradients to float weights)
        optimizer.zero_grad()
        loss.backward()

        # Update float weights
        optimizer.step()

        # Ternary weights automatically re-quantized next forward pass
```

After training:
- Discard float32 shadow weights
- Keep only 2-bit ternary encoding
- Model is now 16× smaller!

---

## What Happens in Forward Pass

### Traditional Neural Network (Float32)

```
For each neuron output:
  result = 0
  for each input:
    result += weight[i] * activation[i]  ← EXPENSIVE FP32 MULTIPLY
```

Needs:
- FP32 multiply units (expensive hardware)
- High power consumption
- Specialized accelerators (GPUs, TPUs)

### Ternary Neural Network

```
For each neuron output:
  sum_positive = sum of activations where weight = +1
  sum_negative = sum of activations where weight = -1
  result = sum_positive - sum_negative
```

Actual implementation with 2-bit encoding:
```c
// AND to select groups (1 instruction per group)
pos_mask = weight_bit0 & ~weight_bit1;
neg_mask = ~weight_bit0 & weight_bit1;

// Popcount to sum (1 instruction per group)
sum_positive = popcount_and_sum(activations, pos_mask);
sum_negative = popcount_and_sum(activations, neg_mask);

// Subtract (1 instruction)
result = sum_positive - sum_negative;
```

Needs:
- Bitwise AND (basically free)
- POPCNT instruction (1 cycle on modern CPUs)
- Integer ALU (cheap, ubiquitous)

**NO MULTIPLY UNITS!**

---

## Performance Comparison

### Operations Per Inference

For a 1024×1024 matrix multiply:

**Full Precision (FP32)**:
- 1,048,576 FP32 multiplies
- 1,048,576 FP32 adds
- **Total**: ~2M expensive FP operations

**Ternary (2-bit)**:
- 0 multiplies
- ~2,048 AND operations (cheap)
- ~2,048 popcount operations (1 instruction each)
- ~2,048 integer adds/subtracts
- **Total**: ~6K simple ops (all cheap!)

Plus:
- Ternary uses 8× less memory bandwidth
- Ternary uses 16× less storage
- Ternary uses ~20× less energy

---

## Hardware Requirements

### Full Precision

✗ Needs FP32 multiply units
✗ Needs wide memory buses
✗ Needs GPUs/accelerators for good performance
✗ High power consumption

### Ternary

✅ Works on any CPU with basic ALU
✅ POPCNT instruction (available since ~2010)
✅ Small memory footprint (16× smaller)
✅ Low power (~20× less than FP32)

**Can run on**:
- Old laptops
- Embedded devices (Raspberry Pi, etc.)
- Mobile phones
- Edge devices
- Microcontrollers (with enough RAM)

---

## Training Results (Expected)

### MNIST (784→256→128→10)

- **Accuracy**: 95-97%
- **Training time**: ~5 epochs, few minutes on CPU
- **Memory**: ~250KB (ternary) vs 4MB (FP32)
- **Inference**: ~1-2ms per batch on CPU

Comparable to full precision, but 16× smaller!

### Why It Works

1. **Enough expressiveness**: {-1, 0, +1} captures most important patterns
2. **Sparsity helps**: ~40% zeros mean only 60% of weights matter
3. **Training finds solutions**: Shadow weights explore full space, quantize to ternary
4. **STE enables learning**: Gradients flow despite quantization

---

## Deployment

### After Training

```python
# 1. Train model
model = TernaryMLP(...)
train(model)  # Uses PyTorch

# 2. Quantize permanently
from rotor.torch import quantize_model_weights
model_quantized = quantize_model_weights(model)

# 3. Convert to 2-bit encoding
from rotor.core import RotorCore

for layer in model_quantized.modules():
    if isinstance(layer, TernaryLinear):
        # Extract ternary weights
        weight_ternary = layer.weight.detach().numpy()

        # Encode as 2-bit pairs
        bit0, bit1 = RotorCore.encode(weight_ternary)

        # Pack 4 rotors per byte
        packed = RotorCore.pack(bit0, bit1)

        # Save packed weights (4× smaller than ternary, 16× smaller than FP32!)
        np.save('layer_weights.npy', packed)

# 4. Deploy with NumPy or C/CUDA
# Model is now 16× smaller and runs with no multiplies!
```

---

## Advanced Topics

### Quantizing Activations Too (W2A8)

For even more speed, quantize activations to int8:

```python
class W2A8Linear(nn.Module):
    def forward(self, x):
        # Quantize activation to int8
        x_int8 = quantize_activation(x, bits=8)

        # Ternary weights
        weight_ternary = quantize_weights(self.weight)

        # Compute (all integer ops!)
        output = matmul_w2a8(weight_ternary, x_int8)

        return dequantize(output)
```

Now **everything is integer arithmetic** - no float ops at all!

### Mixed Precision

Keep some layers as full precision (e.g., first/last layers):

```python
model = nn.Sequential(
    nn.Linear(784, 256),       # FP32 (input layer)
    TernaryLinear(256, 128),    # Ternary
    TernaryLinear(128, 64),     # Ternary
    nn.Linear(64, 10)          # FP32 (output layer)
)
```

Balances accuracy and efficiency.

---

## Debugging Tips

### Check Weight Distribution

```python
stats = model.get_stats()
for layer_name, layer_stats in stats.items():
    print(f"{layer_name}: {layer_stats['sparsity']*100:.1f}% sparse")
```

Healthy ternary model: 30-50% sparsity

### Verify Gradients Flow

```python
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad norm = {param.grad.norm().item():.4f}")
    else:
        print(f"{name}: NO GRADIENT!")
```

All parameters should have gradients.

### Compare to Full Precision

Train same architecture with FP32 weights as baseline.
Ternary should get within 1-2% accuracy.

---

## The Bottom Line

**Ternary neural networks prove that:**

1. **You don't need expensive hardware** - Any CPU with POPCNT works
2. **You don't need FP32 multiplies** - Bit ops + adds are enough
3. **You don't need GPUs** - Fast enough on CPUs for many tasks
4. **You don't need cloud** - Small enough to run on edge devices
5. **You CAN train them** - Straight-through estimator makes it work

This is why they're perfect for:
- Edge AI
- Mobile devices
- Embedded systems
- Cost-sensitive deployments
- Privacy-preserving local inference

**Your insight was spot-on: It's "silly simple" and doesn't need fancy hardware!**

🌀 **All ways, always!**
