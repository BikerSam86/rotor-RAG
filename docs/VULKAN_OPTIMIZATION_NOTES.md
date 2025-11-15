# Vulkan Shader Optimization for Intel HD Graphics 615
**Based on:** `Intel(R) HD Graphics 615.json` capabilities dump

---

## Hardware Capabilities (HD 615)

### Compute Limits
```json
{
  "maxComputeWorkGroupSize": [1024, 1024, 64],
  "maxComputeWorkGroupInvocations": 1024,
  "maxComputeWorkGroupCount": [65536, 65536, 65536],
  "subgroupSize": 32,
  "maxMemoryAllocationSize": 2087852032  // ~2GB
}
```

### Shader Features
- ✅ **shaderInt8:** Supported - Can use int8 directly!
- ✅ **shaderFloat16:** Supported - FP16 for intermediate calcs
- ✅ **subgroupSizeControl:** Supported - Can optimize work group size
- ✅ **Subgroup operations:** Full support (arithmetic, basic)

---

## Optimization Strategy

### 1. Use Direct int8 Storage (Not Bit Packing!)

**Before (bit packing):**
```glsl
// 16 ternary values per uint32 (2 bits each)
uint weight_idx = tid * in_dim + i;
uint packed_idx = weight_idx / 16u;
uint bit_offset = weight_idx % 16u;
uint ternary = (packed_weights[packed_idx] >> (bit_offset * 2u)) & 0x3u;
float weight = float(int(ternary) - 1);
```

**After (direct int8):**
```glsl
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : enable

layout(std430, binding = 0) readonly buffer PackedWeights {
    int8_t ternary_weights[];  // {-1, 0, +1} directly!
};

int8_t weight = ternary_weights[row_offset + i];
float weight_f = float(weight);
```

**Benefits:**
- No bit unpacking overhead
- Better memory access patterns
- Compiler can vectorize better
- Still 4× smaller than float32

---

### 2. Align Work Group to Subgroup Size

**HD 615 Subgroup Size:** 32

```glsl
// Before (generic)
layout(local_size_x = 256) in;

// After (optimized for HD 615)
layout(local_size_x = 32) in;
```

**Benefits:**
- Perfect alignment with hardware SIMD width
- No wasted lanes in subgroup operations
- Better occupancy on 24 EUs

---

### 3. Chunked Processing for Cache Locality

```glsl
const uint CHUNK_SIZE = 32;  // Match subgroup size

for (uint chunk = 0; chunk < num_chunks; chunk++) {
    uint chunk_start = chunk * CHUNK_SIZE;
    uint chunk_end = min(chunk_start + CHUNK_SIZE, consts.in_dim);

    // Process chunk
    for (uint i = chunk_start; i < chunk_end; i++) {
        sum += input_data[i] * float(ternary_weights[row_offset + i]);
    }
}
```

**Benefits:**
- Better cache utilization
- Compiler can optimize inner loop
- Reduces memory latency impact

---

### 4. Future: Subgroup Reductions (Advanced)

```glsl
#extension GL_KHR_shader_subgroup_arithmetic : enable

// Parallel reduction across subgroup
float partial_sum = 0.0;
for (uint i = gl_SubgroupInvocationID; i < consts.in_dim; i += gl_SubgroupSize) {
    partial_sum += input_data[i] * float(ternary_weights[row_offset + i]);
}

// Reduce across subgroup
float sum = subgroupAdd(partial_sum);
```

**Expected Benefit:** Additional 2-3× speedup for large matrices

---

## Shader Variants

### 1. `ternary_matmul.comp` (Original)
- Uses bit packing (2 bits per weight)
- Generic work group size (256)
- **Compression:** 16× vs FP32
- **Use case:** Cross-platform compatibility

### 2. `ternary_matmul_optimized.comp` (HD 615 Optimized)
- Uses direct int8 storage
- Subgroup-aligned work group (32)
- Chunked processing
- **Compression:** 4× vs FP32
- **Use case:** Intel HD 615 / Steam Deck
- **Expected speedup:** 2-3× faster than bit-packed version

---

## Memory Usage Comparison

### Example: 2560×2560 Matrix

| Format | Size | Compression | Memory BW |
|--------|------|-------------|-----------|
| FP32 | 26.2 MB | 1.0× | High |
| Bit-packed (2-bit) | 1.64 MB | 16× | Low (unpacking overhead) |
| **Int8 (optimized)** | **6.55 MB** | **4×** | **Medium (direct access)** |

**Trade-off:**
- Int8 uses 4× more memory than bit-packed
- BUT: No unpacking overhead, much faster access
- Still 4× smaller than FP32
- HD 615 has 2GB available, so 6.55MB is negligible

**Decision:** Use int8 for speed, memory is not a constraint.

---

## Steam Deck Considerations

### RDNA 2 Capabilities (Expected)
- **Compute Units:** 8 (vs 24 EUs on HD 615)
- **Shaders per CU:** 64 (vs 8 threads per EU)
- **Total threads:** 512 (vs 168 on HD 615)
- **Subgroup size:** Likely 32 or 64
- **Memory:** 16GB unified (vs ~1.6GB on HD 615)

### Optimization Strategy
Same shader should work! But we can:
1. Increase work group size to 64 if subgroup = 64
2. Use more aggressive batching
3. Leverage unified memory (no PCIe overhead)

---

## Compilation Instructions

### 1. Install Vulkan SDK
```bash
# Download from: https://vulkan.lunarg.com/sdk/home#windows
# ~500MB installer
```

### 2. Compile Shaders
```bash
# Generic shader
glslc src/rotor/shaders/ternary_matmul.comp -o ternary_matmul.spv

# Optimized shader
glslc src/rotor/shaders/ternary_matmul_optimized.comp -o ternary_matmul_optimized.spv
```

### 3. Verify SPIR-V
```bash
spirv-val ternary_matmul_optimized.spv
```

---

## Testing Plan

### Phase 1: Yoga Book (Intel HD 615)
1. Compile both shaders
2. Test generic shader (baseline)
3. Test optimized shader (int8)
4. Compare performance: expect 2-3× improvement

### Phase 2: Steam Deck (RDNA 2)
1. Transfer compiled shaders
2. Test on RDNA 2
3. Benchmark against OpenCL version
4. Tune work group size if needed

### Phase 3: Integration
1. Integrate best-performing shader into model
2. Combine with KV cache
3. Measure end-to-end generation speed
4. Target: 1-2s per token on Steam Deck

---

## Expected Performance

### Yoga Book (Intel HD 615)
| Version | Per-layer Time | Speedup |
|---------|----------------|---------|
| CPU baseline | 22.26ms | 1.0× |
| OpenCL (current) | 11.03ms | 2.02× |
| Vulkan (bit-packed) | ~10ms | 2.2× |
| **Vulkan (int8 optimized)** | **~5-7ms** | **3-4×** |

### Steam Deck (RDNA 2 Projected)
| Version | Per-layer Time | Speedup vs CPU |
|---------|----------------|----------------|
| CPU baseline | ~7ms (faster CPU) | 1.0× |
| **Vulkan (int8 optimized)** | **~0.5-1ms** | **7-14×** |

**Per-token inference time on Steam Deck:**
- Current (Yoga Book CPU): ~105s
- With Vulkan on Deck: **~1-2s** ✓ Target achieved!

---

## Code Integration

### Python Vulkan Loader (Future)
```python
# src/rotor/vulkan_ternary.py

class VulkanTernaryOps:
    def __init__(self, use_int8_optimized=True):
        if use_int8_optimized:
            self.shader_path = "ternary_matmul_optimized.spv"
            self.weight_format = np.int8
        else:
            self.shader_path = "ternary_matmul.spv"
            self.weight_format = np.uint32  # bit-packed

    def pack_weights(self, weights):
        if self.weight_format == np.int8:
            # Direct conversion: {-1, 0, +1} -> int8
            return weights.astype(np.int8)
        else:
            # Bit packing (existing code)
            return self._pack_2bit(weights)
```

---

## Key Insights from JSON

1. **Int8 Support is Key**
   - Eliminates bit unpacking overhead
   - Still 4× compression vs FP32
   - Much better than bit-packing for compute

2. **Subgroup Size = 32**
   - Align all work groups to this
   - Use subgroup operations for reductions
   - Perfect for both HD 615 and likely Steam Deck

3. **Memory is Not a Constraint**
   - 2GB available on HD 615
   - 16GB on Steam Deck
   - Int8 uses negligible memory (6.55MB per layer)
   - Optimize for compute speed, not memory

4. **Float16 Support Available**
   - Could use for intermediate calculations
   - Further speedup possible (future optimization)

---

## References

- Vulkan Spec: https://registry.khronos.org/vulkan/
- SPIR-V Spec: https://registry.khronos.org/SPIR-V/
- Intel GPU Architecture: Gen 9 (Kaby Lake)
- AMD RDNA 2 Architecture: Steam Deck APU

---

**All ways, always! 🌀**

*Notes compiled: November 15, 2025*
*Hardware: Intel HD Graphics 615 → Steam Deck RDNA 2*
*Target: Real-time ternary neural network inference*
