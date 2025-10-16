# Understanding SM (Streaming Multiprocessor)

## What is SM?

**SM = Streaming Multiprocessor** - NVIDIA's term for the **compute units** inside a GPU.

Think of SMs as the "CPU cores" of a GPU, but optimized for parallel processing.

---

## GPU Architecture Analogy

### CPU vs GPU Structure

**CPU** (Intel i9-14900K):
```
CPU
├─ 24 cores (8 P-cores + 16 E-cores)
├─ Each core runs 1-2 threads
└─ Total: ~32 threads in parallel
```

**GPU** (RTX 3080 Ti):
```
GPU
├─ 80 SMs (Streaming Multiprocessors)
├─ Each SM has 128 CUDA cores
├─ Each SM runs many threads concurrently
└─ Total: 10,240 CUDA cores, thousands of threads!
```

---

## What's Inside an SM?

Each Streaming Multiprocessor contains:

```
SM (Streaming Multiprocessor)
├─ 128 CUDA Cores (FP32 compute)
├─ 4 Tensor Cores (FP16/INT8 accelerated compute)
├─ 64 KB Shared Memory (fast L1 cache)
├─ Register File (thread-local memory)
├─ Warp Schedulers (manage thread execution)
└─ Load/Store Units (memory access)
```

**Warp**: Group of 32 threads that execute together

---

## SM Utilization - What Does It Mean?

### **High SM Utilization (85-95%)**
```
SMs are busy computing
├─ CUDA cores actively processing
├─ Tensor cores working (if using FP16)
├─ Threads executing instructions
└─ Status: COMPUTE-BOUND (good!)
```

**Meaning**: GPU cores are the bottleneck - they're maxed out doing calculations.

---

### **Low SM Utilization (60-65%)** ← This was our problem!
```
SMs are waiting/idle
├─ CUDA cores ready but no work
├─ Waiting for data from memory
├─ Threads stalled on memory loads
└─ Status: BANDWIDTH-BOUND (bad!)
```

**Meaning**: GPU cores are **starving for data** - memory can't feed them fast enough!

---

## The Diagnostic Pattern (From Our Profiling)

### What We Saw:

```
GPU busy: 80%+         ← GPU is powered on and active
SM utilization: 60-65% ← But compute cores are idle!
Memory util: 75-85%    ← Memory bandwidth maxed out
```

### What This Means:

```
┌─────────────────────────────────────┐
│  GPU is ON and trying to work       │
│  ↓                                  │
│  But SMs are waiting for data       │
│  ↓                                  │
│  Memory bandwidth is saturated      │
│  ↓                                  │
│  = BANDWIDTH-BOUND BOTTLENECK       │
└─────────────────────────────────────┘
```

---

## Visual Example

### Bandwidth-Bound (Our Problem):
```
Memory Bus: [████████████████████] 100% busy (saturated!)
            ↓ slow data delivery ↓
SM 1:       [████░░░░░░] 60% busy (waiting for data...)
SM 2:       [███░░░░░░░] 55% busy (waiting for data...)
SM 3:       [█████░░░░░] 65% busy (waiting for data...)
...
SM 80:      [████░░░░░░] 60% busy (waiting for data...)

Problem: Memory can't feed the SMs fast enough!
```

### Compute-Bound (Ideal):
```
Memory Bus: [████████░░] 70% busy (keeping up)
            ↓ fast data delivery ↓
SM 1:       [██████████] 95% busy (computing hard!)
SM 2:       [█████████░] 92% busy (computing hard!)
SM 3:       [██████████] 96% busy (computing hard!)
...
SM 80:      [█████████░] 93% busy (computing hard!)

Good: SMs are maxed out, memory keeping up!
```

---

## Why This Matters for Our Optimization

### The Diagnosis:

1. **GPU busy (80%+)** = GPU is powered on, scheduler active
2. **SM utilization low (60-65%)** = Compute cores idle
3. **Memory utilization high (75-85%)** = Memory bandwidth saturated

**Conclusion**: Memory bandwidth is the bottleneck, not compute power!

---

### Wrong Solution (What We Almost Did):
```
❌ "Let's get a GPU with more SMs!"
   → More compute cores won't help
   → They'll just be idle waiting for data too
   → Waste of money!

❌ "Let's increase batch size!"
   → Can't (architecturally limited to 8)
   → Would make bandwidth problem worse anyway
```

### Right Solution (What We Did):
```
✅ Reduce memory traffic (FP32 → FP16)
   → 2× less data to move
   → Memory bandwidth now sufficient
   → SMs get fed faster

✅ Optimize data pipeline
   → GPU never starves waiting for CPU
   → Pre-load data, async transfers
   → SMs always have work ready

Result: SM utilization 60% → 90%! 🚀
```

---

## How to Check SM Utilization

### Method 1: nvidia-smi (Real-time)
```bash
nvidia-smi dmon -s pucm
# p = power
# u = utilization (this is SM utilization!)
# c = clock speed
# m = memory usage
```

Output:
```
# gpu   pwr gtemp mtemp    sm   mem   enc   dec  mclk  pclk
# Idx     W     C     C     %     %     %     %   MHz   MHz
    0    45    65     -    62    78     0     0  9501  1830
         ↑ SM util    ↑ Memory util
```

### Method 2: PyTorch Profiler
```python
import torch.profiler as profiler

with profiler.profile(
    activities=[profiler.ProfilerActivity.GPU],
    with_stack=True
) as prof:
    # Training code here
    
print(prof.key_averages().table(sort_by="self_cuda_time_total"))
```

Shows which operations have low GPU efficiency (indicates bandwidth issues).

### Method 3: NSight Systems (Advanced)
```bash
nsys profile --stats=true python train.py
```

Detailed CUDA kernel analysis, SM occupancy, and memory transfer visualization.

---

## SM Specifications by GPU

### RTX 3080 Ti (Ampere)
- **80 SMs**
- **128 CUDA cores per SM** = 10,240 total
- **4 Tensor cores per SM** = 320 total
- **FP32 Performance**: 34 TFLOPS
- **FP16 (Tensor Core)**: 68 TFLOPS

### NVIDIA L40 (Ada Lovelace)
- **142 SMs**
- **128 CUDA cores per SM** = 18,176 total
- **4 Tensor cores per SM** = 568 total
- **FP32 Performance**: 90 TFLOPS
- **FP16 (Tensor Core)**: 180 TFLOPS

### RTX 4090 (Ada Lovelace)
- **128 SMs**
- **128 CUDA cores per SM** = 16,384 total
- **4 Tensor cores per SM** = 512 total
- **FP32 Performance**: 82 TFLOPS
- **FP16 (Tensor Core)**: 165 TFLOPS

---

## Common Bottleneck Patterns

### Pattern 1: Bandwidth-Bound (Our Case)
```
GPU: 80%+ busy
SM:  60-65% utilization  ← LOW
Mem: 75-85% utilization  ← HIGH

Diagnosis: Memory bandwidth saturated
Solution:  Reduce memory traffic (FP16, fused ops)
```

### Pattern 2: Compute-Bound
```
GPU: 95%+ busy
SM:  90-95% utilization  ← HIGH
Mem: 60-70% utilization  ← MODERATE

Diagnosis: Compute is bottleneck (good problem to have!)
Solution:  Better algorithm, quantization, or faster GPU
```

### Pattern 3: Data Pipeline Bottleneck
```
GPU: 40-60% busy (spiky)
SM:  30-50% utilization  ← LOW
Mem: 20-40% utilization  ← LOW
CPU: 100% utilization    ← MAXED

Diagnosis: CPU can't feed GPU fast enough
Solution:  Optimize DataLoader (num_workers, pin_memory)
```

### Pattern 4: VRAM-Limited
```
Training crashes with:
RuntimeError: CUDA out of memory

Diagnosis: Literally ran out of VRAM
Solution:  Reduce batch size, use gradient checkpointing, or FP16
```

---

## Key Takeaways

1. **SM = Streaming Multiprocessor** = The compute units in NVIDIA GPUs
2. **Each SM** contains CUDA cores, tensor cores, shared memory, and schedulers
3. **SM Utilization** = Percentage of time compute cores are actively processing
4. **Low SM + High Memory** = Bandwidth-bound (cores starving for data) ← **Our problem**
5. **High SM + High Memory** = Compute-bound (cores maxed out)
6. **Low SM + Low Memory + High CPU** = Data pipeline bottleneck ← **Also our problem**

---

## Our Project Results

### Before Optimization:
- **SM Utilization**: 60-65% (cores idle, waiting for data)
- **Memory Utilization**: 75-85% (bandwidth saturated)
- **CPU Utilization**: 100% (data pipeline bottleneck)
- **Training Speed**: 4 days per model

### After Optimization:
- **SM Utilization**: 88-92% (cores busy computing!)
- **Memory Utilization**: 75% (sufficient, not saturated)
- **CPU Utilization**: 30-40% (data pipeline fixed)
- **Training Speed**: 1.4 days per model

**Result**: Fixed bandwidth + data pipeline → SMs finally working efficiently → **2.8× speedup!** 🎯

---

## Further Reading

- [NVIDIA CUDA Programming Guide - SM Architecture](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities)
- [Understanding GPU Architecture](https://developer.nvidia.com/blog/cuda-refresher-reviewing-the-origins-of-gpu-computing/)
- [Profiling PyTorch Models](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- Our analysis: `THA4_GPU_Architecture_Analysis.md`
