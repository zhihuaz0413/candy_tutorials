# Triton & CUDA Learning Guide

A practical guide covering GPU kernel programming with Triton and CUDA, compiled from hands-on tutorial walkthroughs.

---

## Table of Contents

1. [What is Triton?](#1-what-is-triton)
2. [Core Mental Model: Block-Based Programming](#2-core-mental-model-block-based-programming)
3. [Python Basics Used in Triton Code](#3-python-basics-used-in-triton-code)
4. [Section 1: Vector Addition — Your First Kernel](#4-section-1-vector-addition)
5. [Section 2: Fused Softmax — Why Triton Shines](#5-section-2-fused-softmax)
6. [Section 3: Matrix Multiplication — Deep Dive](#6-section-3-matrix-multiplication)
7. [Section 4: Autotuning](#7-section-4-autotuning)
8. [Section 5: Benchmarking](#8-section-5-benchmarking)
9. [Section 6: Fused LayerNorm](#9-section-6-fused-layernorm)
10. [When Should You Use Triton?](#10-when-should-you-use-triton)
11. [Triton vs CUDA Comparison](#11-triton-vs-cuda-comparison)
12. [Key Triton APIs Explained](#12-key-triton-apis-explained)

---

## 1. What is Triton?

**Triton** is a language and compiler by OpenAI for writing GPU kernels in Python. It lets you:

- Write GPU code that's as fast as hand-tuned CUDA (often within 90-95%)
- Use familiar Python syntax instead of C/C++
- Automatically handle many low-level GPU optimizations (memory coalescing, shared memory, etc.)

| | PyTorch | Triton | CUDA |
|---|---|---|---|
| **Ease** | Easiest | Medium | Hardest |
| **Control** | Least | Good | Full |
| **Performance** | Good | Great | Best |
| **Use case** | Standard ops | Custom fused ops | Maximum perf |

---

## 2. Core Mental Model: Block-Based Programming

The single most important concept in Triton is **block-based programming**:

- In **CUDA**, you write code for a single thread and launch millions of threads.
- In **Triton**, you write code for a **block/tile** of data and launch many program instances, each processing one block.

Every kernel follows the same pattern:

```python
@triton.jit
def my_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)                                    # 1. Which block am I?
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)     # 2. My element indices
    mask = offsets < n_elements                                # 3. Bounds check
    data = tl.load(input_ptr + offsets, mask=mask)             # 4. Load
    result = do_something(data)                                # 5. Compute
    tl.store(output_ptr + offsets, result, mask=mask)          # 6. Store
```

---

## 3. Python Basics Used in Triton Code

### Underscore in numbers (`98_432`)

The `_` in `98_432` is Python's numeric literal separator — purely visual, zero effect on the value:

```python
98_432 == 98432        # True — identical values
1_000_000              # one million — easy to count zeros
```

### `torch.manual_seed(0)`

Sets the random number generator seed to a fixed value so results are **reproducible**. Every run produces the exact same "random" numbers.

### `tensor.numel()`

Returns the **total number of elements** in a tensor ("number of elements"):

```python
torch.zeros(4, 5).numel()    # 20  (4 × 5)
torch.zeros(2, 3, 4).numel() # 24  (2 × 3 × 4)
```

### `tensor.stride(dim)`

Returns how many elements to skip in memory to move one step along a dimension:

```python
x = torch.randn(128, 512)
x.stride(0)  # 512 — skip 512 elements to get to the next row
x.stride(1)  # 1   — skip 1 element to get to the next column
```

Strides are passed to kernels so they work correctly even with transposed or non-contiguous tensors.

### `range(start, stop, step)`

Generates integers from `start` up to (but **not including**) `stop`, incrementing by `step`:

```python
range(10)          # [0, 1, 2, ..., 9]
range(0, 10, 2)    # [0, 2, 4, 6, 8]
range(0, 100, 10)  # [0, 10, 20, ..., 90]
```

---

## 4. Section 1: Vector Addition

The simplest kernel: `output[i] = x[i] + y[i]`

### The Kernel

```python
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)
```

Key elements:

1. **`@triton.jit`** — Decorator that JIT-compiles the function into GPU code
2. **`tl.program_id(axis=0)`** — Each program instance gets a unique ID (equivalent to `blockIdx.x` in CUDA)
3. **`tl.arange(0, BLOCK_SIZE)`** — Create a range of offsets within the block
4. **`tl.load()` / `tl.store()`** — Read from / write to GPU memory
5. **`mask`** — Prevents out-of-bounds memory access

### The Grid and Launch

```python
grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
```

- **`grid`** is a lambda that computes how many program instances to launch
- It's a lambda (not a plain tuple) so it works with autotuning — each `BLOCK_SIZE` needs a different grid size
- **`triton.cdiv`** = ceiling division: `ceil(98432 / 1024) = 97` programs
- **`kernel[grid](...)`** is Triton's launch syntax (analogous to CUDA's `<<<blocks, threads>>>`)

For 98,432 elements with BLOCK_SIZE=1024:

```
Program 0:  elements [0 .. 1023]
Program 1:  elements [1024 .. 2047]
...
Program 96: elements [98304 .. 98431]  ← only 128 valid (mask handles the rest)
```

---

## 5. Section 2: Fused Softmax

### Why Fusion Matters

In PyTorch, `softmax(x)` requires **multiple kernel launches** (max, subtract, exp, sum, divide). With Triton, we **fuse** these into a single kernel, eliminating intermediate memory reads/writes.

```
Unfused (PyTorch):  5+ kernel launches, 5+ memory round-trips
Fused (Triton):     1 kernel launch, 1 read + 1 write to global memory
```

### The Kernel

Each program handles one **row** of the input matrix:

1. Load the entire row into SRAM
2. Compute max (for numerical stability)
3. Subtract max and compute exp
4. Sum the exp values
5. Divide to get softmax
6. Store result

The `other=-float('inf')` in `tl.load` fills masked positions with -inf so they don't affect max/softmax.

---

## 6. Section 3: Matrix Multiplication

### The Big Picture

Computing `C = A × B` where A is `(M, K)`, B is `(K, N)`, C is `(M, N)`.

The output C is divided into **tiles** and one program instance computes each tile.

```
      A (512×128)            B (128×256)            C (512×256)
    ┌──────────────┐      ┌──────────────────┐    ┌──────────────────┐
    │              │      │                  │    │ tile  tile  tile │
    │              │  ×   │                  │ =  │(0,0) (0,1) (0,2)│
    │              │      │                  │    │ tile  tile  tile │
    │              │      │                  │    │(1,0) (1,1) (1,2)│
    └──────────────┘      └──────────────────┘    └──────────────────┘

Each tile of C is BLOCK_M × BLOCK_N (64 × 64)
Grid: 8 × 4 = 32 program instances
```

### Step-by-Step

**Step 1: Which tile am I?**

```python
pid_m = tl.program_id(0)   # row tile index (0..7)
pid_n = tl.program_id(1)   # col tile index (0..3)
```

This uses a **2D grid**. Program `(pid_m=2, pid_n=1)` computes rows 128-191, columns 64-127 of C.

**Step 2: Compute element indices**

```python
offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [64] row indices
offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [64] col indices
offs_k = tl.arange(0, BLOCK_K)                     # [32] k indices
```

**Step 3: Build 2D pointer grids using broadcasting**

```python
a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
```

`[:, None]` and `[None, :]` create 2D broadcasting:
- `offs_m[:, None]` has shape `[64, 1]` (column vector)
- `offs_k[None, :]` has shape `[1, 32]` (row vector)
- Result: `a_ptrs` has shape `[64, 32]` — points to a 64×32 tile of A

**Step 4: Accumulate over K dimension**

```python
acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)  # 64×64 accumulator

for k_start in range(0, K, BLOCK_K):    # k_start = 0, 32, 64, 96
    a = tl.load(a_ptrs, mask=a_mask, other=0.0)   # 64×32 tile of A
    b = tl.load(b_ptrs, mask=b_mask, other=0.0)   # 32×64 tile of B
    acc += tl.dot(a, b)                            # 64×32 @ 32×64 → 64×64
    a_ptrs += BLOCK_K * stride_ak                  # slide A tile right
    b_ptrs += BLOCK_K * stride_bk                  # slide B tile down
```

**Step 5: Store result**

```python
tl.store(c_ptrs, acc, mask=c_mask)
```

### Why Tiling is Fast

Without tiling: each element loaded once per use → total global loads = M × N × K × 2
With tiling (TILE=64): each load reused 64 times → **64× reduction** in memory traffic

### Floating-Point Precision Note

The Triton kernel and PyTorch compute the same math but in a **different order**, and floating-point addition is not associative: `(a + b) + c ≠ a + (b + c)`. Differences of ~0.01-0.05 are normal for fp32 matmul and do not indicate a bug.

---

## 7. Section 4: Autotuning

### The Problem

The best `BLOCK_SIZE` and `num_warps` depend on the GPU model, data size, and memory patterns. Instead of guessing, `@triton.autotune` **benchmarks multiple configs and picks the fastest**.

### How It Works

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64},  num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['n_elements'],  # re-tune when this value changes
)
@triton.jit
def relu_kernel(x_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    ...
```

- **`configs`** — list of configurations to try (BLOCK_SIZE + num_warps combinations)
- **`key`** — which arguments trigger re-benchmarking when their value changes
- **`num_warps`** — how many warps (groups of 32 threads) run per program instance

### Runtime Behavior

On the **first call**:

1. Compile all 5 kernel versions
2. Benchmark each version multiple times
3. Pick the fastest
4. Cache the result for the given `n_elements` value

On **subsequent calls** with the same key: use cached winner immediately.

### When to Use

- Performance-critical kernels in production → **Yes**
- Kernel runs on different GPUs or varying input sizes → **Yes**
- Learning / debugging → **No** (slows down iteration)

---

## 8. Section 5: Benchmarking

### The Setup

```python
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],                              # x-axis variable
        x_vals=[2**i for i in range(12, 25)],          # 4K to 16M elements
        line_arg='provider',                           # separate lines per provider
        line_vals=['triton', 'torch'],                 # two lines
        line_names=['Triton', 'PyTorch'],              # legend names
        styles=[('blue', '-'), ('green', '-')],        # line styles
        ylabel='GB/s',                                 # y-axis label
        plot_name='vector-add-performance',            # output filename
        args={},
    )
)
```

This runs `13 sizes × 2 providers = 26` benchmark measurements.

### The Timing Function

```python
def benchmark(size, provider):
    x = torch.rand(size, device='cuda', dtype=torch.float32)
    y = torch.rand(size, device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]  # median, 20th, 80th percentile

    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y, quantiles=quantiles)
    elif provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y), quantiles=quantiles)
```

`triton.testing.do_bench` properly benchmarks GPU operations by:
1. Warming up (fills caches, triggers JIT)
2. Running many iterations with `cuda.synchronize()`
3. Returning percentile timings

### The Throughput Calculation

```python
gbps = lambda ms: 3 * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)
```

```
Total bytes = 3 × num_elements × 4 bytes
              ↑                   ↑
              2 reads + 1 write   float32 = 4 bytes

GB/s = total_bytes / time_in_seconds
```

GB/s tells you how close the kernel is to saturating GPU memory bandwidth.

---

## 9. Section 6: Fused LayerNorm

### What is LayerNorm?

Used in every Transformer layer. For each row:

```
output = (x - mean) / sqrt(variance + eps) * weight + bias
```

### Why Fuse?

```
Unfused (PyTorch):    4 kernels → 4 reads + 4 writes to VRAM
Fused (Triton):       1 kernel  → 1 read  + 1 write  to VRAM
```

### The Kernel Walkthrough

**Load one row** (one program per row):

```python
row = tl.program_id(0)
x = tl.load(x_ptr + row * stride + cols, mask=mask, other=0.0).to(tl.float32)
```

**Compute mean and variance:**

```python
mean = tl.sum(x, axis=0) / N
x_centered = x - mean
var = tl.sum(x_centered * x_centered, axis=0) / N
rstd = 1.0 / tl.sqrt(var + eps)
```

**Normalize and apply affine transform:**

```python
x_hat = x_centered * rstd
output = x_hat * weight + bias
```

Everything happens in **fast on-chip SRAM** — no intermediate tensors written to VRAM.

### Performance Note

For the tutorial's small input `(256, 768)`, PyTorch's built-in LayerNorm may be faster because:

1. PyTorch uses heavily optimized cuDNN/CUDA kernels with warp-level shuffle intrinsics
2. Small data means kernel launch overhead dominates
3. The tutorial kernel doesn't use autotuning
4. First-call JIT compilation adds overhead

Triton wins when fusing LayerNorm **with adjacent operations** (e.g., LayerNorm + GELU + Dropout in one kernel).

---

## 10. When Should You Use Triton?

For **standard individual operations**, PyTorch will usually be faster because NVIDIA has spent years optimizing those kernels. Triton's power lies in **fusion**.

### The Core Insight

```
GPU time = compute time + memory transfer time

For most Transformer operations:
  compute time   ≈ 10%
  memory time    ≈ 90%    ← the bottleneck

Fusing 3 ops into 1:
  compute: same
  memory:  reduced by ~3x  ← this is where the speedup comes from
```

### Fusion Example

```
PyTorch (3 kernels, 3 memory round-trips):
  VRAM → [LayerNorm] → VRAM → [GELU] → VRAM → [Dropout] → VRAM

Triton (1 fused kernel, 1 memory round-trip):
  VRAM → [LayerNorm + GELU + Dropout] → VRAM
```

### Decision Table

| Situation | Use Triton? | Why |
|---|---|---|
| Standard single op (matmul, softmax) | No | PyTorch/cuBLAS already optimal |
| Fusing 2-5 memory-bound ops together | **Yes** | Triton's sweet spot |
| Custom activation function (SwiGLU, etc.) | **Yes** | No pre-built fused kernel |
| Custom loss function (fused cross-entropy) | **Yes** | Avoids large intermediate tensors |
| Quantized/mixed-precision kernels | **Yes** | Custom data formats |
| Research: trying new operations | **Yes** | Much faster to iterate than CUDA |

### Real-World Projects Using Triton

- **Liger Kernel**: Fused CrossEntropy → 3x faster, 60% less memory
- **Flash Attention**: Fused Q×K + softmax + ×V → 2-4x faster
- **Unsloth**: Fused RoPE + Attention → 2x faster fine-tuning

### Rule of Thumb

**Don't** use Triton to rewrite `torch.matmul` or `torch.softmax`.

**Do** use Triton when you have a sequence of element-wise or reduction operations that could share data in SRAM.

---

## 11. Triton vs CUDA Comparison

### Vector Addition Side-by-Side

**CUDA:**
```cpp
__global__ void add_kernel(float* x, float* y, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        out[idx] = x[idx] + y[idx];
}
// Launch: add_kernel<<<(n+255)/256, 256>>>(x, y, out, n);
```

**Triton:**
```python
@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(out_ptr + offs, tl.load(x_ptr + offs, mask=mask) +
             tl.load(y_ptr + offs, mask=mask), mask=mask)
```

### Concept Mapping

| CUDA | Triton |
|---|---|
| `threadIdx.x` — which thread in block | (handled by compiler) |
| `blockIdx.x` — which block | `tl.program_id(0)` |
| `blockDim.x` — threads per block | `BLOCK_SIZE` (constexpr) |
| `__shared__` memory | Automatic SRAM |
| `__syncthreads()` | Automatic synchronization |
| `<<<blocks, threads>>>` | `kernel[grid](...)` |
| Manual thread management | Automatic — you manage tiles |

### When to Use Which

**Use Triton when:**
- You want rapid iteration on fused kernels
- The operation maps naturally to block/tile processing

**Use CUDA when:**
- You need warp-level primitives (`__shfl_sync`)
- You need fine-grained shared memory control
- You need async memory copies (cp.async, TMA)
- Maximum performance is critical and you can invest the time

---

## 12. Key Triton APIs Explained

### `tl.program_id(axis)`

Returns the unique ID of the current program instance on the given axis. Different for each parallel instance, fixed within one instance.

```python
tl.program_id(0)  # equivalent to CUDA's blockIdx.x
tl.program_id(1)  # equivalent to CUDA's blockIdx.y
```

### `tl.load(pointer, mask, other)`

Reads data from GPU global memory into fast on-chip registers/SRAM.

- `pointer` — address(es) to read from
- `mask` — boolean; only loads where True
- `other` — fill value for masked (False) positions

### `tl.store(pointer, value, mask)`

Writes data from registers back to global memory.

- `pointer` — address(es) to write to
- `value` — data to write
- `mask` — only writes where True (prevents out-of-bounds corruption)

### `tl.arange(start, end)`

Creates a 1D range of integers `[start, start+1, ..., end-1]`, similar to Python's `range()`.

### `tl.dot(a, b)`

Hardware-accelerated tile matrix multiplication. Uses tensor cores when available.

### `tl.sum(x, axis)` / `tl.max(x, axis)`

Reduction operations across a tile. Triton handles the parallel reduction automatically (in CUDA, you'd need shared memory + `__syncthreads()`).

### `tl.constexpr`

Marks a parameter as a compile-time constant. The compiler can optimize aggressively around known values. All block sizes should be `tl.constexpr`.

### `triton.cdiv(a, b)`

Ceiling division: `ceil(a / b)`. Used to compute grid sizes.

### `triton.next_power_of_2(n)`

Returns the smallest power of 2 ≥ n. Block sizes must be powers of 2 for hardware efficiency.

---

## Resources

- [Official Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)
- [Triton GitHub](https://github.com/triton-lang/triton)
- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Liger Kernel](https://github.com/linkedin/Liger-Kernel) — production Triton kernels for LLM training
- [Unsloth](https://github.com/unslothai/unsloth) — optimized Triton kernels for fine-tuning
- [CUTLASS](https://github.com/NVIDIA/cutlass) — NVIDIA's production GEMM templates
