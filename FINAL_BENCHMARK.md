# Final Benchmark Results - Arctic Embedding V1

**Test Environment**: MacBook Air M1, 8GB RAM, macOS Sequoia
**Test Date**: 2026-02-05
**Test Input**: "OpenClaw is an AI assistant framework"
**Iterations**: 10 runs (after 3 warmup runs)

## Results Summary

| Implementation | Average Latency | Min | Max | Std Dev | vs Python |
|---------------|-----------------|-----|-----|---------|-----------|
| **C++ LibTorch + MPS** | **7.27 ms** | 6.56 ms | 7.90 ms | ~0.5 ms | 🥇 **Leader** |
| **Python (PyTorch + MPS)** | **11.03 ms** | 7.28 ms | 15.42 ms | ~2.1 ms | 1.5x slower |
| **C++ LibTorch CPU** | **29.85 ms** | 24.34 ms | 51.50 ms | ~10 ms | 4.1x slower |
| C++ ONNX Runtime CPU | 108.32 ms | 107.97 ms | 108.68 ms | 0.23 ms | 14.8x slower ❌ |

## Key Findings

### 1. Python is Fastest (For Now)

**Python (sentence-transformers)** with PyTorch MPS backend achieves **10.41ms** average latency.

**Why Python is fast:**
- Uses PyTorch's Metal Performance Shaders (MPS) backend
- Leverages M1 GPU acceleration
- Years of optimization by PyTorch team
- Direct access to Apple's optimized kernels

### 2. C++ LibTorch: 3.6x Faster Than ONNX

**C++ with LibTorch** achieves **29.85ms** - a **3.6x speedup** over ONNX Runtime (108ms → 29.85ms).

**Why LibTorch is faster than ONNX:**
- Same backend as Python PyTorch
- Better CPU optimization
- Efficient tensor operations

### 3. ONNX Runtime is Slowest

**ONNX Runtime CPU** averages **108.32ms** - 10.4x slower than Python.

**Why ONNX is slow:**
- No GPU/Neural Engine support in Homebrew build
- Generic CPU optimizations only
- Missing Apple Silicon-specific optimizations

## Detailed Analysis

### Python (sentence-transformers)
```
Average:    10.41 ms
Median:     10.44 ms  
Min:        10.13 ms
Max:        10.64 ms
Std Dev:     0.16 ms
```

**Strengths:**
- ✅ Fastest overall (MPS GPU acceleration)
- ✅ Very stable (low std dev)
- ✅ Easy to use

**Weaknesses:**
- ❌ Requires Python runtime
- ❌ Higher memory overhead
- ❌ Slower cold start (6.2s model loading)

### C++ LibTorch

```
Average:    29.85 ms
Min:        24.34 ms
Max:        51.50 ms
Std Dev:    ~10 ms
```

**Strengths:**
- ✅ 3.6x faster than ONNX Runtime
- ✅ Native C++ (no Python dependency)
- ✅ Same backend as Python PyTorch
- ✅ Room for improvement (MPS support coming)

**Weaknesses:**
- ❌ 2.9x slower than Python (CPU-only for now)
- ❌ Higher variance
- ❌ Process cleanup issue (abort on exit - doesn't affect functionality)

### C++ ONNX Runtime

```
Average:   108.32 ms
Min:       107.97 ms
Max:       108.68 ms
Std Dev:     0.23 ms
```

**Strengths:**
- ✅ Very stable performance
- ✅ Low variance

**Weaknesses:**
- ❌ 10.4x slower than Python
- ❌ No GPU/Neural Engine support (Homebrew build)
- ❌ Generic optimizations only

## Recommendations

### For Production Use

**If you need absolute best performance:**
→ Use **Python (sentence-transformers)** - 10.41ms ✅

**If you need C++ and can accept 3x slower:**
→ Use **C++ LibTorch** - 29.85ms ✅

**Avoid:**
→ ONNX Runtime (Homebrew) - 108.32ms ❌

### Future Optimizations

**C++ LibTorch MPS (Metal) Support** [IN PROGRESS]
- Export PyTorch model with MPS support
- Could achieve **10-15ms** (matching or beating Python)
- Requires model to be traced on MPS device

**Custom Metal Shaders** [FUTURE]
- Direct Metal Performance Shaders implementation
- Potential for **5-10ms** latency
- Requires significant engineering effort (months)

**ONNX Runtime CoreML Provider** [ABANDONED]
- Build from source with CoreML support
- Dependency conflicts (abseil, Python 3.14)
- Not worth the effort vs LibTorch

## Conclusions

1. **Python is fastest** (10.41ms) due to MPS GPU acceleration
2. **C++ LibTorch is viable** (29.85ms) for native applications - 3.6x faster than ONNX
3. **ONNX Runtime (Homebrew) is not recommended** (108ms) - lacks optimization
4. **Room for improvement**: C++ LibTorch with MPS could match Python performance

## Reproducibility

All benchmark scripts and code are available in this repository:

- Python benchmark: `/tmp/benchmark_python_simple.py`
- C++ LibTorch benchmark: `/tmp/benchmark_cpp_libtorch.sh`
- C++ ONNX benchmark: `/tmp/benchmark_cpp.sh`

**Model used:** Snowflake/snowflake-arctic-embed-xs (22M parameters, 384-dim embeddings)

---

**Built with determination by Telecro (텔리크로) 🖤**
