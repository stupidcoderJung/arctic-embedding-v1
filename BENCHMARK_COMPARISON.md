# Benchmark Comparison: Python vs C++ Implementations

**Test Environment**: MacBook Air M1, 8GB RAM, macOS
**Test Input**: "OpenClaw is an AI assistant framework"
**Iterations**: 10 runs (after 3 warmup runs)

## Current Results

| Implementation | Average Latency | Min | Max | Std Dev |
|---------------|-----------------|-----|-----|---------|
| **Python (sentence-transformers)** | **10.41 ms** | 10.13 ms | 10.64 ms | 0.16 ms |
| **C++ Original** | **108.32 ms** | 107.97 ms | 108.68 ms | ~0.23 ms |

## Key Finding

**Python is 10.4x faster than the original C++ implementation.**

This is a critical performance gap that needs to be addressed.

## Why is Python Faster?

The Python `sentence-transformers` library likely benefits from:

1. **Optimized ONNX Runtime configuration**
   - Better thread management
   - Optimized execution providers
   - Memory pattern optimization

2. **Apple Accelerate framework integration**
   - Leverages ARM NEON instructions
   - Efficient BLAS/LAPACK operations

3. **Buffer reuse and memory management**
   - Avoids repeated allocations
   - Smart tensor caching

4. **Years of optimization**
   - sentence-transformers is a mature, heavily optimized library
   - Benefits from community contributions and profiling

## Planned Optimizations (C++)

Based on expert analysis, the following optimizations will be applied:

1. ✅ **ONNX Runtime Session Options**
   - `SetIntraOpNumThreads(4)` → Utilize M1 performance cores
   - `EnableMemPattern()` → Reuse memory allocations
   - `SetGraphOptimizationLevel(ORT_ENABLE_ALL)` → Full optimization

2. ✅ **Apple Accelerate Framework**
   - Link against `-framework Accelerate`
   - Leverage optimized BLAS operations for ARM

3. ✅ **Buffer Reuse**
   - Allocate tensors once, reuse for subsequent inferences
   - Avoid repeated `std::vector` allocations

4. ✅ **Remove Debug Output**
   - Eliminate unnecessary I/O during inference
   - Move memory logging outside hot path

## Target Performance

Based on expert analysis and Python's performance:

- **Conservative target**: 35-50 ms (3-5x improvement)
- **Optimistic target**: 15-25 ms (approaching Python performance)
- **Stretch goal**: < 15 ms (matching or beating Python)

## Next Steps

1. ✅ Compile optimized C++ version with expert recommendations
2. ⏳ Run benchmark on optimized version
3. ⏳ Compare results and iterate if needed
4. ⏳ Update README with final performance metrics
5. ⏳ Push optimized version to GitHub

## Notes

- Python's model loading (6249 ms) is slower than expected - cold start optimization not critical for this use case
- C++ cold start not yet measured - will add in next iteration
- Both implementations produce correct 384-dimensional embeddings (verified)
