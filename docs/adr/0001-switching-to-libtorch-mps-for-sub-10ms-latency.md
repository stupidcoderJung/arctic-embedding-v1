# ADR 0001: Switching to LibTorch MPS for Sub-10ms Latency

## Status
Accepted

## Context
The initial goal was to provide a high-performance C++ embedding engine for Arctic-Embed-Tiny on Apple Silicon. We explored multiple backends to achieve the lowest possible latency.

### Evolution of Performance
1. **Python Baseline (sentence-transformers)**: ~10.41ms (PyTorch + MPS)
2. **C++ ONNX Runtime (CPU)**: ~108.32ms (Lack of CoreML/EP optimization in standard builds)
3. **C++ LibTorch (CPU)**: ~29.85ms (Better than ONNX, but slower than Python)
4. **C++ LibTorch (MPS - Final)**: **7.27ms** (Achieved via dedicated GPU acceleration)

## Decision
We decided to abandon ONNX Runtime in favor of **LibTorch (PyTorch C++ API)** with **Metal Performance Shaders (MPS)** acceleration.

### Key Technical Choices
1. **TorchScript Export on MPS**: The model was traced specifically on an MPS device to avoid device-mismatch errors in C++.
2. **Hybrid Execution**: Enabled `PYTORCH_ENABLE_MPS_FALLBACK=1` to handle ops not yet implemented in Metal while keeping the heavy matrix multiplications on the GPU.
3. **Direct C++ Implementation**: Eliminated Python interpreter overhead, resulting in a ~1.4x speedup over the Python baseline.

## Consequences
- **Pros**:
    - Achieved breakthrough latency of **7.27ms**, beating the 8ms target.
    - Seamless integration with PyTorch-based training workflows.
    - Lowest memory overhead compared to full Python environments.
- **Cons**:
    - Increased binary size due to LibTorch dependencies (~200MB).
    - Requires `libtorch` libraries to be present in the runtime environment.

## Validation
Results were validated through 1,000 controlled iterations on a MacBook Air M1 (8GB), ensuring thermal stability and consistent performance.
