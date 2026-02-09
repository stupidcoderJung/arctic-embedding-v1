# Final Benchmark Results - Arctic Embedding V1

[🇰🇷 한국어 버전(Korean)](./FINAL_BENCHMARK_KR.md)

---

**Test Environment**: MacBook Air M1, 8GB RAM, macOS Sequoia
**Test Date**: 2026-02-09 (v2 — WordPiece tokenizer + OpenClaw plugin)
**Test Input**: "OpenClaw is an AI assistant framework"
**Iterations**: 1000 runs (after 50 warmup runs) - **Strict Variable Control Applied**

## Results Summary

| Implementation | Average Latency | Min | Max | Std Dev | Status |
|---------------|-----------------|-----|-----|---------|-----------|
| **C++ LibTorch + MPS (v2)** | **6.55 ms** | 5.92 ms | 7.21 ms | ~0.4 ms | 🥇 **Absolute Leader** |
| Python (PyTorch + MPS) | 11.03 ms | 7.28 ms | 15.42 ms | ~2.1 ms | ✅ Baseline |
| C++ LibTorch CPU | 29.85 ms | 24.34 ms | 51.50 ms | ~10 ms | ✅ Practical |
| C++ ONNX Runtime CPU | 108.32 ms | 107.97 ms | 108.68 ms | 0.23 ms | ❌ Not recommended |
| OpenAI API (network) | ~300 ms | ~200 ms | ~500 ms | — | ❌ Cloud dependency |

## Key Findings

### 1. C++ LibTorch MPS v2 — 6.55ms

By leveraging **LibTorch with Metal Performance Shaders (MPS)** and a fully native **WordPiece tokenizer** in C++, we achieved **6.55ms** latency. This is **1.7x faster than Python**, **16.5x faster than ONNX Runtime**, and **~46x faster than OpenAI API**.

**Why C++ LibTorch MPS is the winner:**
- **Zero Interpreter Overhead**: No Python GIL or runtime overhead.
- **Full GPU Acceleration**: Direct access to M1 GPU via MPS.
- **Native WordPiece Tokenizer**: BERT-compatible 30,522-token vocabulary in C++.
- **Optimized C++ Core**: Minimal memory management overhead.

### 2. v1 → v2 Improvement

| | v1 (2026-02-05) | v2 (2026-02-09) | Improvement |
|---|---|---|---|
| **Avg Latency** | 7.27 ms | **6.55 ms** | **-9.9%** |
| **Tokenizer** | Placeholder (hardcoded IDs) | Full WordPiece (30,522 vocab) | Production-ready |
| **Output** | Benchmark only | `--json` mode for plugin | Plugin-integrated |
| **Plugin** | Standalone binary | OpenClaw `memory-arctic` | Drop-in replacement |

### 3. Reliable Variable Control

Results obtained under **strict environmental control**:
- Thermal stabilization (30s idle before test).
- Memory purging (`sudo purge`) and bloatware termination.
- 1,000 iterations to eliminate statistical noise.

### 4. Comparison with Cloud APIs

| | Arctic V1 (Local) | OpenAI text-embedding-3-small |
|---|---|---|
| **Latency** | 6.55 ms | ~300 ms |
| **Cost** | $0 | $0.02/1M tokens |
| **Privacy** | 100% local | Cloud |
| **Offline** | Yes | No |
| **Dimensions** | 384 | 1536 |

## Detailed Analysis

### C++ LibTorch + MPS v2 (Production)
```
Average:     6.55 ms
Min:         5.92 ms
Max:         7.21 ms
Status:      🥇 Target Achieved (< 7ms)
```

**Strengths:**
- ✅ Fastest overall (M1 GPU acceleration)
- ✅ Native C++ implementation (no Python dependency)
- ✅ Full WordPiece tokenizer (BERT-compatible)
- ✅ Sub-7ms latency for real-time RAG applications
- ✅ `--json` output for OpenClaw plugin integration

### Python (PyTorch + MPS)
```
Average:    11.03 ms
Min:         7.28 ms
Max:        15.42 ms
```

**Strengths:**
- ✅ Stable and easy to implement
- ✅ Good baseline for GPU acceleration

## OpenClaw Plugin Integration

Arctic V1 v2 serves as the embedding backend for the **`memory-arctic`** OpenClaw plugin:

```
Agent Turn → memory-arctic plugin → spawn("--json") → arctic_embed_libtorch (C++/MPS) → LanceDB (384-dim L2)
```

- **Tools**: `memory_recall`, `memory_store`, `memory_forget`
- **Hooks**: `before_agent_start` (auto-recall), `agent_end` (auto-capture)
- **Zero API keys required** — fully local, privacy-first

## Recommendations

### For Production Use

**For absolute best performance in OpenClaw:**
→ Use **C++ LibTorch + MPS v2** (6.55ms) as `memory-arctic` plugin ✅

### Future Roadmap

- **Batch Processing**: Native batch embedding in C++ binary (single process, multiple texts).
- **Model Quantization**: INT8/FP16 for even lower latency.

## Conclusions

1. **C++ LibTorch MPS v2 is fastest** (6.55ms) — 9.9% faster than v1, beating the 7ms target.
2. **Python is 1.7x slower** than the optimized C++ implementation.
3. **~46x faster than OpenAI API** with zero cost and full privacy.
4. **Production-ready** as OpenClaw `memory-arctic` plugin with full WordPiece tokenizer.

---

**Optimized with precision by Telecro (텔리크로) 🖤**
