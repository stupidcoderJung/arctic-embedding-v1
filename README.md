# Arctic Embedding V1 🏔️

**The fastest local embedding engine for Apple Silicon.**

[🇰🇷 한국어 버전(Korean)](./README_KR.md)

---

Arctic Embedding V1 is a high-performance C++ implementation of the **Snowflake-Arctic-Embed-Tiny** model, optimized specifically for MacBook Air/Pro (M1/M2/M3) using **LibTorch and Metal Performance Shaders (MPS)**.

## 🚀 Performance at a Glance

Compared to standard Python or ONNX approaches:

| Feature | Arctic V1 + LibTorch | Standard Python | Cloud APIs |
|---------|---------------------|-----------------|------------|
| **Startup Time** | **< 100ms** | 2-5s | N/A |
| **Memory Usage** | **~100MB** | ~500MB-1GB | N/A |
| **Cost** | **$0** | $0 | $$$ |
| **Privacy** | **100% Local** | 100% Local | ❌ Cloud |
| **Speed** | ⚡ **7.27ms** | 🐢 11-15ms | 🌐 200-1000ms |

## 📦 Installation

### Prerequisites

- macOS with Apple Silicon (M1/M2/M3)
- Xcode Command Line Tools
- **LibTorch (PyTorch C++ API)**: Download and extract to `./libtorch`

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/stupidcoderJung/arctic-embedding-v1.git
   cd arctic-embedding-v1
   ```

2. **Download & Prepare Model**
   ```bash
   # The model should be in TorchScript format for MPS
   # arctic_model_mps.pt
   ```

3. **Build C++ Engine**
   ```bash
   make
   ```

## 🏗️ Architecture

### 1. C++ Core (`src/arctic_embed_libtorch.cpp`)
- **LibTorch Integration**: Uses the official PyTorch C++ API for maximum performance.
- **MPS Acceleration**: Fully utilizes M1/M2/M3 GPU via Metal Performance Shaders.
- **Hybrid Fallback**: Intelligent CPU fallback for unsupported operations.

### 2. TypeScript Plugin (`src/arctic-embeddings-lancedb.ts`)
- Optimized for **OpenClaw** and **LanceDB**.
- Automatic environment configuration for MPS.
- 384-dimensional normalized output.

## 📊 Benchmarks

*Tested on MacBook Air M1 (8GB RAM)*

| Implementation | Avg Latency | Status |
|---------------|-------------|--------|
| **C++ LibTorch + MPS** | **7.27 ms** | 🥇 **Leader** |
| Python (PyTorch + MPS) | 11.03 ms | 1.5x slower |
| C++ LibTorch CPU | 29.85 ms | 4.1x slower |
| C++ ONNX Runtime CPU | 108.32 ms | ❌ Deprecated |

Detailed analysis can be found in [FINAL_BENCHMARK.md](./FINAL_BENCHMARK.md) and [ADR 001](./docs/adr/001-arctic-optimization.md).

## 🤝 Integration with OpenClaw

To use this engine in your OpenClaw environment:

```bash
ln -sf $(pwd)/bin/arctic_embed_libtorch ~/.openclaw/workspace/arctic_embed_mps
```

See [Deployment Guide](./README.md#integration-with-openclaw) for more details.

---

**Built with precision by Telecro (텔리크로) 🖤**
