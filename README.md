# Arctic Embedding V1 🏔️

**The fastest local embedding engine for Apple Silicon — now an OpenClaw memory plugin.**

[🇰🇷 한국어 버전(Korean)](./README_KR.md)

---

Arctic Embedding V1 is a high-performance C++ implementation of the **Snowflake-Arctic-Embed-Tiny** model, optimized for Apple Silicon (M1/M2/M3) using **LibTorch + Metal Performance Shaders (MPS)**. It serves as the embedding backend for the **`memory-arctic`** OpenClaw plugin, providing 100% local, zero-cost, privacy-first long-term memory.

## 🚀 Performance at a Glance

| Feature | Arctic V1 (Local) | OpenAI Embedding API | Standard Python |
|---------|-------------------|---------------------|-----------------|
| **Latency** | ⚡ **6.55ms** | 🌐 200-500ms | 🐢 11ms |
| **Cost** | **$0** | $0.02/1M tokens | $0 |
| **Privacy** | **100% Local** | ❌ Cloud | 100% Local |
| **Offline** | ✅ | ❌ | ✅ |
| **API Key** | **Not needed** | Required | Not needed |
| **Dimensions** | 384 | 1536 | 384 |

## 📦 Installation

### Prerequisites

- macOS with Apple Silicon (M1/M2/M3)
- Xcode Command Line Tools
- Homebrew PyTorch: `brew install pytorch`

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/stupidcoderJung/arctic-embedding-v1.git
   cd arctic-embedding-v1
   ```

2. **Symlink LibTorch libraries**
   ```bash
   ln -sf /opt/homebrew/Cellar/pytorch/$(brew list --versions pytorch | awk '{print $2}')/lib ./libtorch/lib
   ```

3. **Build C++ Engine**
   ```bash
   make
   ```

4. **Verify**
   ```bash
   # Benchmark mode
   PYTORCH_ENABLE_MPS_FALLBACK=1 ./bin/arctic_embed_libtorch arctic_model_mps.pt "Hello world"

   # JSON embedding mode
   PYTORCH_ENABLE_MPS_FALLBACK=1 ./bin/arctic_embed_libtorch arctic_model_mps.pt "Hello world" --json
   ```

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────┐
│                  OpenClaw Gateway                     │
│                                                      │
│  ┌─────────────┐   ┌──────────────────────────────┐ │
│  │ Agent Turn   │──▶│  memory-arctic plugin         │ │
│  │ (any model)  │   │  (index.ts)                   │ │
│  └─────────────┘   │                                │ │
│                     │  memory_recall / memory_store  │ │
│                     │         │                      │ │
│                     │    spawn("--json")             │ │
│                     │         │                      │ │
│                     │  ┌──────▼──────────────────┐  │ │
│                     │  │ arctic_embed_libtorch    │  │ │
│                     │  │ (C++ / MPS GPU)          │  │ │
│                     │  │ WordPiece → Model → JSON │  │ │
│                     │  └─────────────────────────┘  │ │
│                     │         │                      │ │
│                     │  ┌──────▼──────┐              │ │
│                     │  │   LanceDB   │              │ │
│                     │  │ (384-dim L2) │              │ │
│                     │  └─────────────┘              │ │
│                     └──────────────────────────────┘ │
└──────────────────────────────────────────────────────┘
```

### C++ Engine (`src/arctic_embed_libtorch.cpp`)
- **WordPiece Tokenizer**: Full BERT-compatible tokenizer (30,522 vocab) implemented in C++
- **LibTorch + MPS**: PyTorch C++ API with Metal GPU acceleration
- **Dual Mode**: `--json` for plugin integration, default for benchmarking
- **Auto vocab detection**: Loads `vocab.txt` relative to binary path

### OpenClaw Plugin (`index.ts`)
- **Tools**: `memory_recall`, `memory_store`, `memory_forget`
- **Hooks**: `before_agent_start` (auto-recall), `agent_end` (auto-capture)
- **CLI**: `openclaw ltm list|search|stats`
- **Storage**: LanceDB with 384-dimensional L2 vector search

## 🔧 OpenClaw Integration

### As a Plugin (Recommended)

Add to your `openclaw.json`:

```json
{
  "plugins": {
    "load": {
      "paths": ["/path/to/arctic-embedding-v1"]
    },
    "slots": {
      "memory": "memory-arctic"
    },
    "entries": {
      "memory-arctic": {
        "enabled": true,
        "config": {
          "autoRecall": true,
          "autoCapture": true
        }
      }
    }
  }
}
```

Then install dependencies:
```bash
cd /path/to/arctic-embedding-v1
npm install
```

### Plugin Configuration

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `autoRecall` | boolean | `true` | Inject relevant memories before each agent turn |
| `autoCapture` | boolean | `true` | Auto-capture important info after conversations |
| `dbPath` | string | `~/.openclaw/memory/lancedb` | LanceDB storage path |

## 📊 Benchmarks

*MacBook Air M1 (8GB RAM), 1000 iterations, 50 warmup*

| Implementation | Avg Latency | vs Arctic V1 |
|---------------|-------------|--------------|
| **C++ LibTorch + MPS** | **6.55 ms** | — |
| Python (PyTorch + MPS) | 11.03 ms | 1.7x slower |
| C++ LibTorch CPU | 29.85 ms | 4.6x slower |
| C++ ONNX Runtime CPU | 108.32 ms | 16.5x slower |
| OpenAI API (network) | ~300 ms | ~46x slower |

Detailed analysis: [FINAL_BENCHMARK.md](./FINAL_BENCHMARK.md)

## 📁 Project Structure

```
arctic-embedding-v1/
├── src/
│   ├── arctic_embed_libtorch.cpp   # C++ engine (tokenizer + model + JSON output)
│   └── arctic-embeddings-lancedb.ts # Legacy standalone TS wrapper
├── bin/
│   ├── arctic_embed_libtorch       # Compiled binary (arm64)
│   └── vocab.txt                   # BERT WordPiece vocabulary (30,522 tokens)
├── arctic_model_mps.pt             # TorchScript model (86.8MB, MPS-traced)
├── index.ts                        # OpenClaw plugin entry point
├── config.ts                       # Plugin config schema
├── openclaw.plugin.json            # Plugin manifest
├── package.json                    # npm dependencies
├── Makefile                        # Build config (Homebrew PyTorch)
├── libtorch/                       # Headers + lib symlink
└── docs/adr/
    ├── 001-arctic-optimization.md  # ONNX → LibTorch decision
    └── 002-openclaw-plugin-integration.md  # Plugin integration decision
```

## 📝 Architecture Decision Records

- [ADR 001: Arctic Optimization](./docs/adr/001-arctic-optimization.md) — ONNX → LibTorch migration
- [ADR 002: OpenClaw Plugin Integration](./docs/adr/002-openclaw-plugin-integration.md) — Plugin architecture decisions

---

**Built with precision by Telecro (텔리크로) 🖤**
