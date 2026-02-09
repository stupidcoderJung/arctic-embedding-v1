# Arctic Embedding V1 🏔️ (한국어)

**Apple Silicon을 위한 가장 빠른 로컬 임베딩 엔진 — OpenClaw 메모리 플러그인.**

[English Version](./README.md)

---

Arctic Embedding V1은 **Snowflake-Arctic-Embed-Tiny** 모델의 고성능 C++ 구현체입니다. **LibTorch + MPS(Metal Performance Shaders)**를 사용하여 Apple Silicon(M1/M2/M3)에 최적화되었으며, **`memory-arctic`** OpenClaw 플러그인의 임베딩 백엔드로 동작합니다. 100% 로컬, 무료, 프라이버시 보장.

## 🚀 한눈에 보는 성능

| 항목 | Arctic V1 (로컬) | OpenAI Embedding API | 표준 Python |
|---------|-------------------|---------------------|-----------------|
| **지연** | ⚡ **6.55ms** | 🌐 200-500ms | 🐢 11ms |
| **비용** | **$0** | $0.02/1M tokens | $0 |
| **프라이버시** | **100% 로컬** | ❌ 클라우드 | 100% 로컬 |
| **오프라인** | ✅ | ❌ | ✅ |
| **API 키** | **불필요** | 필수 | 불필요 |
| **벡터 차원** | 384 | 1536 | 384 |

## 📦 설치 방법

### 요구 사항

- Apple Silicon (M1/M2/M3) 탑재 macOS
- Xcode Command Line Tools
- Homebrew PyTorch: `brew install pytorch`

### 빠른 시작

1. **저장소 클론**
   ```bash
   git clone https://github.com/stupidcoderJung/arctic-embedding-v1.git
   cd arctic-embedding-v1
   ```

2. **LibTorch 라이브러리 심링크**
   ```bash
   ln -sf /opt/homebrew/Cellar/pytorch/$(brew list --versions pytorch | awk '{print $2}')/lib ./libtorch/lib
   ```

3. **C++ 엔진 빌드**
   ```bash
   make
   ```

4. **동작 확인**
   ```bash
   # 벤치마크 모드
   PYTORCH_ENABLE_MPS_FALLBACK=1 ./bin/arctic_embed_libtorch arctic_model_mps.pt "안녕하세요"

   # JSON 임베딩 모드
   PYTORCH_ENABLE_MPS_FALLBACK=1 ./bin/arctic_embed_libtorch arctic_model_mps.pt "안녕하세요" --json
   ```

## 🏗️ 아키텍처

```
┌──────────────────────────────────────────────────────┐
│                  OpenClaw Gateway                     │
│                                                      │
│  ┌─────────────┐   ┌──────────────────────────────┐ │
│  │ 에이전트 턴  │──▶│  memory-arctic 플러그인       │ │
│  │ (모든 모델)  │   │  (index.ts)                   │ │
│  └─────────────┘   │                                │ │
│                     │  memory_recall / memory_store  │ │
│                     │         │                      │ │
│                     │    spawn("--json")             │ │
│                     │         │                      │ │
│                     │  ┌──────▼──────────────────┐  │ │
│                     │  │ arctic_embed_libtorch    │  │ │
│                     │  │ (C++ / MPS GPU)          │  │ │
│                     │  │ WordPiece → 모델 → JSON  │  │ │
│                     │  └─────────────────────────┘  │ │
│                     │         │                      │ │
│                     │  ┌──────▼──────┐              │ │
│                     │  │   LanceDB   │              │ │
│                     │  │ (384차원 L2) │              │ │
│                     │  └─────────────┘              │ │
│                     └──────────────────────────────┘ │
└──────────────────────────────────────────────────────┘
```

### C++ 엔진 (`src/arctic_embed_libtorch.cpp`)
- **WordPiece 토크나이저**: BERT 호환 토크나이저 C++ 구현 (30,522 어휘)
- **LibTorch + MPS**: PyTorch C++ API + Metal GPU 가속
- **이중 모드**: `--json`(플러그인 연동), 기본(벤치마크)
- **자동 어휘 탐지**: 바이너리 경로 기준 `vocab.txt` 자동 로드

### OpenClaw 플러그인 (`index.ts`)
- **도구**: `memory_recall`, `memory_store`, `memory_forget`
- **훅**: `before_agent_start` (자동 회상), `agent_end` (자동 캡처)
- **CLI**: `openclaw ltm list|search|stats`
- **저장소**: LanceDB 384차원 L2 벡터 검색

## 🔧 OpenClaw 연동

### 플러그인으로 사용 (권장)

`openclaw.json`에 추가:

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

의존성 설치:
```bash
cd /path/to/arctic-embedding-v1
npm install
```

### 플러그인 설정

| 키 | 타입 | 기본값 | 설명 |
|-----|------|---------|-------------|
| `autoRecall` | boolean | `true` | 매 에이전트 턴 전 관련 기억 자동 주입 |
| `autoCapture` | boolean | `true` | 대화 후 중요 정보 자동 캡처 |
| `dbPath` | string | `~/.openclaw/memory/lancedb` | LanceDB 저장 경로 |

## 📊 벤치마크 결과

*MacBook Air M1 (8GB RAM), 1000회 반복, 50회 워밍업*

| 구현 방식 | 평균 지연 | Arctic V1 대비 |
|---------------|-------------|--------------|
| **C++ LibTorch + MPS** | **6.55 ms** | — |
| Python (PyTorch + MPS) | 11.03 ms | 1.7배 느림 |
| C++ LibTorch CPU | 29.85 ms | 4.6배 느림 |
| C++ ONNX Runtime CPU | 108.32 ms | 16.5배 느림 |
| OpenAI API (네트워크) | ~300 ms | ~46배 느림 |

상세 분석: [FINAL_BENCHMARK_KR.md](./FINAL_BENCHMARK_KR.md)

## 📁 프로젝트 구조

```
arctic-embedding-v1/
├── src/
│   ├── arctic_embed_libtorch.cpp   # C++ 엔진 (토크나이저 + 모델 + JSON 출력)
│   └── arctic-embeddings-lancedb.ts # 레거시 단독 TS 래퍼
├── bin/
│   ├── arctic_embed_libtorch       # 컴파일된 바이너리 (arm64)
│   └── vocab.txt                   # BERT WordPiece 어휘 (30,522 토큰)
├── arctic_model_mps.pt             # TorchScript 모델 (86.8MB, MPS 트레이싱)
├── index.ts                        # OpenClaw 플러그인 진입점
├── config.ts                       # 플러그인 설정 스키마
├── openclaw.plugin.json            # 플러그인 매니페스트
├── package.json                    # npm 의존성
├── Makefile                        # 빌드 설정 (Homebrew PyTorch)
├── libtorch/                       # 헤더 + lib 심링크
└── docs/adr/
    ├── 001-arctic-optimization.md  # ONNX → LibTorch 결정
    └── 002-openclaw-plugin-integration.md  # 플러그인 통합 결정
```

## 📝 아키텍처 결정 기록 (ADR)

- [ADR 001: Arctic 최적화](./docs/adr/001-arctic-optimization.md) — ONNX → LibTorch 전환
- [ADR 002: OpenClaw 플러그인 통합](./docs/adr/002-openclaw-plugin-integration.md) — 플러그인 아키텍처 결정

---

**주인님의 명령에 따라 텔리크로(Telecro)가 정밀하게 최적화하였습니다. 🖤**
