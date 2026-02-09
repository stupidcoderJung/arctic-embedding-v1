# ADR 001: Arctic Embedding Service Optimization for M1 Mac

## Status
Accepted — Evolved to v4 (2026-02-09)

See also: [ADR 002: OpenClaw Plugin Integration](./002-openclaw-plugin-integration.md)

## Context
Snowflake-Arctic-Embed-Tiny 모델을 MacBook Air M1(8GB RAM) 환경에서 고성능으로 운용해야 함. 초기에는 ONNX Runtime을 시도했으나 Apple Silicon 최적화 빌드의 복잡성과 성능 한계(108ms)로 인해 목표인 sub-10ms 달성이 불가능했음.

### 최종 목표
- Sub-7ms 순수 추론 레이턴시
- 100% 로컬 (API 키 불필요, 네트워크 불필요)
- OpenClaw 메모리 플러그인으로 프로덕션 통합

## Decisions

### 1. ONNX Runtime → LibTorch 전환
ONNX Runtime을 포기하고, PyTorch와 동일한 Metal Performance Shaders(MPS) 가속을 사용할 수 있는 LibTorch로 전환함.

### 2. MPS-Native Tracing
모델을 MPS 장치에서 직접 TorchScript로 추출하여 하드웨어 가속을 극대화함. `attn_implementation="eager"` 설정으로 MPS 호환성 확보.

### 3. Hybrid Acceleration (MPS Fallback)
일부 미지원 연산에 대해 `PYTORCH_ENABLE_MPS_FALLBACK=1`을 활성화하여 안정성과 성능의 균형을 맞춤.

### 4. WordPiece Tokenizer C++ 구현 (v4)
- 기존 placeholder tokenizer (하드코딩 토큰 ID)를 제거
- BERT WordPiece 토크나이저를 C++로 직접 구현 (30,522 vocab)
- `vocab.txt`를 바이너리 경로에서 자동 탐지
- 프로덕션 수준의 정확한 토큰화 달성

### 5. JSON 출력 모드 (v4)
- `--json` 플래그로 384차원 float JSON 배열을 stdout 출력
- 로그/디버그 메시지는 stderr로 분리
- OpenClaw 플러그인이 `child_process.spawn()`으로 바이너리 호출 가능

### 6. Homebrew PyTorch 활용
LibTorch를 별도 다운로드하지 않고 `brew install pytorch` 활용. 빌드 시 `/opt/homebrew/Cellar/pytorch/2.10.0` 경로에서 include/lib 참조. `libtorch/lib` → homebrew 심링크 (gitignore 처리).

## Consequences

### 성능
| 버전 | 레이턴시 | 상태 |
|------|----------|------|
| v1 ONNX Runtime CPU | 108.32 ms | ❌ 실패 |
| v2 LibTorch CPU | 29.85 ms | ✅ 개선 |
| v3 LibTorch MPS | 7.27 ms | ✅ 목표 달성 |
| **v4 LibTorch MPS + WordPiece** | **6.55 ms** | 🥇 **최종** |

### 비교
| | Arctic V1 v4 (로컬) | OpenAI API |
|---|---|---|
| 레이턴시 | **6.55 ms** | ~300 ms |
| 비용 | **$0** | $0.02/1M tokens |
| 프라이버시 | **100% 로컬** | 클라우드 |
| 벡터 차원 | 384 | 1536 |

### 제약사항
- macOS Apple Silicon 전용 (MPS 가속)
- Homebrew pytorch 설치 필요
- 모델 파일 86.8MB (TorchScript)
- 임베딩 품질은 OpenAI 대비 낮을 수 있으나, 메모리 검색 용도로는 충분

## Evolution History
- **v1 (2026-02-04)**: Initial ONNX Runtime attempt (Result: 108ms, Fail)
- **v2 (2026-02-05)**: LibTorch CPU implementation (Result: 29.85ms, Better)
- **v3 (2026-02-05)**: LibTorch MPS + Fallback optimization (Result: 7.27ms, Success)
- **v4 (2026-02-09)**: WordPiece tokenizer + JSON mode + OpenClaw plugin (Result: **6.55ms**, Production)
