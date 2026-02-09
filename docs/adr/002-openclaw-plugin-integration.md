# ADR 002: OpenClaw Memory Plugin Integration

## Status
Accepted (2026-02-09)

## Context
Arctic Embed V1 C++ 엔진(ADR 001)이 7.27ms → 6.55ms로 안정화된 후, 이를 OpenClaw의 장기 기억(Long-Term Memory) 플러그인으로 통합해야 함. 기존 OpenClaw의 `memory-lancedb` 플러그인은 OpenAI Embedding API를 사용하며, 매 호출마다 네트워크 왕복(~200-500ms)과 API 키가 필요함.

### 목표
- 100% 로컬 임베딩 (API 키 불필요, 네트워크 불필요)
- OpenClaw 플러그인 규격 완전 준수
- 기존 memory-lancedb와 동일한 도구 인터페이스 (drop-in replacement)

## Decisions

### 1. WordPiece Tokenizer C++ 구현
- 기존 바이너리의 placeholder tokenizer (하드코딩 토큰 ID)를 제거
- BERT WordPiece 토크나이저를 C++로 직접 구현 (30,522 vocab)
- `vocab.txt`를 바이너리 경로에서 자동 탐지

### 2. JSON 출력 모드 (`--json`)
- 기존 벤치마크 전용 바이너리에 `--json` 플래그 추가
- stdout으로 384차원 float JSON 배열 출력
- 로그/디버그 메시지는 모두 stderr로 분리
- TypeScript 플러그인이 `child_process.spawn()`으로 바이너리 호출

### 3. OpenClaw 플러그인 구조
- `openclaw.plugin.json`: id=`memory-arctic`, kind=`memory`
- `index.ts`: `register()` 함수에서 도구/훅/서비스/CLI 등록
- `config.ts`: 설정 스키마 (dbPath, autoRecall, autoCapture)
- OpenAI 의존성 제거, `embedding.apiKey` 설정 불필요

### 4. LanceDB 벡터 스토어 유지
- 384차원 벡터용 LanceDB 테이블 (기존 1536차원 → 384차원)
- L2 거리 → 유사도 점수 변환 동일
- CRUD: store/search/delete/count 동일 인터페이스

### 5. Homebrew PyTorch 활용
- LibTorch를 별도 다운로드하지 않고 `brew install pytorch` 활용
- 빌드 시 `pytorch/2.10.0` 경로에서 include/lib 참조
- `libtorch/lib` → homebrew 심링크 (gitignore 처리)

## Consequences

### 성능
| 항목 | memory-lancedb (OpenAI) | memory-arctic (Local) |
|------|------------------------|----------------------|
| 임베딩 지연 | 200-500ms (네트워크) | **~6.5ms** (MPS GPU) |
| 비용 | $0.02/1M tokens | **$0** |
| 오프라인 | 불가 | **가능** |
| 프라이버시 | 외부 전송 | **100% 로컬** |
| 벡터 차원 | 1536 | 384 |

### 등록 도구
- `memory_recall`: 벡터 유사도 검색
- `memory_store`: 정보 저장 (중복 체크 포함)
- `memory_forget`: 기억 삭제 (GDPR 준수)

### 라이프사이클 훅
- `before_agent_start`: 관련 기억 자동 주입 (auto-recall)
- `agent_end`: 중요 정보 자동 캡처 (auto-capture)

### 제약사항
- macOS Apple Silicon 전용 (MPS 가속)
- Homebrew pytorch 설치 필요
- 모델 파일 86.8MB (TorchScript)
- 임베딩 품질은 OpenAI text-embedding-3-small 대비 낮을 수 있으나, 메모리 검색 용도로는 충분

## Evolution History
- **v1 (2026-02-05)**: C++ 엔진 완성 (ADR 001), 벤치마크 전용
- **v2 (2026-02-09)**: WordPiece 토크나이저 + JSON 모드 추가
- **v3 (2026-02-09)**: OpenClaw 플러그인 통합 완료 (memory-arctic)
