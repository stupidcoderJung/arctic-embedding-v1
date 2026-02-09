# 최종 벤치마크 결과 - Arctic Embedding V1 🇰🇷

**테스트 환경**: MacBook Air M1, 8GB RAM, macOS Sequoia
**테스트 날짜**: 2026-02-09 (v2 — WordPiece 토크나이저 + OpenClaw 플러그인)
**테스트 입력**: "OpenClaw is an AI assistant framework"
**반복 횟수**: 1,000회 (50회 웜업 후) - **엄격한 변인 통제 적용**

[English Version](./FINAL_BENCHMARK.md)

---

## 결과 요약

| 구현 방식 | 평균 레이턴시 | 최소 | 최대 | 표준 편차 | 상태 |
|---------------|-----------------|-----|-----|---------|-----------|
| **C++ LibTorch + MPS (v2)** | **6.55 ms** | 5.92 ms | 7.21 ms | ~0.4 ms | 🥇 **압도적 1위** |
| Python (PyTorch + MPS) | 11.03 ms | 7.28 ms | 15.42 ms | ~2.1 ms | ✅ 기준점 |
| C++ LibTorch CPU | 29.85 ms | 24.34 ms | 51.50 ms | ~10 ms | ✅ 실용적 |
| C++ ONNX Runtime CPU | 108.32 ms | 107.97 ms | 108.68 ms | 0.23 ms | ❌ 비권장 |
| OpenAI API (네트워크) | ~300 ms | ~200 ms | ~500 ms | — | ❌ 클라우드 의존 |

## 핵심 발견 사항

### 1. C++ LibTorch MPS v2: 6.55ms 달성

**LibTorch + MPS**에 **네이티브 WordPiece 토크나이저**(C++ 구현, 30,522 어휘)를 결합하여 **6.55ms**를 달성했습니다. Python 대비 **1.7배**, ONNX Runtime 대비 **16.5배**, OpenAI API 대비 **~46배** 빠른 수치입니다.

**C++ LibTorch MPS가 최강인 이유:**
- **인터프리터 오버헤드 제로**: Python GIL/런타임 없음
- **완전한 GPU 가속**: MPS를 통해 M1 GPU 직접 활용
- **네이티브 WordPiece 토크나이저**: BERT 호환 30,522 토큰 어휘 C++ 구현
- **최적화된 C++ 코어**: 최소한의 메모리 관리 오버헤드

### 2. v1 → v2 개선 내역

| | v1 (2026-02-05) | v2 (2026-02-09) | 개선 |
|---|---|---|---|
| **평균 레이턴시** | 7.27 ms | **6.55 ms** | **-9.9%** |
| **토크나이저** | 플레이스홀더 (하드코딩 ID) | 완전한 WordPiece (30,522 어휘) | 프로덕션 수준 |
| **출력** | 벤치마크 전용 | `--json` 모드 | 플러그인 연동 |
| **플러그인** | 단독 바이너리 | OpenClaw `memory-arctic` | 드롭인 교체 |

### 3. 엄격한 변인 통제

본 결과는 다음의 엄격한 통제 하에 도출되었습니다:
- **열 제어**: 측정 전 30초 유휴 상태 유지를 통해 스로틀링 배제.
- **자원 정리**: `sudo purge` 및 불필요한 백그라운드 프로세스 완전 종료.
- **통계적 유의성**: 1,000회 반복 측정을 통해 일시적인 오차 제거.

### 4. 클라우드 API 대비 비교

| | Arctic V1 (로컬) | OpenAI text-embedding-3-small |
|---|---|---|
| **레이턴시** | 6.55 ms | ~300 ms |
| **비용** | $0 | $0.02/1M tokens |
| **프라이버시** | 100% 로컬 | 클라우드 |
| **오프라인** | 가능 | 불가 |
| **벡터 차원** | 384 | 1536 |

## OpenClaw 플러그인 통합

Arctic V1 v2는 **`memory-arctic`** OpenClaw 플러그인의 임베딩 백엔드로 동작합니다:

```
에이전트 턴 → memory-arctic 플러그인 → spawn("--json") → arctic_embed_libtorch (C++/MPS) → LanceDB (384차원 L2)
```

- **도구**: `memory_recall`, `memory_store`, `memory_forget`
- **훅**: `before_agent_start` (자동 회상), `agent_end` (자동 캡처)
- **API 키 불필요** — 완전 로컬, 프라이버시 우선

## 결론

1. **C++ LibTorch MPS v2가 최고 성능** (6.55ms) — v1 대비 9.9% 향상, 7ms 벽 돌파.
2. **Python 대비 1.7배 빠름** — 인터프리터 오버헤드 완전 제거.
3. **OpenAI API 대비 ~46배 빠름** — 비용 제로, 완전한 프라이버시.
4. **프로덕션 준비 완료** — OpenClaw `memory-arctic` 플러그인으로 완전한 WordPiece 토크나이저 탑재.

---

**텔리크로(Telecro)의 정밀 측정 및 보고. 🖤**
