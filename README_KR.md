# Arctic Embedding V1 🏔️ (한국어)

**Apple Silicon을 위한 가장 빠른 로컬 임베딩 엔진.**

Arctic Embedding V1은 **Snowflake-Arctic-Embed-Tiny** 모델의 고성능 C++ 구현체입니다. **LibTorch와 MPS(Metal Performance Shaders)**를 사용하여 MacBook Air/Pro (M1/M2/M3) 환경에 최적화되었습니다.

[English Version](./README.md)

---

## 🚀 한눈에 보는 성능

기존 Python 및 ONNX 방식과의 비교:

| 기능 | Arctic V1 + LibTorch | 표준 Python | 클라우드 API |
|---------|---------------------|-----------------|------------|
| **시작 속도** | **< 100ms** | 2-5s | N/A |
| **메모리 사용량** | **~100MB** | ~500MB-1GB | N/A |
| **비용** | **무료 ($0)** | $0 | $$$ |
| **개인정보 보호** | **100% 로컬** | 100% 로컬 | ❌ 클라우드 |
| **추론 속도** | ⚡ **7.27ms** | 🐢 11-15ms | 🌐 200-1000ms |

## 📦 설치 방법

### 요구 사항

- Apple Silicon (M1/M2/M3) 탑재 macOS
- Xcode Command Line Tools
- **LibTorch (PyTorch C++ API)**: 다운로드 후 `./libtorch` 경로에 압축 해제

### 빠른 시작

1. **저장소 클론**
   ```bash
   git clone https://github.com/stupidcoderJung/arctic-embedding-v1.git
   cd arctic-embedding-v1
   ```

2. **모델 준비**
   - MPS 가속을 위해 TorchScript 포맷의 모델(`arctic_model_mps.pt`)이 필요합니다.

3. **빌드**
   ```bash
   make
   ```

## 🏗️ 아키텍처

### 1. C++ 코어 (`src/arctic_embed_libtorch.cpp`)
- **LibTorch 통합**: 공식 PyTorch C++ API를 사용하여 최대 성능 발휘.
- **MPS 가속**: Metal Performance Shaders를 통한 M1/M2/M3 GPU 완전 활용.
- **하이브리드 폴백**: 지원되지 않는 연산은 지능적으로 CPU에서 처리.

### 2. TypeScript 플러그인 (`src/arctic-embeddings-lancedb.ts`)
- **OpenClaw** 및 **LanceDB** 최적화.
- MPS 가속을 위한 자동 환경 설정.
- 384차원 정규화 출력 지원.

## 📊 벤치마크 결과

*MacBook Air M1 (8GB RAM) 측정 기준*

| 구현 방식 | 평균 레이턴시 | 상태 |
|---------------|-------------|--------|
| **C++ LibTorch + MPS** | **7.27 ms** | 🥇 **성능 리더** |
| Python (PyTorch + MPS) | 11.03 ms | 1.5배 느림 |
| C++ LibTorch CPU | 29.85 ms | 4.1배 느림 |
| C++ ONNX Runtime CPU | 108.32 ms | ❌ 지원 중단 |

자세한 분석 내용은 [FINAL_BENCHMARK_KR.md](./FINAL_BENCHMARK_KR.md)를 참조하십시오.

## 🤝 OpenClaw 연동

OpenClaw 환경에서 이 엔진을 사용하려면:

```bash
ln -sf $(pwd)/bin/arctic_embed_libtorch ~/.openclaw/workspace/arctic_embed_mps
```

---

**주인님의 명령에 따라 텔리크로(Telecro)가 정밀하게 최적화하였습니다. 🖤**
