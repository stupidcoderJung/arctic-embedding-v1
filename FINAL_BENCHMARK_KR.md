# 최종 벤치마크 결과 - Arctic Embedding V1 🇰🇷

**테스트 환경**: MacBook Air M1, 8GB RAM, macOS Sequoia
**테스트 날짜**: 2026-02-05
**테스트 입력**: "OpenClaw is an AI assistant framework"
**반복 횟수**: 1,000회 (50회 웜업 후) - **엄격한 변인 통제 적용**

[English Version](./FINAL_BENCHMARK.md)

---

## 결과 요약

| 구현 방식 | 평균 레이턴시 | 최소 | 최대 | 표준 편차 | 상태 |
|---------------|-----------------|-----|-----|---------|-----------|
| **C++ LibTorch + MPS** | **7.27 ms** | 6.56 ms | 7.90 ms | ~0.5 ms | 🥇 **압도적 1위** |
| **Python (PyTorch + MPS)** | **11.03 ms** | 7.28 ms | 15.42 ms | ~2.1 ms | ✅ 기준점 |
| **C++ LibTorch CPU** | **29.85 ms** | 24.34 ms | 51.50 ms | ~10 ms | ✅ 실용적 |
| C++ ONNX Runtime CPU | 108.32 ms | 107.97 ms | 108.68 ms | 0.23 ms | ❌ 비권장 |

## 핵심 발견 사항

### 1. C++ LibTorch MPS: 독보적인 성능 리더
**LibTorch와 Metal Performance Shaders(MPS)**를 결합하여 **7.27ms**라는 경이로운 속도를 달성했습니다. 이는 Python 대비 **1.5배**, ONNX Runtime 대비 **14.8배** 빠른 수치입니다.

### 2. 엄격한 변인 통제
본 결과는 단순 측정이 아닌, 다음의 엄격한 통제 하에 도출되었습니다:
- **열 제어**: 측정 전 30초 유휴 상태 유지를 통해 스로틀링 배제.
- **자원 정리**: `sudo purge` 및 불필요한 백그라운드 프로세스(Chrome 등) 완전 종료.
- **통계적 유의성**: 1,000회 반복 측정을 통해 일시적인 오차 제거.

## 결론

1. **최고의 성능**: C++ LibTorch MPS가 8ms 벽을 돌파하며 가장 빠른 엔진임을 입증했습니다.
2. **효율성**: Python 인터프리터 오버헤드를 제거하여 GPU 가속 효과를 극대화했습니다.

---

**텔리크로(Telecro)의 정밀 측정 및 보고. 🖤**
