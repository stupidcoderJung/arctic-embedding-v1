# ADR 001: Arctic Embedding Service Optimization for M1 Mac

## Status
Accepted (2026-02-04)

## Context
Snowflake-Arctic-Embed-Tiny 모델을 MacBook Air M1(8GB RAM) 환경에서 고성능으로 운용해야 함. 기존의 Python 런타임은 메모리 점유율이 높고 실행 속도가 느려 실시간 벡터 검색 플러그인에 부적합함.

## Decisions
1. **C++ Native Engine**: ONNX Runtime C++ API를 사용하여 추론 엔진을 구축함.
2. **Persistent Process (IPC)**: 매번 바이너리를 실행(Spawn)하지 않고, TypeScript에서 프로세스를 유지하며 표준 입출력(stdin/stdout)을 통해 통신하여 모델 로딩 오버헤드를 제거함.
3. **M1 Hardware Acceleration**: ONNX Runtime의 CoreML Execution Provider 설정을 포함하여 향후 가속 가능성을 확보함 (현재 빌드는 안정성을 위해 CPU 최적화 모드 사용).
4. **Dynamic Tensor Resizing**: 모델의 동적 차원(-1, -1)을 감지하여 가변 길이 문장에 대응하도록 설계함.

## Consequences
- **속도**: 모델 로딩이 1회로 제한되어 실시간 응답성 확보.
- **자원**: 8GB RAM 환경에서 0.2GB 미만의 메모리 사용량 유지.
- **안정성**: TypeScript 측에서 큐(Queue)와 샌니타이징(Sanitization)을 통해 프로세스 안전성 강화.
