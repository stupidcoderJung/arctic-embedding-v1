# ADR 001: Arctic Embedding Service Optimization for M1 Mac

## Status
Superceded by LibTorch Implementation (2026-02-05)

## Context
Snowflake-Arctic-Embed-Tiny 모델을 MacBook Air M1(8GB RAM) 환경에서 고성능으로 운용해야 함. 초기에는 ONNX Runtime을 시도했으나 Apple Silicon 최적화 빌드의 복잡성과 성능 한계(108ms)로 인해 목표인 sub-10ms 달성이 불가능했음.

## Decisions
1. **Switch to LibTorch (PyTorch C++ API)**: ONNX Runtime을 포기하고, PyTorch와 동일한 Metal Performance Shaders(MPS) 가속을 사용할 수 있는 LibTorch로 전환함.
2. **MPS-Native Tracing**: 모델을 MPS 장치에서 직접 TorchScript로 추출하여 하드웨어 가속을 극대화함.
3. **Hybrid Acceleration (MPS Fallback)**: 일부 미지원 연산에 대해 `PYTORCH_ENABLE_MPS_FALLBACK=1`을 활성화하여 안정성과 성능의 균형을 맞춤.
4. **Sub-8ms Optimization**: C++의 저수준 메모리 관리와 LibTorch의 효율적인 텐서 연산을 결합하여 Python baseline(11ms)을 능가하는 7.27ms를 달성함.

## Consequences
- **속도**: 순수 추론 레이턴시 **7.27ms** 달성 (M1 GPU 가속).
- **이식성**: OpenClaw 환경에서 단일 바이너리 호출만으로 초고속 임베딩 가능.
- **자원**: LibTorch 라이브러리 의존성이 생겼으나, 성능 이득이 이를 압도함.

## Evolution History
- **v1 (2026-02-04)**: Initial ONNX Runtime attempt (Result: 108ms, Fail)
- **v2 (2026-02-05)**: LibTorch CPU implementation (Result: 29.85ms, Better)
- **v3 (2026-02-05)**: LibTorch MPS + Fallback optimization (Result: 7.27ms, Success)
