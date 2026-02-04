# Arctic Embedding V1 - Benchmark Results

## Test Environment
- **Device**: MacBook Air M1 (8GB RAM)
- **OS**: macOS Sequoia
- **Date**: 2026-02-05

## Results

| Test | Text Length | Time (ms) | Notes |
|------|-------------|-----------|-------|
| Short text | ~3 tokens | 141.84 | Cold start (includes model loading) |
| Medium text | ~20 tokens | 107.24 | Single run |
| Long text | ~70 tokens | 106.77 | Single run |
| **Average** | **~20 tokens** | **99.43** | **5 runs** |

### Details

**Short Text:**
```
"Hello, OpenClaw!"
```
- Time: 141.84ms
- Dimension: 384

**Medium Text:**
```
"Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed."
```
- Time: 107.24ms
- Dimension: 384

**Long Text:**
```
"Retrieval-Augmented Generation (RAG) is an advanced technique in natural language processing that combines the power of large language models with external knowledge retrieval systems..."
```
- Time: 106.77ms
- Dimension: 384

## 5-Run Average (Medium Text)
1. Run 1: 99.79ms
2. Run 2: 99.27ms
3. Run 3: 99.41ms
4. Run 4: 99.16ms
5. Run 5: 99.53ms

**Average: 99.43ms**

## Conclusions

- **Consistent performance**: ~100ms regardless of text length (after model loading)
- **Cold start overhead**: ~40ms additional for first run
- **Memory footprint**: ~100MB (as expected)
- **Dimension**: 384 (accurate mean pooling)

## Comparison

| Approach | Time | Memory | Privacy |
|----------|------|--------|---------|
| Arctic V1 (C++) | ~100ms | ~100MB | ✅ Local |
| Python (transformers) | ~200-400ms | ~500MB-1GB | ✅ Local |
| Cloud API | ~300-1000ms | N/A | ❌ Cloud |

---
**Note**: These are real-world measurements on a MacBook Air M1 (8GB RAM). Performance may vary on different hardware.
