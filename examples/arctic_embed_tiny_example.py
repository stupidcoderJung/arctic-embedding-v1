"""
Arctic Embed V1 — C++ Binary Example

Demonstrates using the arctic_embed_libtorch binary for:
1. Single text embedding
2. Batch embedding
3. LanceDB vector search

Requires: arctic_model_mps.pt + bin/arctic_embed_libtorch (built via `make`)
"""

import subprocess
import json
import os
import time
import lancedb
import numpy as np

# Paths (relative to project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BINARY = os.path.join(PROJECT_ROOT, "bin", "arctic_embed_libtorch")
MODEL = os.path.join(PROJECT_ROOT, "arctic_model_mps.pt")
VECTOR_DIM = 384


def embed_text(text: str) -> list[float]:
    """Embed a single text using the C++ binary."""
    result = subprocess.run(
        [BINARY, MODEL, text, "--json"],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1"},
    )
    if result.returncode != 0:
        raise RuntimeError(f"Embedding failed: {result.stderr}")
    return json.loads(result.stdout.strip())


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed multiple texts sequentially."""
    return [embed_text(t) for t in texts]


def main():
    print("=== Arctic Embed V1 — C++ Binary Example ===\n")

    # 1. Single embedding
    print("1. Single text embedding")
    text = "OpenClaw is an AI assistant framework"
    start = time.perf_counter()
    vec = embed_text(text)
    elapsed = (time.perf_counter() - start) * 1000
    print(f"   Text: \"{text}\"")
    print(f"   Dim:  {len(vec)}")
    print(f"   Time: {elapsed:.0f}ms (includes process spawn)\n")

    # 2. Batch embedding
    sample_texts = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is a subset of artificial intelligence",
        "Python is a versatile programming language",
        "Natural language processing enables computers to understand human language",
        "Vector databases are essential for semantic search applications",
    ]

    print(f"2. Batch embedding ({len(sample_texts)} texts)")
    start = time.perf_counter()
    embeddings = embed_batch(sample_texts)
    elapsed = (time.perf_counter() - start) * 1000
    print(f"   Total: {elapsed:.0f}ms ({elapsed / len(sample_texts):.0f}ms/text)\n")

    # 3. LanceDB vector search
    print("3. LanceDB vector search")
    db_path = os.path.join(PROJECT_ROOT, ".lancedb-example")
    db = lancedb.connect(db_path)

    data = [
        {"id": i, "vector": emb, "text": txt}
        for i, (emb, txt) in enumerate(zip(embeddings, sample_texts))
    ]
    table = db.create_table("demo", data, mode="overwrite")
    print(f"   Stored {len(data)} vectors in LanceDB")

    query = "Find information about machine learning"
    print(f"   Query: \"{query}\"")
    query_vec = embed_text(query)
    results = table.search(query_vec).limit(3).to_list()

    print("\n   Top 3 results:")
    for i, r in enumerate(results):
        print(f"   {i+1}. [{r['_distance']:.4f}] {r['text']}")

    # Cleanup
    import shutil
    shutil.rmtree(db_path, ignore_errors=True)

    print("\nDone.")


if __name__ == "__main__":
    main()
