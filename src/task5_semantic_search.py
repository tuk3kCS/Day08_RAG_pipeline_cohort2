"""
Task 5 — Semantic Search Module.

Dense retrieval: Cohere embed-v4.0 (query) + Weaviate near_vector.
input_type="search_query" cho query embedding (khác với "search_document" lúc index).
"""

import os

from dotenv import load_dotenv

load_dotenv()

from .task4_chunking_indexing import (
    EMBEDDING_MODEL,
    COHERE_API_KEY,
    WEAVIATE_URL,
    COLLECTION_NAME,
)


def _embed_query(query: str) -> list[float]:
    """Embed query với Cohere search_query input type."""
    import cohere
    co = cohere.ClientV2(COHERE_API_KEY)
    resp = co.embed(
        model=EMBEDDING_MODEL,
        input_type="search_query",
        texts=[query],
        embedding_types=["float"],
    )
    return list(resp.embeddings.float[0])


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa dùng Cohere embedding + Weaviate near_vector.

    Args:
        query: Câu truy vấn
        top_k: Số kết quả tối đa

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}
        Sorted by score descending.
    """
    import weaviate

    query_vec = _embed_query(query)
    client = weaviate.Client(WEAVIATE_URL)

    result = (
        client.query
        .get(COLLECTION_NAME, ["content", "source", "doc_type", "chunk_index"])
        .with_near_vector({"vector": query_vec, "certainty": 0.0})
        .with_limit(top_k)
        .with_additional(["certainty", "distance"])
        .do()
    )

    hits = result.get("data", {}).get("Get", {}).get(COLLECTION_NAME, [])
    output = []
    for h in hits:
        add = h.get("_additional", {})
        score = float(add.get("certainty", 0.0))
        output.append({
            "content": h.get("content", ""),
            "score": score,
            "metadata": {
                "source": h.get("source", ""),
                "type": h.get("doc_type", ""),
                "chunk_index": h.get("chunk_index", 0),
            },
        })

    output.sort(key=lambda x: x["score"], reverse=True)
    return output


if __name__ == "__main__":
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
