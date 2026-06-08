"""
Task 9 — Retrieval Pipeline Hoàn Chỉnh.

Pipeline:
    Query
      ├→ Semantic Search (Jina dense)   ──┐
      ├→ Lexical Search (Weaviate BM25) ──┴→ RRF Merge → Cohere Rerank → Results
      │
      └→ Nếu best_score < threshold → Fallback PageIndex

Ngoài ra còn expose hybrid_search() dùng Weaviate hybrid built-in
(alpha=0.5 pha đều dense + BM25) cho trường hợp muốn single-call.
"""

import os

from dotenv import load_dotenv

load_dotenv()

from .task4_chunking_indexing import WEAVIATE_URL, COLLECTION_NAME
from .task5_semantic_search import semantic_search, _embed_query
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank, rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


SCORE_THRESHOLD = 0.3
DEFAULT_TOP_K = 5
RERANK_METHOD = "cross_encoder"  # Cohere rerank-v3.5


def hybrid_search_weaviate(query: str, top_k: int = 10, alpha: float = 0.5) -> list[dict]:
    """
    Weaviate hybrid search built-in: alpha=0.5 blends dense + BM25 equally.
    alpha=1.0 → pure vector, alpha=0.0 → pure BM25.
    """
    import weaviate

    query_vec = _embed_query(query)
    client = weaviate.Client(WEAVIATE_URL)

    result = (
        client.query
        .get(COLLECTION_NAME, ["content", "source", "doc_type", "chunk_index"])
        .with_hybrid(query=query, vector=query_vec, alpha=alpha)
        .with_limit(top_k)
        .with_additional(["score"])
        .do()
    )

    hits = result.get("data", {}).get("Get", {}).get(COLLECTION_NAME, [])
    output = []
    for h in hits:
        score = float(h.get("_additional", {}).get("score", 0.0))
        output.append({
            "content": h.get("content", ""),
            "score": score,
            "metadata": {
                "source": h.get("source", ""),
                "type": h.get("doc_type", ""),
                "chunk_index": h.get("chunk_index", 0),
            },
            "source": "hybrid",
        })
    return output


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """
    Retrieval pipeline hoàn chỉnh.

    1. semantic_search (Jina + Weaviate near_vector)
    2. lexical_search  (Weaviate BM25)
    3. RRF merge
    4. Cohere rerank
    5. PageIndex fallback nếu best_score < threshold

    Returns:
        List of {'content', 'score', 'metadata', 'source'}
    """
    fetch_k = top_k * 3

    dense_results = semantic_search(query, top_k=fetch_k)
    sparse_results = lexical_search(query, top_k=fetch_k)

    for item in dense_results:
        item["source"] = "hybrid"
    for item in sparse_results:
        item["source"] = "hybrid"

    lists = [l for l in [dense_results, sparse_results] if l]
    merged = rerank_rrf(lists, top_k=fetch_k) if lists else []
    for item in merged:
        item["source"] = "hybrid"

    if use_reranking and merged:
        final_results = rerank(query, merged, top_k=top_k, method=RERANK_METHOD)
    else:
        final_results = merged[:top_k]

    best_score = final_results[0]["score"] if final_results else 0.0
    if not final_results or best_score < score_threshold:
        print(
            f"  Hybrid score ({best_score:.3f}) < threshold ({score_threshold}). "
            "Fallback to PageIndex"
        )
        fallback = pageindex_search(query, top_k=top_k)
        if fallback:
            return fallback

    return final_results[:top_k]


if __name__ == "__main__":
    queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý",
        "Nghệ sĩ nào bị bắt vì sử dụng ma tuý năm 2024",
        "Luật phòng chống ma tuý 2021 quy định gì về cai nghiện",
    ]
    for q in queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.4f}] [{r['source']}] {r['content'][:80]}...")
