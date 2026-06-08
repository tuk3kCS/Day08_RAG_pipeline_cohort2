"""
Task 7 — Reranking Module.

Implement 3 phương pháp:
1. Cross-encoder: Cohere Rerank v3.5 (primary — multilingual, tốt cho tiếng Việt)
2. MMR: Maximal Marginal Relevance (giảm trùng lặp, tăng diversity)
3. RRF: Reciprocal Rank Fusion (gộp nhiều ranker, không cần API)

Default method: cross_encoder (Cohere).
"""

import os

from dotenv import load_dotenv

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
JINA_API_KEY = os.getenv("JINA_API_KEY", "")


def _cosine_sim(a: list[float], b: list[float]) -> float:
    import numpy as np
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank dùng Cohere Rerank v3.5 (rerank-v3.5).

    Cohere rerank là cross-encoder: model đọc đồng thời query + document
    để tính relevance score — chính xác hơn bi-encoder vì có full attention
    giữa query và doc. Hỗ trợ tiếng Việt và 100+ ngôn ngữ.

    Fallback: Jina Reranker v2 nếu không có Cohere key.
    """
    if not candidates:
        return []

    docs = [c["content"] for c in candidates]

    # Primary: Cohere rerank-v3.5
    if COHERE_API_KEY and COHERE_API_KEY != "xxx":
        try:
            import cohere
            co = cohere.ClientV2(COHERE_API_KEY)
            resp = co.rerank(
                model="rerank-v3.5",
                query=query,
                documents=docs,
                top_n=top_k,
            )
            return [
                {**candidates[r.index], "score": float(r.relevance_score)}
                for r in resp.results
            ]
        except Exception as e:
            print(f"  Cohere rerank error: {e}, falling back to Jina")

    # Fallback: Jina Reranker v2
    if JINA_API_KEY and JINA_API_KEY != "jina_xxx":
        try:
            import requests
            resp = requests.post(
                "https://api.jina.ai/v1/rerank",
                headers={"Authorization": f"Bearer {JINA_API_KEY}"},
                json={
                    "model": "jina-reranker-v2-base-multilingual",
                    "query": query,
                    "documents": docs,
                    "top_n": top_k,
                },
                timeout=30,
            )
            resp.raise_for_status()
            reranked = resp.json()["results"]
            return [
                {**candidates[r["index"]], "score": float(r["relevance_score"])}
                for r in reranked
            ]
        except Exception as e:
            print(f"  Jina rerank error: {e}, falling back to score sort")

    # Last resort: sort by existing score
    return sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)[:top_k]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))
    lambda_param=0.7: ưu tiên relevance (70%) hơn diversity (30%).
    """
    if not candidates:
        return []

    selected: list[int] = []
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx, best_score = None, float("-inf")
        for idx in remaining:
            emb = candidates[idx].get("embedding")
            relevance = (_cosine_sim(query_embedding, emb) if emb and query_embedding
                         else candidates[idx].get("score", 0.0))
            max_sim = 0.0
            for sel_idx in selected:
                sel_emb = candidates[sel_idx].get("embedding")
                cand_emb = candidates[idx].get("embedding")
                if sel_emb and cand_emb:
                    max_sim = max(max_sim, _cosine_sim(cand_emb, sel_emb))
            mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
            if mmr > best_score:
                best_score, best_idx = mmr, idx
        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)

    return [
        {**candidates[i], "score": 1.0 / (1 + rank)}
        for rank, i in enumerate(selected)
    ]


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion.
    RRF(d) = Σ 1 / (k + rank_r(d))
    k=60 từ Cormack et al. 2009.
    """
    rrf_scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            content_map.setdefault(key, item)

    results = []
    for content, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]:
        item = content_map[content].copy()
        item["score"] = score
        results.append(item)
    return results


def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        method: "cross_encoder" (Cohere) | "mmr" | "rrf"
    """
    if not candidates:
        return []
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "mmr":
        return rerank_mmr([], candidates, top_k)
    elif method == "rrf":
        return rerank_rrf([candidates], top_k)
    else:
        raise ValueError(f"Unknown method: {method}")


if __name__ == "__main__":
    dummy = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 0.6, "metadata": {}},
    ]
    results = rerank("hình phạt tàng trữ ma tuý", dummy, top_k=2)
    for r in results:
        print(f"[{r['score']:.4f}] {r['content']}")
