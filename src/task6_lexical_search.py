"""
Task 6 — Lexical Search Module.

Sử dụng Weaviate BM25 built-in (BONUS: +5 điểm).

Weaviate BM25 hoạt động thế nào:
    - Weaviate tự duy trì inverted index trên trường 'content' khi index data.
    - Khi query với with_bm25(), Weaviate tính BM25 score nội bộ (k1=1.2, b=0.75
      theo Okapi BM25) cho từng document trong collection.
    - Không cần load corpus vào memory — tận dụng Weaviate storage engine.
    - Hỗ trợ tokenization cấu hình được (word/lowercase/whitespace/trigram).
    - Score trả về là điểm BM25 tuyệt đối (không chuẩn hoá về [0,1]).
    - Ưu điểm so với rank-bm25: scalable hơn, không cần rebuild index mỗi session.
"""

import os

from dotenv import load_dotenv

load_dotenv()

from .task4_chunking_indexing import WEAVIATE_URL, COLLECTION_NAME


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa dùng Weaviate BM25 built-in.

    Args:
        query: Câu truy vấn
        top_k: Số kết quả tối đa

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict}
        Sorted by score descending.
    """
    import weaviate

    client = weaviate.Client(WEAVIATE_URL)

    result = (
        client.query
        .get(COLLECTION_NAME, ["content", "source", "doc_type", "chunk_index"])
        .with_bm25(query=query, properties=["content"])
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
        })

    output.sort(key=lambda x: x["score"], reverse=True)
    return output


if __name__ == "__main__":
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
