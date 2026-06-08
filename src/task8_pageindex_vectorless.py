"""
Task 8 — PageIndex Vectorless RAG.

PageIndex dùng structural understanding thay vì embedding để retrieve.
Nếu không có API key, fallback về BM25-based search trên standardized files.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


def upload_documents():
    """Upload toàn bộ markdown documents lên PageIndex."""
    if not PAGEINDEX_API_KEY or PAGEINDEX_API_KEY == "pi_xxx":
        print("⚠ PAGEINDEX_API_KEY not set, skipping upload")
        return

    try:
        from pageindex import PageIndex

        pi = PageIndex(api_key=PAGEINDEX_API_KEY)
        for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
            content = md_file.read_text(encoding="utf-8")
            pi.upload(
                content=content,
                metadata={"filename": md_file.name, "type": md_file.parent.name},
            )
            print(f"  ✓ Uploaded: {md_file.name}")
    except ImportError:
        print("⚠ pageindex package not installed")


def _fallback_search(query: str, top_k: int) -> list[dict]:
    """
    Fallback search dùng BM25 trên standardized files khi không có PageIndex API.
    Trả về kết quả với source='pageindex' để tương thích với pipeline.
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from rank_bm25 import BM25Okapi
    import numpy as np

    docs = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        if not content.strip():
            continue
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        for chunk in splitter.split_text(content):
            docs.append({
                "content": chunk,
                "metadata": {
                    "source": md_file.name,
                    "type": "legal" if "legal" in str(md_file) else "news",
                },
            })

    if not docs:
        return []

    tokenized = [d["content"].lower().split() for d in docs]
    bm25 = BM25Okapi(tokenized)
    scores = bm25.get_scores(query.lower().split())

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append({
                "content": docs[idx]["content"],
                "score": float(scores[idx]),
                "metadata": docs[idx]["metadata"],
                "source": "pageindex",
            })

    return results


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict, 'source': 'pageindex'}
    """
    if not PAGEINDEX_API_KEY or PAGEINDEX_API_KEY == "pi_xxx":
        return _fallback_search(query, top_k)

    try:
        from pageindex import PageIndex

        pi = PageIndex(api_key=PAGEINDEX_API_KEY)
        results = pi.query(query=query, top_k=top_k)
        return [
            {
                "content": r.text,
                "score": r.score,
                "metadata": r.metadata or {},
                "source": "pageindex",
            }
            for r in results
        ]
    except Exception:
        return _fallback_search(query, top_k)


if __name__ == "__main__":
    print("Test PageIndex search:")
    results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
    for r in results:
        print(f"[{r['score']:.3f}] [{r['source']}] {r['content'][:100]}...")
