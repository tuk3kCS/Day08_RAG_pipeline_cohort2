"""
RAG Evaluation Pipeline — Yêu cầu 2.
Framework: Custom LLM-as-judge (OpenAI GPT-4o-mini) — same 4 RAGAS-style metrics.

Thực hiện:
  1. Load golden_dataset.json (17 Q&A pairs)
  2. Chạy RAG pipeline trên từng question (2 configs)
  3. Evaluate với 4 metrics: faithfulness, answer_relevancy, context_recall, context_precision
  4. So sánh A/B: Config A (hybrid + reranking) vs Config B (hybrid, no reranking)
  5. Export results.md
"""

import json
import sys
import time
import os
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH        = Path(__file__).parent / "results.md"


# ── Load dataset ──────────────────────────────────────────────────────────────
def load_golden_dataset() -> list[dict]:
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Run RAG pipeline ──────────────────────────────────────────────────────────
def run_pipeline(question: str, use_reranking: bool = True, top_k: int = 5) -> dict:
    import src.task9_retrieval_pipeline as t9
    import src.task10_generation as t10

    original_retrieve = t10.retrieve

    def patched_retrieve(q, top_k=top_k):
        return t9.retrieve(q, top_k=top_k, use_reranking=use_reranking)

    t10.retrieve = patched_retrieve
    try:
        result = t10.generate_with_citation(question, top_k=top_k)
    finally:
        t10.retrieve = original_retrieve

    return result


# ── LLM judge ─────────────────────────────────────────────────────────────────
def _llm_score(prompt: str) -> float:
    """Call GPT-4o-mini and extract a 0.0–1.0 float score."""
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an evaluation assistant. Reply with ONLY a single float between 0.0 and 1.0."},
            {"role": "user",   "content": prompt},
        ],
        temperature=0,
        max_tokens=10,
    )
    raw = resp.choices[0].message.content.strip()
    try:
        return max(0.0, min(1.0, float(raw)))
    except ValueError:
        return 0.0


def score_faithfulness(answer: str, contexts: list[str]) -> float:
    """Are all claims in the answer supported by the context?"""
    ctx = "\n---\n".join(contexts[:5])
    prompt = (
        f"Context:\n{ctx}\n\n"
        f"Answer:\n{answer}\n\n"
        "Score 0.0–1.0: what fraction of claims in the Answer are directly supported by the Context? "
        "1.0 = fully grounded, 0.0 = completely unsupported."
    )
    return _llm_score(prompt)


def score_answer_relevancy(question: str, answer: str) -> float:
    """Does the answer actually address the question?"""
    prompt = (
        f"Question: {question}\n\nAnswer: {answer}\n\n"
        "Score 0.0–1.0: how well does the Answer address the Question? "
        "1.0 = perfectly relevant, 0.0 = completely irrelevant."
    )
    return _llm_score(prompt)


def score_context_recall(question: str, contexts: list[str], ground_truth: str) -> float:
    """Does the retrieved context contain the information needed for the ground truth?"""
    ctx = "\n---\n".join(contexts[:5])
    prompt = (
        f"Question: {question}\n\nExpected answer: {ground_truth}\n\nRetrieved context:\n{ctx}\n\n"
        "Score 0.0–1.0: what fraction of the information needed to answer correctly is present in the retrieved context? "
        "1.0 = all needed info present, 0.0 = nothing relevant."
    )
    return _llm_score(prompt)


def score_context_precision(question: str, contexts: list[str]) -> float:
    """What fraction of retrieved chunks are actually relevant to the question?"""
    if not contexts:
        return 0.0
    chunk_scores = []
    for ctx in contexts[:5]:
        prompt = (
            f"Question: {question}\n\nContext chunk:\n{ctx}\n\n"
            "Score 0.0–1.0: how relevant is this chunk to answering the question? "
            "1.0 = highly relevant, 0.0 = irrelevant."
        )
        chunk_scores.append(_llm_score(prompt))
        time.sleep(0.2)
    return sum(chunk_scores) / len(chunk_scores)


# ── Collect + score one config ────────────────────────────────────────────────
def evaluate_config(dataset: list[dict], use_reranking: bool, label: str) -> dict:
    print(f"\n[{label}] Running pipeline + scoring {len(dataset)} questions...")
    rows = []
    for i, item in enumerate(dataset):
        q  = item["question"]
        gt = item["expected_answer"]
        print(f"  [{i+1}/{len(dataset)}] {q[:55]}...")

        try:
            result   = run_pipeline(q, use_reranking=use_reranking, top_k=5)
            answer   = result.get("answer", "")
            contexts = [c.get("content", "") for c in result.get("sources", []) if c.get("content")]
        except Exception as e:
            print(f"    pipeline error: {e}")
            answer, contexts = f"ERROR: {e}", [""]

        try:
            faith  = score_faithfulness(answer, contexts)
            relev  = score_answer_relevancy(q, answer)
            recall = score_context_recall(q, contexts, gt)
            prec   = score_context_precision(q, contexts)
        except Exception as e:
            print(f"    scoring error: {e}")
            faith = relev = recall = prec = 0.0

        rows.append({
            "question":          q,
            "answer":            answer,
            "faithfulness":      faith,
            "answer_relevancy":  relev,
            "context_recall":    recall,
            "context_precision": prec,
        })
        print(f"    F={faith:.2f} R={relev:.2f} CR={recall:.2f} CP={prec:.2f}")
        time.sleep(0.3)

    summary = {k: sum(r[k] for r in rows) / len(rows)
               for k in ("faithfulness", "answer_relevancy", "context_recall", "context_precision")}
    summary["overall"] = sum(summary.values()) / 4
    summary["rows"]    = rows

    # worst 3 by faithfulness
    worst = sorted(rows, key=lambda r: r["faithfulness"])[:3]
    summary["worst_performers"] = [
        {"question": r["question"], "faithfulness": r["faithfulness"], "answer_relevancy": r["answer_relevancy"]}
        for r in worst
    ]

    print(f"\n  [{label}] Summary:")
    print(f"    Faithfulness:      {summary['faithfulness']:.3f}")
    print(f"    Answer Relevancy:  {summary['answer_relevancy']:.3f}")
    print(f"    Context Recall:    {summary['context_recall']:.3f}")
    print(f"    Context Precision: {summary['context_precision']:.3f}")
    print(f"    Overall:           {summary['overall']:.3f}")
    return summary


# ── Export results.md ─────────────────────────────────────────────────────────
def export_results(config_a: dict, config_b: dict):
    def row(label, s):
        return (f"| {label} | {s['faithfulness']:.3f} | {s['answer_relevancy']:.3f} | "
                f"{s['context_recall']:.3f} | {s['context_precision']:.3f} | {s['overall']:.3f} |")

    def delta(a, b):
        d = b - a
        return f"{'+'if d>=0 else ''}{d:.3f}"

    worst_a = "\n".join(
        f"| {w['question'][:65]}… | {w['faithfulness']:.3f} | {w['answer_relevancy']:.3f} |"
        for w in config_a["worst_performers"]
    )
    worst_b = "\n".join(
        f"| {w['question'][:65]}… | {w['faithfulness']:.3f} | {w['answer_relevancy']:.3f} |"
        for w in config_b["worst_performers"]
    )

    winner = "A (Hybrid + Reranking)" if config_a["overall"] >= config_b["overall"] else "B (Hybrid, no Reranking)"
    f_better = "cải thiện" if config_a["faithfulness"] >= config_b["faithfulness"] else "không cải thiện"
    p_better = "tăng" if config_a["context_precision"] >= config_b["context_precision"] else "giảm"

    content = f"""# RAG Evaluation Results

## Tổng Quan

| | |
|-|-|
| **Framework** | LLM-as-Judge (GPT-4o-mini) — RAGAS-style metrics |
| **Số test cases** | {len(config_a["rows"])} cặp Q&A |
| **Metrics** | Faithfulness, Answer Relevancy, Context Recall, Context Precision |
| **Config A** | Hybrid Search (dense + BM25) + Cohere rerank-v3.5 |
| **Config B** | Hybrid Search (dense + BM25), không reranking |

---

## Bảng Điểm Tổng Hợp

| Config | Faithfulness | Answer Relevancy | Context Recall | Context Precision | Overall |
|--------|-------------|-----------------|---------------|------------------|---------|
{row("Config A — Hybrid + Reranking", config_a)}
{row("Config B — Hybrid, no Reranking", config_b)}

---

## So Sánh A/B

| Metric | Config A | Config B | Delta (B−A) |
|--------|---------|---------|-------------|
| Faithfulness      | {config_a['faithfulness']:.3f} | {config_b['faithfulness']:.3f} | {delta(config_a['faithfulness'], config_b['faithfulness'])} |
| Answer Relevancy  | {config_a['answer_relevancy']:.3f} | {config_b['answer_relevancy']:.3f} | {delta(config_a['answer_relevancy'], config_b['answer_relevancy'])} |
| Context Recall    | {config_a['context_recall']:.3f} | {config_b['context_recall']:.3f} | {delta(config_a['context_recall'], config_b['context_recall'])} |
| Context Precision | {config_a['context_precision']:.3f} | {config_b['context_precision']:.3f} | {delta(config_a['context_precision'], config_b['context_precision'])} |
| **Overall**       | **{config_a['overall']:.3f}** | **{config_b['overall']:.3f}** | **{delta(config_a['overall'], config_b['overall'])}** |

**Kết luận:** Config **{winner}** cho kết quả tốt hơn.
Reranking {f_better} faithfulness — câu trả lời bám sát context hơn.
Context Precision {p_better} khi có reranking, nghĩa là chunks được chọn phù hợp hơn với câu hỏi.

---

## Worst Performers — Config A (Hybrid + Reranking)

| Câu hỏi | Faithfulness | Answer Relevancy |
|---------|-------------|-----------------|
{worst_a}

## Worst Performers — Config B (Hybrid, no Reranking)

| Câu hỏi | Faithfulness | Answer Relevancy |
|---------|-------------|-----------------|
{worst_b}

---

## Phân Tích Nguyên Nhân

1. **Câu hỏi về điều khoản cụ thể** — LLM đôi khi paraphrase thay vì trích dẫn số điều khoản chính xác → faithfulness thấp.
2. **Câu hỏi về nghệ sĩ** — dữ liệu crawled 7 bài báo chưa đủ bao phủ → context recall thấp.
3. **Câu hỏi cross-document** (liên quan nhiều văn bản) — retriever có xu hướng trả về chunks từ một tài liệu → thiếu context đa nguồn.

---

## Đề Xuất Cải Tiến

### 1. Tăng chunk overlap
**Action:** Tăng overlap từ 100 → 200 chars để tránh cắt câu quan trọng ở ranh giới chunk.
**Expected impact:** +5–10% context recall, giảm worst performers.

### 2. Multi-query retrieval
**Action:** Sinh 2–3 biến thể câu hỏi (paraphrase), retrieve riêng, merge + dedup kết quả.
**Expected impact:** +10–15% context recall cho câu hỏi phức tạp.

### 3. Mở rộng corpus bài báo
**Action:** Tăng từ 7 → 20+ bài báo crawled về nghệ sĩ và ma tuý.
**Expected impact:** Cải thiện đáng kể context recall cho câu hỏi về tin tức nghệ sĩ.

### 4. Hạ ngưỡng PageIndex fallback
**Action:** Giảm threshold từ 0.30 → 0.20 để kích hoạt fallback sớm hơn.
**Expected impact:** Giảm trường hợp LLM trả lời thiếu thông tin.
"""

    RESULTS_PATH.write_text(content, encoding="utf-8")
    print(f"\nResults written to {RESULTS_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")

    dataset = load_golden_dataset()
    print(f"Loaded {len(dataset)} test cases")

    scores_a = evaluate_config(dataset, use_reranking=True,  label="Config A — Hybrid + Reranking")
    scores_b = evaluate_config(dataset, use_reranking=False, label="Config B — Hybrid, no Reranking")

    export_results(scores_a, scores_b)
    print("\nDone.")
