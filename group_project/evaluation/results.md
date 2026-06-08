# RAG Evaluation Results

## Tổng Quan

| | |
|-|-|
| **Framework** | LLM-as-Judge (GPT-4o-mini) — RAGAS-style metrics |
| **Số test cases** | 17 cặp Q&A |
| **Metrics** | Faithfulness, Answer Relevancy, Context Recall, Context Precision |
| **Config A** | Hybrid Search (dense + BM25) + Cohere rerank-v3.5 |
| **Config B** | Hybrid Search (dense + BM25), không reranking |

---

## Bảng Điểm Tổng Hợp

| Config | Faithfulness | Answer Relevancy | Context Recall | Context Precision | Overall |
|--------|-------------|-----------------|---------------|------------------|---------|
| Config A — Hybrid + Reranking | 0.929 | 0.906 | 0.706 | 0.671 | 0.803 |
| Config B — Hybrid, no Reranking | 0.647 | 0.671 | 0.441 | 0.460 | 0.555 |

---

## So Sánh A/B

| Metric | Config A | Config B | Delta (B−A) |
|--------|---------|---------|-------------|
| Faithfulness      | 0.929 | 0.647 | -0.282 |
| Answer Relevancy  | 0.906 | 0.671 | -0.235 |
| Context Recall    | 0.706 | 0.441 | -0.265 |
| Context Precision | 0.671 | 0.460 | -0.211 |
| **Overall**       | **0.803** | **0.555** | **-0.248** |

**Kết luận:** Config **A (Hybrid + Reranking)** cho kết quả tốt hơn.
Reranking cải thiện faithfulness — câu trả lời bám sát context hơn.
Context Precision tăng khi có reranking, nghĩa là chunks được chọn phù hợp hơn với câu hỏi.

---

## Worst Performers — Config A (Hybrid + Reranking)

| Câu hỏi | Faithfulness | Answer Relevancy |
|---------|-------------|-----------------|
| Nghệ sĩ nào bị bắt vì liên quan đến ma tuý theo các bài báo đã th… | 0.000 | 0.200 |
| Các tội phạm về ma tuý nào thuộc Chương XX Bộ luật Hình sự 2015?… | 0.800 | 1.000 |
| Hình phạt cho tội tàng trữ trái phép chất ma tuý theo Điều 249 Bộ… | 1.000 | 1.000 |

## Worst Performers — Config B (Hybrid, no Reranking)

| Câu hỏi | Faithfulness | Answer Relevancy |
|---------|-------------|-----------------|
| Danh mục các chất ma tuý thuộc nhóm I theo quy định pháp luật Việ… | 0.000 | 0.000 |
| Điều kiện để một người được coi là nghiện ma tuý theo Luật Phòng … | 0.000 | 0.000 |
| Các tội phạm về ma tuý nào thuộc Chương XX Bộ luật Hình sự 2015?… | 0.000 | 0.200 |

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
