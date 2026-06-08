"""
RAG Chat UI — Vietnamese Drug Law Q&A
Run: streamlit run app.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from src.task10_generation import generate_with_citation

st.set_page_config(
    page_title="RAG Chat · Luật Ma Túy",
    page_icon="⚖️",
    layout="centered",
)

# ── Session state ─────────────────────────────────────────────────────────────
if "conversations" not in st.session_state:
    # List of {id, title, messages}
    st.session_state.conversations = [{"id": 0, "title": "Cuộc trò chuyện 1", "messages": []}]
    st.session_state.active_id = 0
    st.session_state.next_id = 1

def active_conv():
    for c in st.session_state.conversations:
        if c["id"] == st.session_state.active_id:
            return c
    return st.session_state.conversations[0]

def new_conversation():
    cid = st.session_state.next_id
    st.session_state.next_id += 1
    st.session_state.conversations.append({
        "id": cid,
        "title": f"Cuộc trò chuyện {cid + 1}",
        "messages": [],
    })
    st.session_state.active_id = cid

def delete_conversation(cid):
    st.session_state.conversations = [c for c in st.session_state.conversations if c["id"] != cid]
    if not st.session_state.conversations:
        new_conversation()
    elif st.session_state.active_id == cid:
        st.session_state.active_id = st.session_state.conversations[0]["id"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚖️ RAG Chat")

    # New chat button
    if st.button("+ Cuộc trò chuyện mới", use_container_width=True, type="primary"):
        new_conversation()
        st.rerun()

    st.divider()

    # Conversation list
    st.markdown("**Lịch sử**")
    for conv in reversed(st.session_state.conversations):
        is_active = conv["id"] == st.session_state.active_id
        col_title, col_del = st.columns([5, 1])
        label = ("▶ " if is_active else "") + conv["title"]
        if col_title.button(label, key=f"conv_{conv['id']}", use_container_width=True,
                            type="secondary" if not is_active else "primary"):
            st.session_state.active_id = conv["id"]
            st.rerun()
        if col_del.button("✕", key=f"del_{conv['id']}", help="Xoá"):
            delete_conversation(conv["id"])
            st.rerun()

    st.divider()

    # Compact options in an expander
    with st.expander("⚙️ Tuỳ chọn"):
        top_k = st.slider("top_k", min_value=1, max_value=10, value=5,
                          help="Số chunks tìm kiếm")
        show_sources = st.toggle("Hiển thị nguồn", value=True)

    st.caption("Luật PCMT 2021 · BLHS 2015 · NĐ 105/2021 · Tin tức")

# ── Pull active conversation ──────────────────────────────────────────────────
conv = active_conv()
messages = conv["messages"]

# ── Main area header ──────────────────────────────────────────────────────────
st.title("⚖️ RAG Chat · Luật Ma Túy Việt Nam")
st.caption("Hỏi đáp dựa trên văn bản pháp luật và tin tức về ma túy tại Việt Nam — mọi câu trả lời đều có trích dẫn nguồn.")

# ── Render chat history ───────────────────────────────────────────────────────
def render_sources(sources, retrieval_source="hybrid"):
    with st.expander(f"📚 Nguồn tài liệu ({len(sources)} chunks)"):
        for i, src in enumerate(sources, 1):
            meta = src.get("metadata", {})
            score = src.get("score", 0.0)
            st.markdown(
                f"**{i}. {meta.get('source', f'Source {i}')}** · "
                f"`{meta.get('type', 'unknown')}` · score `{score:.4f}` · via `{retrieval_source}`"
            )
            preview = src["content"][:300]
            st.markdown(f"> {preview}{'…' if len(src['content']) > 300 else ''}")
            if i < len(sources):
                st.divider()

for msg in messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and show_sources and msg.get("sources"):
            render_sources(msg["sources"], msg.get("retrieval_source", "hybrid"))

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Nhập câu hỏi về luật ma túy Việt Nam..."):
    messages.append({"role": "user", "content": prompt})

    # Auto-title from first user message
    if conv["title"].startswith("Cuộc trò chuyện"):
        conv["title"] = prompt[:40] + ("…" if len(prompt) > 40 else "")

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Đang tìm kiếm và tổng hợp..."):
            result = generate_with_citation(prompt, top_k=top_k)

        answer = result["answer"]
        sources = result["sources"]
        retrieval_source = result.get("retrieval_source", "hybrid")

        st.markdown(answer)
        if show_sources and sources:
            render_sources(sources, retrieval_source)

    messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
        "retrieval_source": retrieval_source,
    })
