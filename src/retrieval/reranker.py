# src/retrieval/reranker.py
from __future__ import annotations
from typing import List, Dict, Any

import streamlit as st
from sentence_transformers import CrossEncoder


@st.cache_resource(show_spinner=False)
def get_reranker() -> CrossEncoder:
    # Lazy-load once per app session
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(
    *,
    query: str,
    results: List[Dict[str, Any]],
    chunk_store: Dict[str, str],
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    pairs = []
    kept = []

    for r in results:
        cid = r["chunk_id"]
        txt = chunk_store.get(cid, "")
        if not txt.strip():
            continue
        kept.append(r)
        pairs.append((query, txt[:2000]))  # cap for speed

    if not pairs:
        return []

    reranker = get_reranker()
    scores = reranker.predict(pairs)

    for r, s in zip(kept, scores):
        r["rerank_score"] = float(s)

    kept.sort(key=lambda x: x["rerank_score"], reverse=True)
    return kept[:top_k]