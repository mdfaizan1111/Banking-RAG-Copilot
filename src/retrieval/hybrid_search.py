# src/retrieval/hybrid_search.py
from __future__ import annotations
from typing import List, Dict, Any, Optional
import numpy as np

from src.index.embedder import Embedder
from src.index.build_bm25 import bm25_scores
from src.observability.timers import Timers


# We keep a global embedder instance for speed (same model used in indexing)
_EMBEDDER: Optional[Embedder] = None


def _get_embedder() -> Embedder:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = Embedder("sentence-transformers/all-MiniLM-L6-v2")
    return _EMBEDDER


def hybrid_search(
    *,
    query: str,
    index,
    metadata: List[Dict[str, Any]],
    bm25,
    top_k: int = 15,
    alpha: float = 0.7,
    timers: Optional[Timers] = None,
) -> List[Dict[str, Any]]:
    """
    Returns top_k candidates after combining:
    - dense score from FAISS (cosine/IP)
    - sparse score from BM25
    """
    t = timers

    # 1) Embed query
    if t:
        with t.span("embed_ms"):
            q = _get_embedder().embed([query])  # (1, D)
    else:
        q = _get_embedder().embed([query])

    # 2) Dense search
    if t:
        with t.span("faiss_ms"):
            D, I = index.search(q, top_k)
    else:
        D, I = index.search(q, top_k)

    dense_hits = []
    for score, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx < 0:
            continue
        m = metadata[idx]
        dense_hits.append(
            {
                "chunk_id": m["chunk_id"],
                "file_name": m["file_name"],
                "score_dense": float(score),
                "faiss_id": int(idx),
            }
        )

    # 3) BM25 for ALL docs (simple but fine for your small corpus)
    if t:
        with t.span("bm25_ms"):
            sparse_all = bm25_scores(bm25, query)  # len = N
    else:
        sparse_all = bm25_scores(bm25, query)

    # normalize sparse to 0..1 (avoid division by zero)
    s = np.array(sparse_all, dtype="float32")
    s_min, s_max = float(s.min()), float(s.max())
    if s_max - s_min < 1e-9:
        s_norm = np.zeros_like(s)
    else:
        s_norm = (s - s_min) / (s_max - s_min)

    # merge: take union of dense hits + top sparse hits
    if t:
        with t.span("merge_ms"):
            out = _merge_scores(dense_hits, metadata, s_norm, top_k, alpha)
    else:
        out = _merge_scores(dense_hits, metadata, s_norm, top_k, alpha)

    # sort by combined score desc
    out.sort(key=lambda r: r["score"], reverse=True)
    return out[:top_k]


def _merge_scores(dense_hits, metadata, s_norm, top_k, alpha):
    # pick top sparse ids too
    top_sparse_ids = np.argsort(-s_norm)[:top_k].tolist()

    by_chunk = {}

    # seed with dense
    for h in dense_hits:
        cid = h["chunk_id"]
        by_chunk[cid] = {
            "chunk_id": cid,
            "file_name": h["file_name"],
            "score_dense": h["score_dense"],
            "score_sparse": float(s_norm[h["faiss_id"]]),
        }

    # add sparse
    for idx in top_sparse_ids:
        m = metadata[idx]
        cid = m["chunk_id"]
        if cid not in by_chunk:
            by_chunk[cid] = {
                "chunk_id": cid,
                "file_name": m["file_name"],
                "score_dense": 0.0,
                "score_sparse": float(s_norm[idx]),
            }

    # combined
    out = []
    for v in by_chunk.values():
        v["score"] = float(alpha * v["score_dense"] + (1.0 - alpha) * v["score_sparse"])
        out.append(v)
    return out