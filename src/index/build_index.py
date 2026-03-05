# src/index/build_index.py
from __future__ import annotations
from typing import List, Dict, Any, Tuple
import faiss
import numpy as np
from .embedder import Embedder


def build_faiss_index(chunks: List[Dict[str, Any]], model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    texts = [c["text"] for c in chunks]
    embedder = Embedder(model_name=model_name)
    X = embedder.embed(texts)  # (N, D) float32

    d = X.shape[1]
    index = faiss.IndexFlatIP(d)  # cosine because normalized embeddings
    index.add(X)

    # metadata aligned to FAISS ids 0..N-1
    metadata = []
    for i, c in enumerate(chunks):
        metadata.append(
            {
                "faiss_id": i,
                "chunk_id": c["chunk_id"],
                "file_name": c["file_name"],
            }
        )

    return index, metadata, embedder