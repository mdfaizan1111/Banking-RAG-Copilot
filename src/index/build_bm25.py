# src/index/build_bm25.py
from __future__ import annotations
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    return (text or "").lower().split()


def build_bm25(chunks: List[Dict[str, Any]]) -> BM25Okapi:
    corpus_tokens = [_tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(corpus_tokens)
    return bm25


def bm25_scores(bm25: BM25Okapi, query: str) -> list[float]:
    q_tokens = _tokenize(query)
    scores = bm25.get_scores(q_tokens)
    return list(scores)