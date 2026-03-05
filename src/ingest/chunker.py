# src/ingest/chunker.py
from __future__ import annotations
from typing import List, Dict, Any


def simple_chunk(
    text: str,
    chunk_chars: int = 1200,
    overlap_chars: int = 200,
) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_chars, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap_chars
        if start < 0:
            start = 0
        if end == n:
            break

    return chunks


def chunk_documents(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Output items:
    {
      chunk_id: "file.pdf::chunk_12",
      file_name: "file.pdf",
      chunk_index: 12,
      text: "...",
    }
    """
    out: List[Dict[str, Any]] = []
    for d in docs:
        file_name = d["file_name"]
        chunks = simple_chunk(d.get("text", ""))
        for i, ch in enumerate(chunks):
            out.append(
                {
                    "chunk_id": f"{file_name}::chunk_{i}",
                    "file_name": file_name,
                    "chunk_index": i,
                    "text": ch,
                }
            )
    return out