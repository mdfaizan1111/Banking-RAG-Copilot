# src/retrieval/context_builder.py
from __future__ import annotations
from typing import List, Dict


def build_prompt_context(
    results: List[Dict],
    chunk_store: Dict[str, str],
    max_chars_per_chunk: int = 1200,
) -> str:
    blocks = []
    for r in results:
        chunk_id = r["chunk_id"]
        file_name = r.get("file_name", "unknown")

        text = chunk_store.get(chunk_id, "")
        text = text.strip().replace("\n\n\n", "\n\n")
        text = text[:max_chars_per_chunk]

        blocks.append(f"[SOURCE: {file_name} | {chunk_id}]\n{text}")

    return "\n\n---\n\n".join(blocks)


def build_answer_instructions() -> str:
    return (
        "You are a compliance assistant. Answer using ONLY the provided context.\n"
        "If the answer is not in the context, say: 'Not found in the provided documents.'\n"
        "Cite sources at the end of each sentence EXACTLY like [file.pdf::chunk_12].\n"
        "Be concise and precise.\n"
    )