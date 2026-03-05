# src/index/persist.py
from __future__ import annotations
from pathlib import Path
import json
import joblib
import faiss


def save_faiss_bundle(index, metadata, chunk_store, out_dir: str = "data/processed"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(out / "faiss.index"))

    with open(out / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    joblib.dump(chunk_store, out / "chunk_store.joblib")


def load_faiss_bundle(in_dir: str = "data/processed"):
    inp = Path(in_dir)

    index = faiss.read_index(str(inp / "faiss.index"))

    with open(inp / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    chunk_store = joblib.load(inp / "chunk_store.joblib")

    return index, metadata, chunk_store