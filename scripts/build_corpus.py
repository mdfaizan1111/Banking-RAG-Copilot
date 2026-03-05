# scripts/build_corpus.py
from __future__ import annotations
from pathlib import Path
import joblib

from src.ingest.pdf_loader import load_corpus_from_raw
from src.ingest.chunker import chunk_documents
from src.index.build_index import build_faiss_index
from src.index.build_bm25 import build_bm25
from src.index.persist import save_faiss_bundle


RAW_DIR = "data/raw"
OUT_DIR = "data/processed"


def main():
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    docs = load_corpus_from_raw(RAW_DIR)
    if not docs:
        raise RuntimeError("No PDFs found in data/raw. Add PDFs first.")

    chunks = chunk_documents(docs)
    if not chunks:
        raise RuntimeError("No chunks created. PDFs might be empty/scanned.")

    # chunk_store: chunk_id -> text
    chunk_store = {c["chunk_id"]: c["text"] for c in chunks}

    # FAISS index + metadata
    index, metadata, _embedder = build_faiss_index(chunks)

    # BM25
    bm25 = build_bm25(chunks)

    # Persist
    save_faiss_bundle(index, metadata, chunk_store, OUT_DIR)
    joblib.dump(bm25, str(Path(OUT_DIR) / "bm25.joblib"))

    # Corpus id
    corpus_id = Path(RAW_DIR).resolve().name
    # better: timestamp-based
    import datetime
    cid = datetime.datetime.now().strftime("%Y-%m-%d_corpus_v1")
    (Path(OUT_DIR) / "corpus_id.txt").write_text(cid, encoding="utf-8")

    print(f"✅ Corpus built: {cid}")
    print(f"- chunks: {len(chunks)}")
    print(f"- saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()