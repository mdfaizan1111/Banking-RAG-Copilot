# src/ingest/pdf_loader.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any
from pypdf import PdfReader


def load_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    parts = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        parts.append(txt)
    return "\n".join(parts)


def load_corpus_from_raw(raw_dir: str = "data/raw") -> List[Dict[str, Any]]:
    raw = Path(raw_dir)
    pdfs = sorted(raw.glob("*.pdf"))
    docs: List[Dict[str, Any]] = []
    for p in pdfs:
        docs.append(
            {
                "file_name": p.name,
                "text": load_pdf_text(p),
            }
        )
    return docs