# src/observability/audit.py
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class AuditEvent:
    ts_utc: str
    request_id: str
    query: str
    answer: str
    top_chunks: List[Dict[str, Any]]
    context_chunks: List[Dict[str, Any]]
    latency_ms: Dict[str, float]
    meta: Dict[str, Any]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def ensure_logs_dir(logs_dir: str = "logs") -> Path:
    p = Path(logs_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _compact_chunks(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in results or []:
        out.append(
            {
                "chunk_id": r.get("chunk_id"),
                "file_name": r.get("file_name"),
                "score": r.get("score"),
                "rerank_score": r.get("rerank_score"),
            }
        )
    return out


def log_audit_event(
    *,
    query: str,
    answer: str,
    results: List[Dict[str, Any]],
    latency_ms: Dict[str, float],
    logs_dir: str = "logs",
    file_name: str = "query_audit.jsonl",
    meta: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None,
    context_results: Optional[List[Dict[str, Any]]] = None,
) -> str:
    rid = request_id or str(uuid.uuid4())

    top_chunks = _compact_chunks(results)
    context_chunks = _compact_chunks(context_results if context_results is not None else results)

    evt = AuditEvent(
        ts_utc=_utc_now_iso(),
        request_id=rid,
        query=query,
        answer=answer,
        top_chunks=top_chunks,
        context_chunks=context_chunks,
        latency_ms={k: float(v) for k, v in (latency_ms or {}).items()},
        meta=meta or {},
    )

    logs_path = ensure_logs_dir(logs_dir) / file_name
    append_jsonl(logs_path, asdict(evt))
    return rid


def log_exception(
    *,
    query: str,
    stage: str,
    error: Exception,
    logs_dir: str = "logs",
    file_name: str = "errors.jsonl",
    meta: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None,
) -> str:
    rid = request_id or str(uuid.uuid4())
    logs_path = ensure_logs_dir(logs_dir) / file_name

    payload = {
        "ts_utc": _utc_now_iso(),
        "request_id": rid,
        "query": query,
        "stage": stage,
        "error_type": type(error).__name__,
        "error_msg": str(error),
        "meta": meta or {},
    }
    append_jsonl(logs_path, payload)
    return rid