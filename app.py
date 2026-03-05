import os
import uuid
from pathlib import Path

import joblib
import streamlit as st

from src.index.persist import load_faiss_bundle
from src.retrieval.hybrid_search import hybrid_search
from src.retrieval.reranker import rerank
from src.retrieval.context_builder import build_prompt_context, build_answer_instructions
from src.retrieval.generator import generate_answer
from src.observability.timers import Timers
from src.observability.audit import log_audit_event, log_exception

PROCESSED_DIR = "data/processed"
RAW_DIR = "data/raw"


def load_corpus_id(processed_dir: str = PROCESSED_DIR) -> str:
    p = Path(processed_dir) / "corpus_id.txt"
    if p.exists():
        return p.read_text(encoding="utf-8").strip() or "unknown"
    return "unknown"


@st.cache_resource
def load_runtime():
    """Load indexes + stores once per app session."""
    index, metadata, chunk_store = load_faiss_bundle(PROCESSED_DIR)
    bm25 = joblib.load(f"{PROCESSED_DIR}/bm25.joblib")
    instructions = build_answer_instructions()
    corpus_id = load_corpus_id(PROCESSED_DIR)
    return index, metadata, chunk_store, bm25, instructions, corpus_id


@st.cache_data(show_spinner=False)
def cached_pdf_names(raw_dir: str = RAW_DIR):
    p = Path(raw_dir)
    if not p.exists():
        return []
    return [x.name for x in sorted(p.glob("*.pdf"))]


@st.cache_data(show_spinner=False)
def cached_pdf_bytes(raw_dir: str, filename: str) -> bytes:
    return (Path(raw_dir) / filename).read_bytes()


def init_state():
    defaults = {
        "query": "",
        "last_request_id": None,
        "last_query": None,
        "last_answer": None,
        "last_results": None,
        "last_candidates": None,
        "last_latency_ms": None,
        "last_error": None,
        "is_running": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def run_pipeline(
    *,
    q: str,
    index,
    metadata,
    chunk_store,
    bm25,
    instructions,
    corpus_id: str,
    retrieve_k: int,
    final_k: int,
    alpha: float,
):
    request_id = str(uuid.uuid4())
    timers = Timers()

    with timers.span("hybrid_ms"):
        candidates = hybrid_search(
            query=q,
            index=index,
            metadata=metadata,
            bm25=bm25,
            top_k=retrieve_k,
            alpha=alpha,
            timers=timers,
        )

    with timers.span("rerank_ms"):
        results = rerank(
            query=q,
            results=candidates,
            chunk_store=chunk_store,
            top_k=final_k,
        )

    with timers.span("context_ms"):
        context = build_prompt_context(results, chunk_store)

    with timers.span("llm_ms"):
        answer = generate_answer(q, instructions, context)

    log_audit_event(
        query=q,
        answer=answer,
        results=candidates,
        context_results=results,
        latency_ms=timers.ms,
        meta={
            "mode": "hybrid_rerank",
            "retrieve_k": retrieve_k,
            "final_k": final_k,
            "alpha": alpha,
            "corpus_id": corpus_id,
        },
        request_id=request_id,
    )

    return request_id, answer, results, candidates, dict(timers.ms)


def main():
    st.set_page_config(page_title="Banking RAG Copilot", layout="wide")
    init_state()

    st.title("🏦 Banking RAG Copilot (Enterprise-style)")
    st.caption(
        "Hybrid Retrieval (FAISS + BM25) → Cross-Encoder Rerank → Grounded Answer + Citations + Latency + Audit Logs"
    )

    # Key from secrets/env
    if "OPENAI_API_KEY" in st.secrets and not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

    if not os.getenv("OPENAI_API_KEY"):
        st.warning("OPENAI_API_KEY is not set. Add it in .streamlit/secrets.toml or environment variables.")
        st.stop()

    # Load runtime (cached)
    try:
        index, metadata, chunk_store, bm25, instructions, corpus_id = load_runtime()
    except Exception as e:
        st.error(f"Could not load processed artifacts. Did you run build_corpus? Error: {e}")
        st.stop()

    # ---------------- Sidebar ----------------
    st.sidebar.header("📚 Corpus")
    st.sidebar.write(f"**Corpus ID:** `{corpus_id}`")

    st.sidebar.header("⚙️ Retrieval Settings")
    retrieve_k = st.sidebar.slider("Retrieve K (candidates before rerank)", 5, 50, 15, step=1)
    final_k = st.sidebar.slider("Final K (chunks sent to LLM)", 1, 10, 3, step=1)
    alpha = st.sidebar.slider("Alpha (Dense weight)", 0.0, 1.0, 0.7, step=0.05)

    show_debug = st.sidebar.checkbox("Show debug panels", value=True)
    show_latency = st.sidebar.checkbox("Show latency", value=True)

    # OPTIONAL PDF tools moved behind expander so they don't slow every rerun
    with st.sidebar.expander("📄 PDF download (optional)", expanded=False):
        pdf_names = cached_pdf_names(RAW_DIR)
        if not pdf_names:
            st.error("No PDFs found in data/raw. Add PDFs and rebuild corpus.")
        else:
            selected_pdf = st.selectbox("Select a PDF", options=pdf_names, index=0)
            try:
                data = cached_pdf_bytes(RAW_DIR, selected_pdf)
                st.download_button(
                    "⬇️ Download selected PDF",
                    data=data,
                    file_name=selected_pdf,
                    mime="application/pdf",
                )
            except Exception as e:
                st.error(f"Could not read PDF: {e}")

    # ---------------- Main ----------------
    st.subheader("Please enter your question here")

    # NOTE: typing will still rerun, but now reruns are cheap, so flicker reduces drastically.
    with st.form("rag_form", clear_on_submit=False):
        st.text_input(
            "Query",
            key="query",
            placeholder="e.g., What are the categories under PSL?",
            disabled=st.session_state.get("is_running", False),
        )
        run_btn = st.form_submit_button("Run RAG", type="primary", disabled=st.session_state.get("is_running", False))

    if run_btn:
        q = (st.session_state.get("query") or "").strip()

        if not q:
            st.session_state["last_error"] = "Please enter a query."
        else:
            # Mark running + clear old answer immediately (prevents “old answer still shown”)
            st.session_state["is_running"] = True
            st.session_state["last_error"] = None
            st.session_state["last_answer"] = None
            st.session_state["last_results"] = None
            st.session_state["last_candidates"] = None
            st.session_state["last_latency_ms"] = None
            st.session_state["last_query"] = q

            try:
                req_id, answer, results, candidates, latency_ms = run_pipeline(
                    q=q,
                    index=index,
                    metadata=metadata,
                    chunk_store=chunk_store,
                    bm25=bm25,
                    instructions=instructions,
                    corpus_id=corpus_id,
                    retrieve_k=retrieve_k,
                    final_k=final_k,
                    alpha=alpha,
                )

                st.session_state["last_request_id"] = req_id
                st.session_state["last_answer"] = answer
                st.session_state["last_results"] = results
                st.session_state["last_candidates"] = candidates
                st.session_state["last_latency_ms"] = latency_ms
                st.session_state["last_error"] = None

            except Exception as e:
                log_exception(
                    query=q,
                    stage="streamlit_run",
                    error=e,
                    meta={"corpus_id": corpus_id},
                    request_id=str(uuid.uuid4()),
                )
                st.session_state["last_error"] = f"Error while running pipeline: {e}"

            finally:
                st.session_state["is_running"] = False

    # ---------------- Output ----------------
    if st.session_state.get("last_error"):
        st.error(st.session_state["last_error"])

    if st.session_state.get("last_answer"):
        st.success("Done")
        st.write(f"**Request ID:** `{st.session_state['last_request_id']}`")
        st.write(f"**Query:** {st.session_state.get('last_query','')}")

        st.markdown("### ✅ Answer")
        st.write(st.session_state["last_answer"])

        results = st.session_state.get("last_results") or []
        candidates = st.session_state.get("last_candidates") or []
        latency_ms = st.session_state.get("last_latency_ms") or {}

        if show_debug:
            st.markdown("### 🔎 Evidence (context chunks sent to LLM)")
            for r in results:
                cid = r.get("chunk_id")
                txt = chunk_store.get(cid, "")
                with st.expander(
                    f"{cid} | rerank={r.get('rerank_score', 'NA')} | combined={r.get('score', 'NA')}"
                ):
                    st.write(txt[:3000])

            st.markdown("### 🧾 Retrieval candidates (top chunks)")
            st.json(candidates[: min(len(candidates), 20)])

        if show_latency:
            st.markdown("### ⏱️ Latency (ms)")
            stage_keys = ["hybrid_ms", "rerank_ms", "context_ms", "llm_ms"]
            st.write({k: round(latency_ms[k], 3) for k in stage_keys if k in latency_ms})

            st.markdown("#### Hybrid breakdown")
            hybrid_keys = ["embed_ms", "faiss_ms", "bm25_ms", "merge_ms"]
            st.write({k: round(latency_ms[k], 3) for k in hybrid_keys if k in latency_ms})


if __name__ == "__main__":
    main()