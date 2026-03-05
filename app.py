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
    """
    Load indexes + stores once per app session.
    This is the biggest speed win for Streamlit deployments.
    """
    index, metadata, chunk_store = load_faiss_bundle(PROCESSED_DIR)
    bm25 = joblib.load(str(Path(PROCESSED_DIR) / "bm25.joblib"))
    instructions = build_answer_instructions()
    corpus_id = load_corpus_id(PROCESSED_DIR)
    return index, metadata, chunk_store, bm25, instructions, corpus_id


def list_pdfs(raw_dir: str = RAW_DIR):
    p = Path(raw_dir)
    if not p.exists():
        return []
    return sorted([x for x in p.glob("*.pdf")])


def main():
    st.set_page_config(page_title="Banking RAG Copilot", layout="wide")

    st.title("🏦 Banking RAG Copilot (Enterprise-style)")
    st.caption(
        "Hybrid Retrieval (FAISS + BM25) → Cross-Encoder Rerank → Grounded Answer + Citations + Latency + Audit Logs"
    )

    # ---- API key check (Streamlit secrets first, else env var) ----
    # Your generator expects OPENAI_API_KEY from environment,
    # so we set it from Streamlit secrets if present.
    if "OPENAI_API_KEY" in st.secrets and not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

    if not os.getenv("OPENAI_API_KEY"):
        st.warning("OPENAI_API_KEY is not set. Add it in .streamlit/secrets.toml or environment variables.")
        st.stop()

    # ---- Load runtime objects (cached) ----
    try:
        index, metadata, chunk_store, bm25, instructions, corpus_id = load_runtime()
    except Exception as e:
        st.error(f"Could not load processed artifacts. Did you run build_corpus? Error: {e}")
        st.stop()

    # ---- Sidebar: corpus + PDF download ----
    st.sidebar.header("📚 Corpus")
    st.sidebar.write(f"**Corpus ID:** `{corpus_id}`")

    pdfs = list_pdfs(RAW_DIR)
    if not pdfs:
        st.sidebar.error("No PDFs found in data/raw. Add PDFs and rebuild corpus.")
    else:
        selected_pdf = st.sidebar.selectbox(
            "Select a PDF to download/inspect",
            options=[p.name for p in pdfs],
            index=0,
        )
        sel_path = Path(RAW_DIR) / selected_pdf
        try:
            data = sel_path.read_bytes()
            st.sidebar.download_button(
                "⬇️ Download selected PDF",
                data=data,
                file_name=selected_pdf,
                mime="application/pdf",
            )
        except Exception as e:
            st.sidebar.error(f"Could not read PDF: {e}")

    # ---- Sidebar: retrieval controls ----
    st.sidebar.header("⚙️ Retrieval Settings")
    retrieve_k = st.sidebar.slider("Retrieve K (candidates before rerank)", 5, 50, 15, step=1)
    final_k = st.sidebar.slider("Final K (chunks sent to LLM)", 1, 10, 3, step=1)
    alpha = st.sidebar.slider("Alpha (Dense weight)", 0.0, 1.0, 0.7, step=0.05)

    show_debug = st.sidebar.checkbox("Show debug panels", value=True)
    show_latency = st.sidebar.checkbox("Show latency", value=True)

    # ---- Main: query input (2 lines, no 'press enter to apply') ----
    st.subheader("Ask a compliance question")

    # Persist query so Clear can safely reset it
    if "query" not in st.session_state:
        st.session_state["query"] = ""

    query = st.text_area(
        "Query",
        key="query",
        height=70,  # ~2 lines
        placeholder="e.g., What are the categories under PSL?",
        label_visibility="visible",
    )

    def safe_clear():
        # IMPORTANT: do NOT use st.session_state.clear()
        # It can wipe Streamlit internal keys and throw errors.
        st.session_state["query"] = ""

        # Optional: clear any stored outputs if you ever add them
        for k in ["last_answer", "last_results", "last_candidates", "last_request_id"]:
            if k in st.session_state:
                del st.session_state[k]

        st.rerun()

    colA, colB = st.columns([1, 1])
    with colA:
        run_btn = st.button("Run RAG", type="primary")
    with colB:
        st.button("Clear", on_click=safe_clear)

    # Only run pipeline when user explicitly clicks Run RAG
    if not run_btn:
        st.stop()

    if not query.strip():
        st.error("Please enter a query.")
        st.stop()

    request_id = str(uuid.uuid4())
    timers = Timers()

    try:
        # 1) Hybrid retrieve
        with timers.span("hybrid_ms"):
            candidates = hybrid_search(
                query=query,
                index=index,
                metadata=metadata,
                bm25=bm25,
                top_k=retrieve_k,
                alpha=alpha,
                timers=timers,
            )

        # 2) Rerank
        with timers.span("rerank_ms"):
            results = rerank(
                query=query,
                results=candidates,
                chunk_store=chunk_store,
                top_k=final_k,
            )

        # 3) Build context
        with timers.span("context_ms"):
            context = build_prompt_context(results, chunk_store)

        # 4) Generate answer
        with timers.span("llm_ms"):
            answer = generate_answer(query, instructions, context)

        # 5) Audit trail
        log_audit_event(
            query=query,
            answer=answer,
            results=candidates,          # retrieval transparency
            context_results=results,     # what went into LLM
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

    except Exception as e:
        log_exception(
            query=query,
            stage="streamlit_run",
            error=e,
            meta={"corpus_id": corpus_id},
            request_id=request_id,
        )
        st.error(f"Error while running pipeline: {e}")
        st.stop()

    # ---- Output ----
    st.success("Done")
    st.write(f"**Request ID:** `{request_id}`")

    st.markdown("### ✅ Answer")
    st.write(answer)

    # ---- Evidence / debug panels ----
    if show_debug:
        st.markdown("### 🔎 Evidence (context chunks sent to LLM)")
        for r in results:
            cid = r.get("chunk_id")
            txt = chunk_store.get(cid, "")
            with st.expander(f"{cid} | rerank={r.get('rerank_score', 'NA')} | combined={r.get('score', 'NA')}"):
                st.write(txt[:3000])

        st.markdown("### 🧾 Retrieval candidates (top chunks)")
        st.json(candidates[: min(len(candidates), 20)])

    if show_latency:
        st.markdown("### ⏱️ Latency (ms)")
        stage_keys = ["hybrid_ms", "rerank_ms", "context_ms", "llm_ms"]
        st.write({k: round(timers.ms[k], 3) for k in stage_keys if k in timers.ms})

        st.markdown("#### Hybrid breakdown")
        hybrid_keys = ["embed_ms", "faiss_ms", "bm25_ms", "merge_ms"]
        st.write({k: round(timers.ms[k], 3) for k in hybrid_keys if k in timers.ms})

    st.caption("Audit logs: logs/query_audit.jsonl | Errors: logs/errors.jsonl")


if __name__ == "__main__":
    main()