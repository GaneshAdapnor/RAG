"""
Streamlit frontend for the Production RAG QA system.

Works in two modes — auto-detected at startup:

  STANDALONE (Streamlit Cloud / single process):
      Calls service functions directly (embedding, FAISS, OpenAI).
      OPENAI_API_KEY is read from st.secrets or os.environ.
      No FastAPI server required.

  API MODE (local dev with both servers running):
      Set STREAMLIT_API_BASE=http://localhost:8000 in the environment
      or .streamlit/secrets.toml to route calls through the FastAPI backend.

Run locally (standalone):
    streamlit run streamlit_app.py

Run locally (API mode):
    # Terminal 1: uvicorn app.main:app --reload --port 8000
    # Terminal 2: STREAMLIT_API_BASE=http://localhost:8000 streamlit run streamlit_app.py

Deploy to Streamlit Cloud:
    Push this repo to GitHub, connect it in https://share.streamlit.io,
    and add OPENAI_API_KEY to the app's Secrets panel.
"""

import os
import sys

import streamlit as st

# ---------------------------------------------------------------------------
# Secrets → environment (must happen before any service imports)
# ---------------------------------------------------------------------------
# Streamlit Cloud stores secrets in st.secrets. We mirror them into os.environ
# so that pydantic-settings (config.py) picks them up without code changes.

for key in ("OPENAI_API_KEY", "LLM_MODEL_NAME", "EMBEDDING_MODEL_NAME",
            "CHUNK_SIZE", "CHUNK_OVERLAP", "TOP_K_RETRIEVAL"):
    if key in st.secrets:
        os.environ.setdefault(key, str(st.secrets[key]))

# ---------------------------------------------------------------------------
# Mode detection
# ---------------------------------------------------------------------------

API_BASE = os.environ.get("STREAMLIT_API_BASE", st.secrets.get("STREAMLIT_API_BASE", ""))
USE_API_MODE = bool(API_BASE)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RAG Document QA",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Service imports (standalone mode only)
# ---------------------------------------------------------------------------

if not USE_API_MODE:
    # Add project root to path so 'app' package is importable
    sys.path.insert(0, os.path.dirname(__file__))
    try:
        from app.core.logging_config import setup_logging
        from app.core.config import get_settings
        from app.core.dependencies import get_vector_store, get_embedding_service, get_query_service
        from app.models.api import QueryRequest
        from app.services.ingestion_service import (
            ingest_document, get_job, create_doc_id, validate_upload,
        )

        setup_logging("WARNING")  # Keep logs quiet in the UI process

        _services_ok = True
    except Exception as e:
        st.error(f"Failed to load service modules: {e}")
        st.stop()
        _services_ok = False
else:
    import requests
    _services_ok = True

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "documents" not in st.session_state:
    st.session_state.documents = []   # [{doc_id, filename, status, chunk_count, error}]

if "query_history" not in st.session_state:
    st.session_state.query_history = []

if "selected_doc_ids" not in st.session_state:
    st.session_state.selected_doc_ids = []

if "store_loaded" not in st.session_state:
    st.session_state.store_loaded = False

# ---------------------------------------------------------------------------
# One-time startup: load FAISS index (standalone mode)
# ---------------------------------------------------------------------------

if not USE_API_MODE and not st.session_state.store_loaded:
    with st.spinner("Loading vector store…"):
        get_vector_store()       # lru_cache: loads index from disk on first call
        get_embedding_service()  # warm-up: downloads model if not cached
    st.session_state.store_loaded = True

# ---------------------------------------------------------------------------
# Backend helpers (unified interface regardless of mode)
# ---------------------------------------------------------------------------

def backend_health() -> dict | None:
    if USE_API_MODE:
        try:
            r = requests.get(f"{API_BASE}/health", timeout=5)
            return r.json() if r.status_code == 200 else None
        except Exception:
            return None
    else:
        stats = get_vector_store().stats()
        return {
            "status": "ok",
            "indexed_chunks": stats["indexed_chunks"],
            "indexed_documents": stats["indexed_documents"],
            "embedding_model": get_settings().embedding_model_name,
        }


def backend_upload(file_bytes: bytes, filename: str, content_type: str) -> dict | None:
    if USE_API_MODE:
        try:
            r = requests.post(
                f"{API_BASE}/upload",
                files={"file": (filename, file_bytes, content_type)},
                timeout=30,
            )
            if r.status_code == 202:
                return r.json()
            st.error(f"Upload failed ({r.status_code}): {r.json().get('detail', r.text)}")
            return None
        except requests.ConnectionError:
            st.error("Cannot connect to the API backend.")
            return None
    else:
        try:
            validate_upload(filename, len(file_bytes), content_type)
        except ValueError as e:
            st.error(str(e))
            return None

        doc_id = create_doc_id()
        return {"doc_id": doc_id, "filename": filename, "status": "pending"}


def backend_ingest_sync(doc_id: str, filename: str, file_bytes: bytes, content_type: str):
    """Run ingestion synchronously (standalone mode — no background threads in Streamlit)."""
    ingest_document(doc_id, filename, file_bytes, content_type)


def backend_status(doc_id: str) -> dict | None:
    if USE_API_MODE:
        try:
            r = requests.get(f"{API_BASE}/upload/status/{doc_id}", timeout=10)
            return r.json() if r.status_code == 200 else None
        except Exception:
            return None
    else:
        job = get_job(doc_id)
        if not job:
            return None
        return {
            "doc_id": job.doc_id,
            "filename": job.filename,
            "status": job.status.value,
            "chunk_count": job.chunk_count,
            "error": job.error,
        }


def backend_query(query: str, top_k: int, doc_ids: list[str] | None) -> dict | None:
    if USE_API_MODE:
        payload = {"query": query, "top_k": top_k}
        if doc_ids:
            payload["doc_ids"] = doc_ids
        try:
            r = requests.post(f"{API_BASE}/query", json=payload, timeout=60)
            if r.status_code == 200:
                return r.json()
            st.error(f"Query failed ({r.status_code}): {r.json().get('detail', r.text)}")
            return None
        except requests.ConnectionError:
            st.error("Cannot connect to the API backend.")
            return None
    else:
        try:
            result = get_query_service().answer(
                QueryRequest(question=query, top_k=top_k, document_ids=doc_ids)
            )
        except Exception as e:
            st.error(str(e))
            return None

        return {
            "query": result.question,
            "answer": result.answer,
            "retrieval_latency_ms": 0.0,
            "generation_latency_ms": result.latency_ms,
            "sources": [
                {
                    "doc_id": s.document_id,
                    "filename": s.filename,
                    "chunk_id": s.chunk_id,
                    "page": s.page_start,
                    "text": s.excerpt,
                    "score": s.score,
                }
                for s in result.sources
            ],
        }

# ---------------------------------------------------------------------------
# Status badge
# ---------------------------------------------------------------------------

STATUS_ICON = {
    "pending": "🕐", "processing": "⚙️", "completed": "✅", "failed": "❌",
}

# ---------------------------------------------------------------------------
# Sidebar — health + upload
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("📚 RAG Document QA")
    mode_label = "API mode" if USE_API_MODE else "Standalone mode"
    st.caption(f"Running in **{mode_label}**")
    st.divider()

    health = backend_health()
    if health:
        st.success(
            f"Ready — {health['indexed_chunks']} chunks / "
            f"{health['indexed_documents']} docs indexed"
        )
    else:
        st.error("Backend unreachable.")

    st.divider()

    # --- File upload ---
    st.subheader("Upload Documents")

    if not os.environ.get("OPENAI_API_KEY") and not st.secrets.get("OPENAI_API_KEY"):
        st.warning("OPENAI_API_KEY not set. Add it to `.streamlit/secrets.toml` or Streamlit Cloud secrets.")

    uploaded_files = st.file_uploader(
        "PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        help="Files are processed immediately (standalone) or in the background (API mode).",
    )

    if uploaded_files:
        existing_names = {d["filename"] for d in st.session_state.documents}
        new_files = [f for f in uploaded_files if f.name not in existing_names]

        for uf in new_files:
            content_type = "application/pdf" if uf.name.endswith(".pdf") else "text/plain"
            file_bytes = uf.read()

            result = backend_upload(file_bytes, uf.name, content_type)
            if not result:
                continue

            doc_id = result["doc_id"]
            st.session_state.documents.append({
                "doc_id": doc_id,
                "filename": uf.name,
                "status": "processing",
                "chunk_count": 0,
                "error": None,
            })

            if not USE_API_MODE:
                # Standalone: run ingestion synchronously with a progress indicator
                with st.spinner(f"Processing '{uf.name}'…"):
                    backend_ingest_sync(doc_id, uf.name, file_bytes, content_type)
                # Refresh status from job registry
                status_data = backend_status(doc_id)
                if status_data:
                    for doc in st.session_state.documents:
                        if doc["doc_id"] == doc_id:
                            doc["status"] = status_data["status"]
                            doc["chunk_count"] = status_data.get("chunk_count", 0)
                            doc["error"] = status_data.get("error")
                if status_data and status_data["status"] == "completed":
                    st.success(f"'{uf.name}' indexed — {status_data['chunk_count']} chunks.")
                elif status_data and status_data["status"] == "failed":
                    st.error(f"Failed: {status_data.get('error')}")
            else:
                st.info(f"'{uf.name}' accepted. Refresh to update status.")

        st.rerun()

    st.divider()

    # --- Document list ---
    st.subheader("Indexed Documents")

    if not st.session_state.documents:
        st.caption("No documents uploaded yet.")
    else:
        if USE_API_MODE:
            if st.button("🔄 Refresh statuses", use_container_width=True):
                for doc in st.session_state.documents:
                    if doc["status"] not in ("completed", "failed"):
                        data = backend_status(doc["doc_id"])
                        if data:
                            doc["status"] = data["status"]
                            doc["chunk_count"] = data.get("chunk_count", 0)
                            doc["error"] = data.get("error")
                st.rerun()

        for doc in st.session_state.documents:
            icon = STATUS_ICON.get(doc["status"], "?")
            st.markdown(f"{icon} **{doc['filename']}** — {doc['status'].capitalize()}")
            if doc["status"] == "completed":
                st.caption(f"  {doc['chunk_count']} chunks")
            elif doc["status"] == "failed":
                st.caption(f"  Error: {doc['error']}")

    st.divider()

    # --- Scope filter ---
    completed_docs = [d for d in st.session_state.documents if d["status"] == "completed"]
    if completed_docs:
        st.subheader("Search Scope")
        search_all = st.checkbox("Search all documents", value=True)
        if not search_all:
            selected = st.multiselect(
                "Select documents",
                options=[d["doc_id"] for d in completed_docs],
                format_func=lambda did: next(
                    (d["filename"] for d in completed_docs if d["doc_id"] == did), did
                ),
            )
            st.session_state.selected_doc_ids = selected
        else:
            st.session_state.selected_doc_ids = []

# ---------------------------------------------------------------------------
# Main panel — query
# ---------------------------------------------------------------------------

st.header("Ask a Question")

with st.expander("⚙️ Retrieval Settings", expanded=False):
    top_k = st.slider("Chunks to retrieve (top_k)", 1, 10, 5)

query_text = st.text_area(
    "Your question",
    placeholder="e.g. What are the main findings of the study?",
    height=80,
    label_visibility="collapsed",
)

no_docs = not completed_docs
ask_btn = st.button(
    "Ask",
    type="primary",
    disabled=no_docs,
    help="Upload and process at least one document first." if no_docs else "",
)

if no_docs and not st.session_state.query_history:
    st.info("Upload a PDF or TXT document in the sidebar to get started.")

# ---------------------------------------------------------------------------
# Query execution
# ---------------------------------------------------------------------------

if ask_btn:
    if not query_text.strip():
        st.warning("Please enter a question.")
    else:
        doc_ids = st.session_state.selected_doc_ids or None
        with st.spinner("Retrieving context and generating answer…"):
            result = backend_query(query_text.strip(), top_k=top_k, doc_ids=doc_ids)

        if result:
            st.session_state.query_history.insert(0, {
                "query": result["query"],
                "answer": result["answer"],
                "sources": result["sources"],
                "retrieval_ms": result["retrieval_latency_ms"],
                "generation_ms": result["generation_latency_ms"],
            })
            st.rerun()

# ---------------------------------------------------------------------------
# Query history
# ---------------------------------------------------------------------------

for entry in st.session_state.query_history:
    st.markdown(f"**Q: {entry['query']}**")
    st.markdown(entry["answer"])

    col1, col2, col3 = st.columns(3)
    col1.metric("Retrieval", f"{entry['retrieval_ms']:.0f} ms")
    col2.metric("Generation", f"{entry['generation_ms']:.0f} ms")
    col3.metric("Total", f"{entry['retrieval_ms'] + entry['generation_ms']:.0f} ms")

    if entry["sources"]:
        with st.expander(f"📄 Sources ({len(entry['sources'])} chunks retrieved)"):
            for j, src in enumerate(entry["sources"], 1):
                loc = f"page {src['page']}" if src.get("page") else "text file"
                st.markdown(
                    f"**Source {j}** — `{src['filename']}` | {loc} | "
                    f"similarity **{src['score']:.3f}**"
                )
                st.markdown(
                    f"<div style='background:#f0f2f6;padding:10px;border-radius:6px;"
                    f"font-size:0.85rem;color:#333'>{src['text']}</div>",
                    unsafe_allow_html=True,
                )
                if j < len(entry["sources"]):
                    st.divider()
    else:
        st.caption("No chunks met the similarity threshold.")

    st.divider()

if st.session_state.query_history:
    if st.button("🗑️ Clear history"):
        st.session_state.query_history = []
        st.rerun()
