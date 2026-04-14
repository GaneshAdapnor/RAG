# Production RAG QA API

A production-quality **Retrieval-Augmented Generation (RAG) Question Answering API** built from first principles with FastAPI, FAISS, and OpenAI. Upload PDF or TXT documents, then ask questions and receive accurate, grounded answers with full source attribution.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Project Structure](#project-structure)
3. [Setup & Installation](#setup--installation)
4. [Configuration](#configuration)
5. [API Reference](#api-reference)
6. [Streamlit UI](#streamlit-ui)
7. [Streamlit Cloud Deployment](#streamlit-cloud-deployment)
8. [Design Decisions](#design-decisions)
9. [Troubleshooting](#troubleshooting)

---

## Architecture

> **diagram:** open `architecture.drawio` at [app.diagrams.net](https://app.diagrams.net) (File → Import From → Device)

### Component map

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      FastAPI Backend  (port 8000)                           │
│                                                                             │
│  POST /upload          POST /query            GET /health                   │
│       │                     │                      │                        │
│  ─────────────────────────────────────────────────────────────────────────  │
│  INGESTION PIPELINE    QUERY PIPELINE                                       │
│                                                                             │
│  Save file to disk     TokenBucketRateLimiter                               │
│  DocumentRegistry      QueryService.answer()                                │
│    .create()             embed_query()                                      │
│  BackgroundTask ──►      FaissVectorStore.search()                         │
│    DocumentParser        LLMService.generate_answer() ──► OpenAI API       │
│    TextChunker           MetricsService (latency)                           │
│    EmbeddingService      QueryResponse                                      │
│    FaissVectorStore                                                         │
│      .add_embeddings()                                                      │
│                                                                             │
│  ─────────────────────────────────────────────────────────────────────────  │
│  STORAGE LAYER  ./data/                                                     │
│  uploads/        index/faiss.index    vector_metadata.json  documents.json  │
└─────────────────────────────────────────────────────────────────────────────┘
         ▲                                              ▲
         │ HTTP                                         │ HTTP
┌────────┴───────┐                          ┌──────────┴───────┐
│  Streamlit UI  │                          │   curl / client  │
│  (port 8501)   │                          └──────────────────┘
└────────────────┘
```

### End-to-end data flow

#### Upload flow
```
POST /upload (multipart/form-data)
  → validate extension (.pdf / .txt)
  → stream file to  data/uploads/{uuid}_{filename}
  → DocumentRegistry.create()  →  data/index/documents.json  (status: pending)
  → return 202 Accepted  {document_id, filename, status, bytes_written}
       ↓ (BackgroundTask — non-blocking)
  DocumentIngestionService.ingest_document(document_id)
    → DocumentParser.parse()       PDF: PyPDF2 page-by-page | TXT: UTF-8 / latin-1
    → TextChunker.chunk_pages()    sliding window 500 tok / 100 tok overlap
    → EmbeddingService.embed_texts()  all-MiniLM-L6-v2 · 384-dim · L2-normalized
    → FaissVectorStore.add_embeddings()  IndexFlatIP + atomic JSON persist
    → DocumentRegistry.mark_ready()  →  documents.json  (status: ready)
```

#### Query flow
```
POST /query  {question, top_k?, document_ids?}
  → enforce_rate_limit (TokenBucket per-IP)
  → QueryService.answer()
      → EmbeddingService.embed_query()   → float32[384]
      → FaissVectorStore.search()        → top-k  (cosine sim ≥ 0.22)
      → LLMService.generate_answer()     → grounded answer + citations
      → MetricsService.record_latency()
  → return 200  {question, answer, answer_model, latency_ms, sources[]}
```

### Why these choices?

| Component | Choice | Reason |
|---|---|---|
| **Web framework** | FastAPI | Native async, Pydantic v2, BackgroundTasks, auto-OpenAPI docs |
| **Vector index** | `FAISS IndexFlatIP` | Exact cosine search (via L2-norm); no training; < 5 ms at < 500 K vectors |
| **Embedding model** | `all-MiniLM-L6-v2` | Free, local, 384-dim, runs on CPU, competitive semantic retrieval |
| **LLM** | `gpt-4o-mini` | Strong instruction following, $0.15/1 M tokens, 128 K context |
| **Background jobs** | `FastAPI BackgroundTasks` | Zero-dependency; sufficient for sequential ingestion; Celery-ready boundary |
| **Rate limiting** | Custom `TokenBucketRateLimiter` | Sliding window; no external broker; per-IP isolation |
| **Chunking** | Sliding window 500 tok / 100 tok overlap | Balances context preservation vs embedding quality; stays inside MiniLM's 256-wordpiece limit |

---

## Project Structure

```
RAG/
├── app/
│   ├── main.py                          # App factory, startup hooks, route registration
│   ├── api/
│   │   └── routes/
│   │       ├── health.py                # GET /health
│   │       ├── upload.py                # POST /upload
│   │       └── query.py                 # POST /query
│   ├── core/
│   │   ├── config.py                    # Pydantic Settings (env-var driven, lru_cache)
│   │   ├── dependencies.py              # FastAPI dependency factory functions
│   │   └── logging_config.py            # Structured logging setup
│   ├── models/
│   │   ├── domain.py                    # Internal dataclasses (DocumentRecord, ChunkRecord …)
│   │   └── api.py                       # Request/response Pydantic schemas
│   ├── services/
│   │   ├── document_ingestion.py        # Orchestrates the full ingestion pipeline
│   │   ├── document_parser.py           # PDF (PyPDF2) and TXT extraction
│   │   ├── document_registry.py         # Thread-safe document state, persisted to JSON
│   │   ├── chunker.py                   # Token-based sliding-window chunker
│   │   ├── embedding_service.py         # SentenceTransformer singleton, embed_texts / embed_query
│   │   ├── vector_store.py              # FaissVectorStore — add / search / stats
│   │   ├── llm_service.py               # OpenAI GPT-4o-mini + extractive fallback
│   │   ├── query_service.py             # Query orchestration (embed → retrieve → generate)
│   │   └── metrics_service.py           # In-process latency rolling window
│   └── utils/
│       ├── files.py                     # File I/O helpers, atomic JSON write
│       ├── text.py                      # Text normalization, tokenization
│       ├── rate_limiter.py              # TokenBucketRateLimiter (per-IP)
│       └── logging.py                   # configure_logging()
├── data/
│   ├── uploads/                         # Raw uploaded files (gitignored)
│   └── index/                           # FAISS index + JSON metadata (gitignored)
├── examples/
│   ├── sample_handbook.txt              # Sample document for testing
│   └── sample_query.py                  # End-to-end HTTP demo script
├── streamlit_app.py                     # Browser UI (standalone or API-mode)
├── architecture.drawio                  # Architecture diagram (draw.io)
├── requirements.txt
├── .env.example
└── .streamlit/
    └── secrets.toml.example            # Streamlit secrets template
```

---

## Setup & Installation

### Prerequisites

- Python 3.10+
- An OpenAI API key — [get one here](https://platform.openai.com/api-keys)

### 1. Clone and create a virtual environment

```bash
git clone https://github.com/GaneshAdapnor/RAG.git
cd RAG
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **First run note:** `sentence-transformers` downloads `all-MiniLM-L6-v2` (~90 MB) from
> HuggingFace Hub on first startup. It is cached in `~/.cache/huggingface/` for subsequent runs.

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and set your OpenAI key:

```ini
OPENAI_API_KEY=sk-your-key-here
```

All other defaults are production-ready out of the box.

### 4. Run

**API only (REST interface):**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**API + Streamlit browser UI (two terminals):**
```bash
# Terminal 1 — backend
uvicorn app.main:app --reload --port 8000

# Terminal 2 — frontend
streamlit run streamlit_app.py
```

Open:
- API docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- Streamlit UI: [http://localhost:8501](http://localhost:8501)

---

## Configuration

All settings live in `.env` (or environment variables). Managed by Pydantic `BaseSettings`.

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | *(required)* | OpenAI API key |
| `LLM_MODEL_NAME` | `gpt-4o-mini` | OpenAI model for answer generation |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | Sentence-transformers model name |
| `EMBEDDING_DIM` | `384` | Must match model output dimension |
| `CHUNK_SIZE_TOKENS` | `500` | Tokens per chunk (whitespace-split) |
| `CHUNK_OVERLAP_TOKENS` | `100` | Token overlap between adjacent chunks |
| `TOP_K_RETRIEVAL` | `4` | Default chunks retrieved per query |
| `SEARCH_MIN_SCORE` | `0.22` | Minimum cosine similarity to include a chunk |
| `LLM_TEMPERATURE` | `0.1` | LLM sampling temperature (0 = deterministic) |
| `LLM_MAX_TOKENS` | `500` | Maximum tokens in generated answer |
| `ENABLE_EXTRACTIVE_FALLBACK` | `true` | Return extractive answer when no API key is set |
| `RATE_LIMIT_CALLS` | `10` | Max requests per client IP per period |
| `RATE_LIMIT_PERIOD_SECONDS` | `60` | Rate limit window in seconds |
| `MAX_UPLOAD_BYTES` | `20971520` | Max upload size (default 20 MB) |
| `LOG_LEVEL` | `INFO` | Python logging level |

---

## API Reference

### `GET /health`

Returns service status and index statistics.

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "version": "1.0.0",
  "indexed_documents": 2,
  "indexed_chunks": 47,
  "documents_by_status": {
    "pending": 0,
    "processing": 0,
    "ready": 2,
    "failed": 0
  },
  "average_query_latency_ms": 312.4
}
```

---

### `POST /upload`

Upload a PDF or TXT file. Returns immediately with `202 Accepted`; processing happens in the background.

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@./examples/sample_handbook.txt"
```

```json
{
  "document_id": "a4c85b5bcfb44f54bf8130a4baef0d8d",
  "filename": "sample_handbook.txt",
  "status": "pending",
  "bytes_written": 4821,
  "message": "Document accepted and queued for background ingestion."
}
```

**Statuses:** `pending` → `processing` → `ready` | `failed`

Poll `GET /health` or `GET /upload/status/{document_id}` *(if implemented)* to check progress. A document is queryable once `ready`.

**Constraints:**
- Accepted types: `.pdf`, `.txt`
- Max size: 20 MB (configurable via `MAX_UPLOAD_BYTES`)
- Rate limited: configurable per-IP token bucket

---

### `POST /query`

Ask a question. The system retrieves the most relevant chunks and generates a grounded answer.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the main findings of the battery research?",
    "top_k": 4
  }'
```

```json
{
  "question": "What are the main findings of the battery research?",
  "answer": "According to sample_handbook.txt (p.1–2), the main findings include...",
  "answer_model": "gpt-4o-mini",
  "latency_ms": 287.5,
  "retrieved_chunks": 4,
  "documents_considered": ["a4c85b5bcfb44f54bf8130a4baef0d8d"],
  "sources": [
    {
      "document_id": "a4c85b5bcfb44f54bf8130a4baef0d8d",
      "filename": "sample_handbook.txt",
      "chunk_id": "chunk-00002",
      "page_start": 1,
      "page_end": 2,
      "score": 0.8734,
      "excerpt": "Key Findings: The new composite achieved 450 Wh/kg..."
    }
  ]
}
```

**Request fields:**

| Field | Type | Required | Description |
|---|---|---|---|
| `question` | string | Yes | Natural language question (3–2000 chars) |
| `top_k` | integer | No | Chunks to retrieve (1–10, default from config) |
| `document_ids` | string[] | No | Restrict search to these document IDs; omit to search all ready documents |

**Error responses:**

| Status | Condition |
|---|---|
| `400` | Unsupported file type on upload |
| `409` | No ready documents available to query |
| `413` | File exceeds `MAX_UPLOAD_BYTES` |
| `429` | Rate limit exceeded |
| `404` | Unknown `document_id` in query scope |

---

### Run the demo script

```bash
# Server must be running first
python examples/sample_query.py
```

This uploads `examples/sample_handbook.txt`, waits for processing, then runs four example queries and prints answers with source attribution and latency.

---

## Streamlit UI

A browser interface is included — no backend changes required.

```bash
streamlit run streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501).

**Features:**
- Drag-and-drop PDF/TXT upload with live status badges
- Document scope selector (search all or specific docs)
- Question input with configurable `top_k`
- Answer display with per-chunk source cards (filename, page, similarity score, text excerpt)
- Latency breakdown: retrieval vs generation vs total

**Modes:**
- **Standalone** (default, works on Streamlit Cloud): calls service functions directly — no separate API server needed
- **API mode**: set `STREAMLIT_API_BASE=http://localhost:8000` to route calls through the FastAPI backend

---

## Streamlit Cloud Deployment

1. Push this repo to GitHub (already done)
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Fill in:
   - Repository: `GaneshAdapnor/RAG`
   - Branch: `main`
   - Main file: `streamlit_app.py`
4. Click **Advanced settings → Secrets** and paste:
   ```toml
   OPENAI_API_KEY = "sk-your-key-here"
   ```
5. Click **Deploy**

> **Note on persistence:** Streamlit Community Cloud has ephemeral storage. The FAISS index resets on each reboot — users re-upload documents per session. For persistent cross-session storage, mount a cloud volume or store the index in S3/GCS.

---

## Design Decisions

### Chunk size: 500 tokens, 100 token overlap

Token-based chunking (whitespace-split) rather than character-based is more consistent across languages. 500 tokens ≈ 350–400 words ≈ 3–6 sentences.

| | Small (< 150 tok) | **Our choice (500 tok)** | Large (> 1 000 tok) |
|---|---|---|---|
| Embedding quality | Poor — too little context | Good | Degrades (MiniLM 256-piece limit) |
| Retrieval precision | High | Balanced | Low |
| LLM context cost | Very low | Low (~5 × 500 = 2 500 tok/query) | High |
| Boundary information loss | High | Low (100 tok overlap) | Minimal |

The 100-token overlap (20%) ensures that a fact spanning a chunk boundary appears complete in at least one chunk.

### Similarity threshold: 0.22

Chunks with cosine similarity < 0.22 are dropped before the LLM call. Below this score, the retrieved passage is semantically unrelated and adds noise. Tune this value based on your corpus — diverse document sets benefit from a lower threshold.

### Realistic retrieval failure example

**Query:** *"What was the net profit margin in Q3?"*

**Why it fails:** The document contains *"net income as a percentage of revenue was 18.3% in the third quarter"* — no phrase match for "net profit margin" or "Q3". Additionally, the chunk boundary splits the sentence:
- Chunk N ends: `"...revenue was 18.3% in the"`
- Chunk N+1 starts: `"third quarter, compared to 21.1%..."`

Retrieved score ≈ 0.61 (above threshold), but neither chunk alone contains the full fact.

**Fixes (in order of effort):**
1. Increase `CHUNK_OVERLAP_TOKENS` to 150+ so the boundary sentence is complete in at least one chunk.
2. **Hybrid search:** combine FAISS dense score with BM25 sparse score. BM25 would rank "Q3" and "net profit" higher via exact keyword match.
3. **Cross-encoder re-ranking:** run `cross-encoder/ms-marco-MiniLM-L-6-v2` on the top-20 FAISS results as a second stage. Cross-encoders attend jointly to query + passage, catching semantic alignments bi-encoders miss.
4. **Query expansion:** normalize *"Q3" → "third quarter"* before embedding.

### Metric tracked: query latency

`latency_ms` is measured end-to-end per query (embed + retrieve + generate) and stored in a rolling 500-sample window in `MetricsService`. Returned in every `QueryResponse` and surfaced in `GET /health` as `average_query_latency_ms`.

**Why latency over similarity score or accuracy:**
- Observable without a labeled test set.
- Directly maps to user experience.
- A spike in latency is immediately actionable: profile whether the bottleneck is in embedding (< 50 ms expected), FAISS search (< 5 ms), or OpenAI (500–3000 ms).

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `No ready documents available` on query | Document still processing or failed | Check `GET /health` → `documents_by_status`; re-upload if `failed` |
| Empty answer / "I don't have enough information" | `SEARCH_MIN_SCORE` too high, or query semantically distant from corpus | Lower `SEARCH_MIN_SCORE` in `.env`; rephrase query |
| `RuntimeError: No OpenAI API key` | `OPENAI_API_KEY` not set | Add key to `.env`; system uses extractive fallback if `ENABLE_EXTRACTIVE_FALLBACK=true` |
| Slow first query | Embedding model loading | Expected on cold start; subsequent queries are fast (model cached in memory) |
| `413 Request Entity Too Large` | File exceeds `MAX_UPLOAD_BYTES` | Increase limit in `.env` or compress the file |
| `429 Too Many Requests` | Rate limit hit | Wait for the token bucket to refill; increase `RATE_LIMIT_CALLS` in `.env` |
| PDF produces 0 chunks | Image-only / scanned PDF | Add OCR pre-processing with `pytesseract` before `DocumentParser` |
| Index lost on restart (Streamlit Cloud) | Ephemeral storage | Add cloud volume mount or persist index to S3/GCS between sessions |

---

## Extending the System

| Goal | Change |
|---|---|
| **Scale to 1 M+ vectors** | Swap `IndexFlatIP` → `IndexHNSWFlat` in `vector_store.py` (one line) |
| **Multi-replica deployment** | Replace `lru_cache` singletons with Redis-backed stores |
| **Persistent background jobs** | Replace `BackgroundTasks` with Celery + Redis; service boundaries already isolated |
| **Scanned PDFs** | Add `pytesseract` OCR step in `document_parser.py` before PyPDF2 |
| **Hybrid BM25 + FAISS** | Add `rank_bm25` index; merge scores in `query_service.py` |
| **Re-ranking** | Add `cross-encoder/ms-marco-MiniLM-L-6-v2` pass after `vector_store.search()` |
| **Local LLM** | Replace OpenAI client in `llm_service.py` with Ollama or vLLM |
