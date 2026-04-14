# Production RAG Question Answering API

A production-quality Retrieval-Augmented Generation (RAG) system built from first principles. Upload PDF or TXT documents, then ask questions — get accurate, grounded answers backed by source attribution.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           FastAPI Application                             │
│                                                                          │
│  ┌─────────────┐    ┌──────────────────┐    ┌─────────────────────────┐ │
│  │ POST /upload│    │  POST /query     │    │     GET /health         │ │
│  └──────┬──────┘    └────────┬─────────┘    └─────────────────────────┘ │
│         │ 202              │                                             │
│  ┌──────▼──────┐    ┌──────▼─────────────────────────────────────────┐  │
│  │  Background │    │               Query Pipeline                    │  │
│  │    Task     │    │                                                 │  │
│  │             │    │  1. embed_query()  ──► EmbeddingService         │  │
│  │  Ingestion  │    │  2. faiss.search() ──► FAISSVectorStore         │  │
│  │  Pipeline:  │    │  3. build_context()                             │  │
│  │             │    │  4. generate_answer() ──► OpenAI GPT-4o-mini    │  │
│  │ ┌─────────┐ │    └─────────────────────────────────────────────────┘  │
│  │ │Extract  │ │                                                         │
│  │ │(PDF/TXT)│ │                                                         │
│  │ └────┬────┘ │                                                         │
│  │      │      │                                                         │
│  │ ┌────▼────┐ │                                                         │
│  │ │  Chunk  │ │                                                         │
│  │ │(500c/50)│ │                                                         │
│  │ └────┬────┘ │                                                         │
│  │      │      │                                                         │
│  │ ┌────▼────┐ │                                                         │
│  │ │  Embed  │ │                                                         │
│  │ │MiniLM   │ │                                                         │
│  │ └────┬────┘ │                                                         │
│  │      │      │                                                         │
│  │ ┌────▼────┐ │                                                         │
│  │ │  FAISS  │ │                                                         │
│  │ │  Store  │ │                                                         │
│  │ └─────────┘ │                                                         │
│  └─────────────┘                                                         │
└──────────────────────────────────────────────────────────────────────────┘

Persistence Layer:
  ┌──────────────────────────────┐
  │  ./data/faiss_index.bin      │  ← FAISS binary index (float32 vectors)
  │  ./data/metadata.json        │  ← Parallel chunk metadata (text, doc_id, page)
  └──────────────────────────────┘
```

### Component Interactions

| Component | Role | Why chosen |
|-----------|------|-----------|
| **FastAPI** | HTTP server, request routing, background tasks | Native async, auto-docs, Pydantic integration |
| **BackgroundTasks** | Non-blocking document ingestion | Zero-dependency async; sufficient for serial ingestion |
| **sentence-transformers** | Text → dense vector encoding | Free, local, 384-dim, competitive retrieval quality |
| **FAISS IndexFlatIP** | Exact cosine similarity search | No approximation needed at < 500K vectors; simple; fast |
| **OpenAI GPT-4o-mini** | Grounded answer generation | Strong instruction following; cost-effective; 128K context |
| **SlowAPI** | Per-IP rate limiting | Flask-Limiter port for FastAPI; Redis-compatible |
| **PyPDF2** | PDF text extraction | Pure-Python; no system dependencies |

### End-to-End Data Flow

```
Upload:
  HTTP multipart/form-data
    → validate (type, size)
    → assign doc_id (UUID4)
    → schedule BackgroundTask
    → return 202 with doc_id
         ↓ (async)
    → extract_text() per page
    → chunk_pages() sliding window (500c, 50c overlap)
    → embed_texts() → L2-normalized float32[384]
    → faiss.add() + metadata.append()
    → faiss.write_index() + json.dump() (atomic)
    → job status → COMPLETED

Query:
  HTTP POST {query, top_k?, doc_ids?}
    → embed_query() → float32[384]
    → faiss.search(k=5) → [(index, score)]
    → filter by doc_ids + similarity threshold (0.30)
    → build_context() → formatted string
    → openai.chat.completions.create()
    → return {answer, sources, latency_metrics}
```

---

## Project Structure

```
RAG/
├── app/
│   ├── main.py                   # FastAPI app factory, lifespan, middleware
│   ├── core/
│   │   ├── config.py             # Pydantic Settings (env-var driven)
│   │   └── logging_config.py     # Structured logging setup
│   ├── models/
│   │   └── schemas.py            # All Pydantic v2 request/response models
│   ├── routes/
│   │   ├── upload.py             # POST /upload, GET /upload/status/{id}
│   │   ├── query.py              # POST /query
│   │   └── health.py             # GET /health
│   ├── services/
│   │   ├── embedding_service.py  # SentenceTransformer singleton, embed_texts()
│   │   ├── vector_store.py       # FAISSVectorStore (add, search, persist)
│   │   ├── ingestion_service.py  # Full ingestion pipeline + job tracking
│   │   ├── retrieval_service.py  # Query embed → FAISS → context builder
│   │   └── llm_service.py        # OpenAI client, grounded prompt, generate_answer()
│   └── utils/
│       ├── text_extraction.py    # PDF (PyPDF2) + TXT extraction
│       ├── chunking.py           # Sliding window chunker with overlap
│       └── rate_limiter.py       # SlowAPI limiter, client IP detection
├── data/                         # FAISS index + metadata (auto-created)
├── examples/
│   └── sample_query.py           # End-to-end demo script (curl-based)
├── streamlit_app.py              # Browser UI (talks to FastAPI over HTTP)
├── requirements.txt
├── .env.example
└── README.md
```

---

## Streamlit UI

A browser-based frontend is included. It talks to the FastAPI backend over HTTP — no backend changes required.

```
┌─────────────────────────────┐       HTTP        ┌──────────────────────┐
│   Streamlit  (port 8501)    │ ────────────────► │  FastAPI  (port 8000)│
│                             │                   │                      │
│  Sidebar:                   │  POST /upload     │  Embedding           │
│   • File uploader           │  GET  /status     │  FAISS search        │
│   • Doc status badges       │  POST /query      │  OpenAI generation   │
│   • Search scope filter     │ ◄──────────────── │                      │
│  Main:                      │                   └──────────────────────┘
│   • Question input          │
│   • Answer + latency pills  │
│   • Collapsible sources     │
└─────────────────────────────┘
```

**Run both together:**

```bash
# Terminal 1 — API backend
uvicorn app.main:app --reload --port 8000

# Terminal 2 — Streamlit UI
streamlit run streamlit_app.py
```

Then open [http://localhost:8501](http://localhost:8501).

---

## Setup

### 1. Prerequisites

- Python 3.10+
- OpenAI API key ([get one here](https://platform.openai.com/api-keys))

### 2. Install dependencies

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install packages
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-your-key-here
```

### 4. Run the server

**API only:**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**API + Streamlit UI (two terminals):**
```bash
# Terminal 1
uvicorn app.main:app --reload --port 8000

# Terminal 2
streamlit run streamlit_app.py
```

On first start, the embedding model (`all-MiniLM-L6-v2`, ~90MB) is downloaded
from HuggingFace Hub automatically. Subsequent starts use the cached model.

---

## API Usage

### Interactive docs

Open [http://localhost:8000/docs](http://localhost:8000/docs) for the Swagger UI.

---

### GET /health

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "version": "1.0.0",
  "embedding_model": "all-MiniLM-L6-v2",
  "indexed_chunks": 0,
  "indexed_documents": 0
}
```

---

### POST /upload

Upload a PDF or TXT file for indexing.

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@/path/to/your/document.pdf"
```

```json
{
  "doc_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "filename": "document.pdf",
  "status": "pending",
  "message": "Document 'document.pdf' accepted for processing. Poll GET /upload/status/3fa85f64... to track progress."
}
```

#### Poll processing status

```bash
curl http://localhost:8000/upload/status/3fa85f64-5717-4562-b3fc-2c963f66afa6
```

```json
{
  "doc_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "filename": "document.pdf",
  "status": "completed",
  "chunk_count": 42,
  "error": null
}
```

Statuses: `pending` → `processing` → `completed` | `failed`

---

### POST /query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main findings of the battery research?",
    "top_k": 5
  }'
```

```json
{
  "query": "What are the main findings of the battery research?",
  "answer": "According to quantum_battery_report.txt, the main findings are:\n1. The new lithium-ceramic composite achieved an energy density of 450 Wh/kg — 2.5× greater than conventional lithium-ion batteries.\n2. Cycle life of 92% capacity after 1,200 cycles.\n3. Elimination of thermal runaway risk.\n4. Production cost of $185/kWh, projected to drop to $72/kWh at scale.",
  "sources": [
    {
      "doc_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
      "filename": "quantum_battery_report.txt",
      "chunk_id": 2,
      "page": null,
      "text": "Key Findings\n============\n1. Energy Density: The new composite achieved 450 Wh/kg...",
      "score": 0.8921
    }
  ],
  "retrieval_latency_ms": 18.4,
  "generation_latency_ms": 1243.7
}
```

#### Filter by document

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What was the R&D investment?",
    "top_k": 3,
    "doc_ids": ["3fa85f64-5717-4562-b3fc-2c963f66afa6"]
  }'
```

---

## Run the sample demo

```bash
# Make sure the server is running first
python examples/sample_query.py
```

Expected output (with valid OpenAI key):

```
STEP 1: Uploading 'quantum_battery_report.txt'...
Accepted: doc_id=abc123...

STEP 2: Waiting for background processing...
  [  2.0s] status=processing | chunks=18

STEP 3: Querying the RAG system...
Query: What energy density did the new battery achieve?

[ANSWER]
According to quantum_battery_report.txt, the new lithium-ceramic
composite electrolyte achieved an energy density of 450 Wh/kg —
approximately 2.5 times greater than conventional lithium-ion batteries
(180 Wh/kg).

[LATENCY]
  Retrieval (embed + FAISS):  21.3 ms
  Generation (OpenAI API):    987.4 ms
  Total:                      1008.7 ms

[SOURCES] (3 chunks retrieved)
  Source 1: quantum_battery_report.txt | similarity=0.8934
  'Key Findings === 1. Energy Density: The new composite achieved 450 Wh/kg...'
```

---

## Design Decisions & Justifications

### Chunk Size: 500 characters, 50 overlap

| Factor | Small chunks (< 200c) | Our choice (500c) | Large chunks (> 1000c) |
|--------|----------------------|-------------------|----------------------|
| Precision | High — tight focus | Balanced | Low — noisy context |
| Recall | Low — boundary cuts | Good | High |
| Embedding quality | Poor — too little context | Good | Degrades (model limit) |
| LLM cost | Low | Low (~600 tok/query) | High |
| all-MiniLM limit | Safe | Safe (< 256 wordpieces) | Risk of truncation |

The 50-char overlap (10% of chunk size) ensures sentence-spanning facts appear
complete in at least one chunk.

### Similarity Threshold: 0.30

Chunks with cosine similarity < 0.30 are dropped before LLM prompting. Below
this threshold, chunks are likely from a different topic entirely and would
add noise. If your use case has very diverse documents, lower to 0.20.

### top_k = 5 (default)

Retrieves ~2,500 characters of context. Provides enough evidence for multi-fact
questions without triggering "lost in the middle" degradation (LLMs attend
poorly to context buried in long prompts). Configurable per-query via `top_k`.

### Retrieval Failure Example

**Query:** "What was the net profit margin in Q3?"

**Why it fails:** The document contains _"net income as a percentage of revenue
was 18.3% in the third quarter"_ — no phrase match for "net profit margin" or
"Q3". Additionally, the chunk boundary splits:
- Chunk N: `"...revenue was 18.3% in the"`
- Chunk N+1: `"third quarter, compared to..."`

Retrieved score: ~0.62 (above threshold), but neither chunk alone contains
the complete fact.

**Fixes:**
1. Increase `CHUNK_OVERLAP` to 100+ chars so the boundary sentence appears whole.
2. Hybrid search: combine FAISS (dense) + BM25 (sparse) scores. BM25 would
   boost exact matches for "Q3" and "net profit".
3. Cross-encoder re-ranking: use `cross-encoder/ms-marco-MiniLM-L-6-v2` as a
   second-stage ranker after FAISS retrieval.
4. Query expansion: rephrase "Q3" → "third quarter" before embedding.

### Metric Tracked: Latency (retrieval vs. generation)

We split latency into two components:

| Metric | Healthy range | Alert threshold |
|--------|--------------|----------------|
| `retrieval_latency_ms` | 5–50 ms | > 200 ms |
| `generation_latency_ms` | 500–3000 ms | > 8000 ms |

**Why latency over similarity scores:**
- Immediately observable without a labeled test set.
- Directly maps to user experience.
- Separating retrieval vs. generation latency makes root-cause analysis instant:
  spike in retrieval → model reload or FAISS issue; spike in generation → OpenAI
  rate limit or network issue.

In production, emit these as Prometheus metrics and set p95 alerts.

---

## Rate Limits

| Endpoint | Limit |
|----------|-------|
| POST /query | 10 requests/minute/IP |
| POST /upload | 5 requests/minute/IP |
| GET /health | Unlimited |

Exceeded requests receive `HTTP 429` with a `Retry-After: 60` header.

To increase limits, edit `RATE_LIMIT_CALLS` in `.env`.

---

## Configuration Reference

All settings are in `.env` and documented in `.env.example`.

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | required | OpenAI API key |
| `LLM_MODEL_NAME` | `gpt-4o-mini` | OpenAI model for generation |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `EMBEDDING_DIM` | `384` | Must match model output dimension |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between adjacent chunks |
| `TOP_K_RETRIEVAL` | `5` | Default chunks retrieved per query |
| `MAX_UPLOAD_SIZE_MB` | `50` | Max file upload size |
| `RATE_LIMIT_CALLS` | `10` | Max query requests/minute/IP |

---

## Extending the System

| Goal | Change |
|------|--------|
| Scale to multiple replicas | Replace in-memory job store with Redis; use `RedisStorage` for SlowAPI |
| 1M+ vectors | Swap `IndexFlatIP` → `IndexHNSWFlat` (one line in `vector_store.py`) |
| Scanned PDFs | Add `pytesseract` OCR step before `extract_text_from_pdf()` |
| Hybrid search | Add BM25 index (rank_bm25); merge scores with FAISS in `retrieval_service.py` |
| Re-ranking | Add `cross-encoder/ms-marco-MiniLM-L-6-v2` step after FAISS retrieval |
| Local LLM | Replace `llm_service.py` with Ollama or vLLM client |
| Persistent job state | Replace `_jobs` dict with SQLite or Redis |
