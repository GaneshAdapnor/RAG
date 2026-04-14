"""
Ingestion service: orchestrates the full document processing pipeline.

PIPELINE STAGES (in order):
    1. Validate file type and size
    2. Extract text (page-by-page for PDFs, whole file for TXT)
    3. Chunk text with sliding window
    4. Generate embeddings in a single batched call
    5. Add vectors + metadata to FAISS
    6. Persist index to disk
    7. Update in-memory job status

WHY FastAPI BackgroundTasks (not Celery):
    - Celery requires a broker (Redis/RabbitMQ) — a significant operational
      dependency for a service that processes documents serially.
    - FastAPI BackgroundTasks run in the same process as the web server on a
      separate thread. The /upload endpoint returns immediately; processing
      happens asynchronously in the background.
    - Trade-off: if the server crashes mid-ingestion, the job is lost and must
      be re-uploaded. For a production system with SLAs, switch to Celery +
      Redis with result backend. For this system, the simplicity wins.
    - We store job state in-process (dict). For multi-replica deployments,
      this would need to move to Redis — a one-line change.

JOB STATE:
    A simple dict maps doc_id → JobStatus. Thread safety is ensured by
    Python's GIL for dict reads/writes plus explicit locking around FAISS writes
    (handled inside FAISSVectorStore).
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional

from app.core.config import settings
from app.models.schemas import DocumentStatus
from app.services.embedding_service import embed_texts
from app.services.vector_store import ChunkMetadata, get_vector_store
from app.utils.chunking import chunk_pages
from app.utils.text_extraction import extract_text

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Job tracking
# ---------------------------------------------------------------------------

@dataclass
class JobStatus:
    """In-memory state for a document processing job."""
    doc_id: str
    filename: str
    status: DocumentStatus = DocumentStatus.PENDING
    chunk_count: int = 0
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None


# Module-level job registry — keyed by doc_id
_jobs: Dict[str, JobStatus] = {}


def get_job(doc_id: str) -> Optional[JobStatus]:
    return _jobs.get(doc_id)


def list_jobs() -> Dict[str, JobStatus]:
    return dict(_jobs)


# ---------------------------------------------------------------------------
# Main ingestion function
# ---------------------------------------------------------------------------

def ingest_document(
    doc_id: str,
    filename: str,
    file_bytes: bytes,
    content_type: str,
) -> None:
    """
    Full ingestion pipeline: extract → chunk → embed → store → persist.

    This function is designed to run as a BackgroundTask. It updates the
    in-process job registry at each stage so callers can poll status.

    Args:
        doc_id: Pre-generated UUID assigned at upload time.
        filename: Original filename (for metadata and logging).
        file_bytes: Raw bytes of the uploaded file.
        content_type: MIME type from the HTTP Content-Type header.
    """
    # Register job as processing
    _jobs[doc_id] = JobStatus(
        doc_id=doc_id,
        filename=filename,
        status=DocumentStatus.PROCESSING,
    )

    logger.info("Ingestion started: doc_id=%s, filename='%s'.", doc_id, filename)
    start_time = time.perf_counter()

    try:
        # ------------------------------------------------------------------
        # Stage 1: Text extraction
        # ------------------------------------------------------------------
        pages = extract_text(file_bytes, filename, content_type)

        if not pages:
            raise ValueError(
                f"No text could be extracted from '{filename}'. "
                "The file may be empty, image-only, or corrupt."
            )

        logger.info(
            "doc_id=%s: Extracted %d page(s) of text.", doc_id, len(pages)
        )

        # ------------------------------------------------------------------
        # Stage 2: Chunking
        # ------------------------------------------------------------------
        chunks = chunk_pages(
            pages=pages,
            doc_id=doc_id,
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
        )

        if not chunks:
            raise ValueError(
                f"Chunking produced zero chunks for '{filename}'. "
                "The document may contain only whitespace or very short text."
            )

        logger.info("doc_id=%s: Created %d chunks.", doc_id, len(chunks))

        # ------------------------------------------------------------------
        # Stage 3: Embedding generation (single batched call)
        # ------------------------------------------------------------------
        texts = [c.text for c in chunks]
        embed_start = time.perf_counter()
        vectors = embed_texts(texts, batch_size=64)
        embed_elapsed = (time.perf_counter() - embed_start) * 1000

        logger.info(
            "doc_id=%s: Embedded %d chunks in %.1fms.",
            doc_id, len(texts), embed_elapsed,
        )

        # ------------------------------------------------------------------
        # Stage 4: Build metadata records
        # ------------------------------------------------------------------
        metadata_records = [
            ChunkMetadata(
                doc_id=doc_id,
                chunk_id=chunk.chunk_id,
                filename=filename,
                page=chunk.page,
                text=chunk.text,
            )
            for chunk in chunks
        ]

        # ------------------------------------------------------------------
        # Stage 5: Add to FAISS + persist
        # ------------------------------------------------------------------
        store = get_vector_store()
        store.add_vectors(vectors, metadata_records)
        store.save()

        # ------------------------------------------------------------------
        # Stage 6: Update job status
        # ------------------------------------------------------------------
        elapsed = (time.perf_counter() - start_time) * 1000
        _jobs[doc_id].status = DocumentStatus.COMPLETED
        _jobs[doc_id].chunk_count = len(chunks)
        _jobs[doc_id].completed_at = time.time()

        logger.info(
            "Ingestion completed: doc_id=%s, chunks=%d, total_time=%.1fms.",
            doc_id, len(chunks), elapsed,
        )

    except Exception as exc:
        elapsed = (time.perf_counter() - start_time) * 1000
        _jobs[doc_id].status = DocumentStatus.FAILED
        _jobs[doc_id].error = str(exc)
        _jobs[doc_id].completed_at = time.time()

        logger.error(
            "Ingestion failed: doc_id=%s, filename='%s', error='%s', elapsed=%.1fms.",
            doc_id, filename, exc, elapsed,
        )
        # Do NOT re-raise — BackgroundTask exceptions are silently swallowed by
        # FastAPI. We've already logged and recorded the failure state.


def create_doc_id() -> str:
    """Generate a new document ID (UUID4)."""
    return str(uuid.uuid4())


def validate_upload(filename: str, file_size: int, content_type: str) -> None:
    """
    Pre-validate the uploaded file before accepting it.

    Args:
        filename: Original filename.
        file_size: Size in bytes.
        content_type: MIME type.

    Raises:
        ValueError: With a user-readable message if validation fails.
    """
    max_size_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    if file_size == 0:
        raise ValueError("Uploaded file is empty.")

    if file_size > max_size_bytes:
        raise ValueError(
            f"File size ({file_size / 1_048_576:.1f} MB) exceeds the "
            f"{settings.MAX_UPLOAD_SIZE_MB} MB limit."
        )

    allowed_types = {"application/pdf", "text/plain"}
    allowed_extensions = {".pdf", ".txt"}
    from pathlib import Path
    ext = Path(filename).suffix.lower()

    if content_type not in allowed_types and ext not in allowed_extensions:
        raise ValueError(
            f"Unsupported file type '{content_type}' (extension '{ext}'). "
            "Only PDF and TXT files are accepted."
        )
