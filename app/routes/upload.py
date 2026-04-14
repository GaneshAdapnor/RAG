"""
Document upload routes.

POST /upload:
    Accepts a multipart/form-data file upload. Validates the file, registers
    the document, and schedules background ingestion. Returns immediately with
    a doc_id so the client doesn't have to wait for processing.

GET /upload/status/{doc_id}:
    Poll-based status check. Clients should poll this after upload to know
    when the document is ready to query.

WHY polling instead of webhooks/WebSockets:
    - Simple clients (curl, basic scripts) can poll without extra dependencies.
    - Webhooks require the client to expose an endpoint — not feasible for all callers.
    - WebSockets add bidirectional state management overhead.
    - For a document QA use case, the latency difference doesn't matter:
      ingestion takes 5–30 seconds and the user is preparing their first query.
    - If push notifications are needed in future, this is a one-day change.
"""

import logging
import uuid

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Request, UploadFile

from app.models.schemas import DocumentStatus, DocumentStatusResponse, UploadResponse
from app.services.ingestion_service import (
    create_doc_id,
    get_job,
    ingest_document,
    validate_upload,
)
from app.utils.rate_limiter import UPLOAD_RATE_LIMIT, limiter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/upload", tags=["Documents"])


@router.post(
    "",
    response_model=UploadResponse,
    status_code=202,   # 202 Accepted — processing happens asynchronously
    summary="Upload a document for indexing",
    description=(
        "Upload a PDF or TXT file. The file is accepted immediately and processed "
        "in the background. Use GET /upload/status/{doc_id} to track progress."
    ),
)
@limiter.limit(UPLOAD_RATE_LIMIT)
async def upload_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="PDF or TXT file to index."),
) -> UploadResponse:
    """
    Accept a file upload and schedule background ingestion.

    Steps:
        1. Read file bytes
        2. Validate type and size
        3. Generate a doc_id
        4. Schedule ingest_document() as a background task
        5. Return 202 with doc_id immediately

    The client receives a response in < 50ms regardless of file size.
    """
    # Read file into memory. For very large files (>100MB), consider streaming
    # directly to disk. Our 50MB limit keeps this safe.
    file_bytes = await file.read()

    # Validate before scheduling — fail fast on bad input
    try:
        validate_upload(
            filename=file.filename or "unknown",
            file_size=len(file_bytes),
            content_type=file.content_type or "application/octet-stream",
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    doc_id = create_doc_id()
    filename = file.filename or f"document_{doc_id}"

    logger.info(
        "Upload accepted: doc_id=%s, filename='%s', size=%d bytes, content_type='%s'.",
        doc_id, filename, len(file_bytes), file.content_type,
    )

    # Schedule the ingestion pipeline as a background task.
    # FastAPI runs this on a separate thread after the response is sent.
    background_tasks.add_task(
        ingest_document,
        doc_id=doc_id,
        filename=filename,
        file_bytes=file_bytes,
        content_type=file.content_type or "application/octet-stream",
    )

    return UploadResponse(
        doc_id=doc_id,
        filename=filename,
        status=DocumentStatus.PENDING,
        message=(
            f"Document '{filename}' accepted for processing. "
            f"Poll GET /upload/status/{doc_id} to track progress."
        ),
    )


@router.get(
    "/status/{doc_id}",
    response_model=DocumentStatusResponse,
    summary="Check document processing status",
    description="Returns the current processing state of an uploaded document.",
)
async def get_document_status(doc_id: str) -> DocumentStatusResponse:
    """
    Return the processing status of a previously uploaded document.

    Clients should poll this endpoint after upload. Once status is 'completed',
    the document is ready to query. If status is 'failed', the error message
    explains why.
    """
    job = get_job(doc_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail=f"Document '{doc_id}' not found. It may not have been uploaded yet.",
        )

    return DocumentStatusResponse(
        doc_id=job.doc_id,
        filename=job.filename,
        status=job.status,
        chunk_count=job.chunk_count,
        error=job.error,
    )
