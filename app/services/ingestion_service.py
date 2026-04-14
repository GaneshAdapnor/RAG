"""Compatibility ingestion helpers layered on top of the new service stack."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from app.core.config import get_settings
from app.core.dependencies import get_document_registry, get_ingestion_service
from app.models.domain import DocumentStatus
from app.utils.files import sanitize_filename


@dataclass
class JobStatus:
    doc_id: str
    filename: str
    status: DocumentStatus
    chunk_count: int = 0
    error: str | None = None


def create_doc_id() -> str:
    return uuid4().hex


def get_job(doc_id: str) -> JobStatus | None:
    registry = get_document_registry()
    record = registry.get_document(doc_id)
    if record is None:
        return None

    return JobStatus(
        doc_id=record.document_id,
        filename=record.filename,
        status=record.status,
        chunk_count=record.chunks_created,
        error=record.error,
    )


def list_jobs() -> dict[str, JobStatus]:
    registry = get_document_registry()
    return {
        record.document_id: JobStatus(
            doc_id=record.document_id,
            filename=record.filename,
            status=record.status,
            chunk_count=record.chunks_created,
            error=record.error,
        )
        for record in registry.list_documents()
    }


def validate_upload(filename: str, file_size: int, content_type: str) -> None:
    settings = get_settings()
    if file_size <= 0:
        raise ValueError("Uploaded file is empty.")
    if file_size > settings.max_upload_bytes:
        raise ValueError(
            f"File exceeds the maximum size of {settings.max_upload_bytes} bytes."
        )

    suffix = Path(filename).suffix.lower()
    if suffix not in settings.supported_extensions:
        raise ValueError(
            f"Unsupported file type '{content_type}' (extension '{suffix}'). "
            "Only PDF and TXT files are accepted."
        )


def ingest_document(
    doc_id: str,
    filename: str,
    file_bytes: bytes,
    content_type: str,
) -> None:
    settings = get_settings()
    registry = get_document_registry()
    ingestion_service = get_ingestion_service()

    safe_filename = sanitize_filename(filename)
    destination = settings.uploads_dir / f"{doc_id}_{safe_filename}"
    destination.write_bytes(file_bytes)

    if registry.get_document(doc_id) is None:
        registry.create_document(
            document_id=doc_id,
            filename=safe_filename,
            file_path=str(destination),
            media_type=content_type,
        )

    ingestion_service.ingest_document(doc_id)


__all__ = [
    "JobStatus",
    "create_doc_id",
    "get_job",
    "ingest_document",
    "list_jobs",
    "validate_upload",
]
