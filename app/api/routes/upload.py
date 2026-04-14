from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile, status

from app.core.config import Settings, get_settings
from app.core.dependencies import enforce_rate_limit, get_document_registry, get_ingestion_service
from app.models.api import UploadResponse
from app.models.domain import DocumentStatus
from app.services.document_ingestion import DocumentIngestionService
from app.services.document_registry import DocumentRegistry
from app.utils.files import is_supported_extension, sanitize_filename, save_upload_file

router = APIRouter(tags=["documents"])


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(enforce_rate_limit)],
)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
    registry: DocumentRegistry = Depends(get_document_registry),
    ingestion_service: DocumentIngestionService = Depends(get_ingestion_service),
) -> UploadResponse:
    original_name = file.filename or "document"
    if not is_supported_extension(original_name, settings.supported_extensions):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type. Supported extensions: {', '.join(settings.supported_extensions)}",
        )

    document_id = uuid4().hex
    safe_filename = sanitize_filename(original_name)
    destination = Path(settings.uploads_dir) / f"{document_id}_{safe_filename}"
    media_type = file.content_type or "application/octet-stream"
    bytes_written = await save_upload_file(file, destination, settings.max_upload_bytes)

    registry.create_document(
        document_id=document_id,
        filename=safe_filename,
        file_path=str(destination),
        media_type=media_type,
    )
    background_tasks.add_task(ingestion_service.ingest_document, document_id)

    return UploadResponse(
        document_id=document_id,
        filename=safe_filename,
        status=DocumentStatus.pending,
        bytes_written=bytes_written,
        message="Document accepted and queued for background ingestion.",
    )
