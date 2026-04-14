from __future__ import annotations

from fastapi import APIRouter, Depends

from app.core.config import Settings, get_settings
from app.core.dependencies import get_document_registry, get_metrics_service, get_vector_store
from app.models.api import HealthResponse
from app.services.document_registry import DocumentRegistry
from app.services.metrics_service import MetricsService
from app.services.vector_store import FaissVectorStore

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health_check(
    settings: Settings = Depends(get_settings),
    registry: DocumentRegistry = Depends(get_document_registry),
    metrics: MetricsService = Depends(get_metrics_service),
    vector_store: FaissVectorStore = Depends(get_vector_store),
) -> HealthResponse:
    vector_stats = vector_store.stats()
    return HealthResponse(
        status="ok",
        version=settings.version,
        indexed_documents=vector_stats["indexed_documents"],
        indexed_chunks=vector_stats["indexed_chunks"],
        documents_by_status=registry.counts_by_status(),
        average_query_latency_ms=metrics.average_query_latency_ms(),
    )
