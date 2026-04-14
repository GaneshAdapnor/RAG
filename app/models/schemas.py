"""Compatibility exports for earlier schema module paths."""

from app.models.api import HealthResponse, QueryRequest, QueryResponse, SourceChunkResponse, UploadResponse
from app.models.domain import DocumentRecord, DocumentStatus, SearchResult

RetrievedChunk = SourceChunkResponse

__all__ = [
    "DocumentRecord",
    "DocumentStatus",
    "HealthResponse",
    "QueryRequest",
    "QueryResponse",
    "RetrievedChunk",
    "SearchResult",
    "SourceChunkResponse",
    "UploadResponse",
]
