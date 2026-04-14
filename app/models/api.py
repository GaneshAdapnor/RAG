from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from app.models.domain import DocumentStatus


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    status: DocumentStatus
    bytes_written: int
    message: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=10)
    document_ids: list[str] | None = None

    @field_validator("document_ids")
    @classmethod
    def deduplicate_document_ids(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return value
        cleaned = [item.strip() for item in value if item and item.strip()]
        return list(dict.fromkeys(cleaned)) or None


class SourceChunkResponse(BaseModel):
    document_id: str
    filename: str
    chunk_id: str
    page_start: int
    page_end: int
    score: float
    excerpt: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    answer_model: str
    latency_ms: float
    retrieved_chunks: int
    documents_considered: list[str]
    sources: list[SourceChunkResponse]


class HealthResponse(BaseModel):
    status: str
    version: str
    indexed_documents: int
    indexed_chunks: int
    documents_by_status: dict[str, int]
    average_query_latency_ms: float | None
