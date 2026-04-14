"""
Pydantic v2 schemas for all API request and response models.

Pydantic v2 is used (not v1) because it's significantly faster for validation
(written in Rust via pydantic-core) and the json_schema_extra / model_config
API is cleaner. Every field has an explicit description so the auto-generated
OpenAPI spec at /docs is self-documenting without extra annotation.
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DocumentStatus(str, Enum):
    """Processing lifecycle states for an uploaded document."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Upload schemas
# ---------------------------------------------------------------------------

class UploadResponse(BaseModel):
    """Response returned immediately after a file is accepted for processing."""

    doc_id: str = Field(
        ...,
        description="Unique identifier for the uploaded document (UUID4).",
        examples=["3fa85f64-5717-4562-b3fc-2c963f66afa6"],
    )
    filename: str = Field(..., description="Original filename as uploaded.")
    status: DocumentStatus = Field(
        default=DocumentStatus.PENDING,
        description="Current processing status. Ingestion happens in the background.",
    )
    message: str = Field(..., description="Human-readable status message.")


class DocumentStatusResponse(BaseModel):
    """Status of a previously uploaded document."""

    doc_id: str
    filename: str
    status: DocumentStatus
    chunk_count: int = Field(
        default=0,
        description="Number of text chunks stored in the vector index.",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if processing failed.",
    )


# ---------------------------------------------------------------------------
# Query schemas
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    """User query payload."""

    query: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="The natural language question to answer from the document corpus.",
        examples=["What are the main findings of the study?"],
    )
    top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description=(
            "Number of document chunks to retrieve (overrides server default). "
            "Higher values increase recall but risk diluting the context."
        ),
    )
    doc_ids: Optional[List[str]] = Field(
        default=None,
        description=(
            "Restrict retrieval to specific document IDs. "
            "Leave null to search across all indexed documents."
        ),
    )

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Query must not be blank or whitespace only.")
        return v.strip()


class RetrievedChunk(BaseModel):
    """A single document chunk returned from the vector store."""

    doc_id: str = Field(..., description="Source document ID.")
    filename: str = Field(..., description="Source document filename.")
    chunk_id: int = Field(..., description="Zero-based chunk index within the document.")
    page: Optional[int] = Field(
        default=None,
        description="Page number (1-based) for PDF sources; null for TXT.",
    )
    text: str = Field(..., description="Raw text of the retrieved chunk.")
    score: float = Field(
        ...,
        description=(
            "Cosine similarity score in [0, 1]. Higher = more semantically similar."
        ),
    )


class QueryResponse(BaseModel):
    """Full answer response including source attribution."""

    query: str = Field(..., description="The original query string.")
    answer: str = Field(..., description="LLM-generated answer grounded in retrieved chunks.")
    sources: List[RetrievedChunk] = Field(
        ...,
        description="The chunks used as context for answer generation.",
    )
    retrieval_latency_ms: float = Field(
        ...,
        description="Time spent on embedding + FAISS search, in milliseconds.",
    )
    generation_latency_ms: float = Field(
        ...,
        description="Time spent on LLM generation, in milliseconds.",
    )


# ---------------------------------------------------------------------------
# Health schema
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """System health check response."""

    status: str = Field(..., description="'ok' if all components are healthy.")
    version: str
    embedding_model: str
    indexed_chunks: int = Field(
        ...,
        description="Total number of vectors currently in the FAISS index.",
    )
    indexed_documents: int = Field(
        ...,
        description="Total number of unique documents ingested.",
    )


# ---------------------------------------------------------------------------
# Error schema
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    """Standard error envelope returned for 4xx/5xx responses."""

    detail: str
    error_code: Optional[str] = None
