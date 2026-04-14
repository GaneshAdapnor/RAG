from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


class DocumentStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    ready = "ready"
    failed = "failed"


class ParsedPage(BaseModel):
    page_number: int
    text: str


class ParseResult(BaseModel):
    pages: list[ParsedPage]
    warnings: list[str] = Field(default_factory=list)


class ChunkRecord(BaseModel):
    document_id: str
    filename: str
    chunk_id: str
    text: str
    page_start: int
    page_end: int
    token_count: int
    created_at: datetime = Field(default_factory=utc_now)
    vector_id: int | None = None


class DocumentRecord(BaseModel):
    document_id: str
    filename: str
    file_path: str
    media_type: str
    status: DocumentStatus
    pages_extracted: int = 0
    chunks_created: int = 0
    warnings: list[str] = Field(default_factory=list)
    error: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class SearchResult(BaseModel):
    document_id: str
    filename: str
    chunk_id: str
    text: str
    page_start: int
    page_end: int
    score: float
