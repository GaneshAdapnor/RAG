"""Compatibility chunking helpers backed by the new chunker."""

from __future__ import annotations

from dataclasses import dataclass

from app.models.domain import ParsedPage
from app.services.chunker import TextChunker


@dataclass
class TextChunk:
    doc_id: str
    chunk_id: int
    text: str
    page: int | None
    char_start: int
    char_end: int


def chunk_pages(
    pages: list[object],
    doc_id: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[TextChunk]:
    parsed_pages = [
        ParsedPage(page_number=getattr(page, "page", None) or 1, text=getattr(page, "text", ""))
        for page in pages
    ]
    chunker = TextChunker(chunk_size_tokens=chunk_size, chunk_overlap_tokens=chunk_overlap)
    chunks = chunker.chunk_pages(document_id=doc_id, filename=doc_id, pages=parsed_pages)

    compatible_chunks: list[TextChunk] = []
    cursor = 0
    for index, chunk in enumerate(chunks):
        char_start = cursor
        char_end = cursor + len(chunk.text)
        compatible_chunks.append(
            TextChunk(
                doc_id=doc_id,
                chunk_id=index,
                text=chunk.text,
                page=chunk.page_start,
                char_start=char_start,
                char_end=char_end,
            )
        )
        cursor = max(char_start, char_end - chunk_overlap)
    return compatible_chunks


__all__ = ["TextChunk", "chunk_pages"]
