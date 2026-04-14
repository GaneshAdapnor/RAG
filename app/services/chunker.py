from __future__ import annotations

import re

from app.models.domain import ChunkRecord, ParsedPage
from app.utils.text import normalize_whitespace


class TextChunker:
    """Explicit sliding-window chunker over whitespace tokens."""

    def __init__(self, chunk_size_tokens: int, chunk_overlap_tokens: int) -> None:
        if chunk_overlap_tokens >= chunk_size_tokens:
            raise ValueError("Chunk overlap must be smaller than chunk size.")

        self.chunk_size_tokens = chunk_size_tokens
        self.chunk_overlap_tokens = chunk_overlap_tokens

    def chunk_pages(
        self,
        document_id: str,
        filename: str,
        pages: list[ParsedPage],
    ) -> list[ChunkRecord]:
        token_stream: list[tuple[str, int]] = []

        for page in pages:
            tokens = re.findall(r"\S+", page.text)
            token_stream.extend((token, page.page_number) for token in tokens)

        if not token_stream:
            return []

        chunks: list[ChunkRecord] = []
        start = 0
        chunk_index = 0

        while start < len(token_stream):
            end = min(start + self.chunk_size_tokens, len(token_stream))
            window = token_stream[start:end]
            chunk_text = normalize_whitespace(" ".join(token for token, _ in window))
            page_numbers = [page for _, page in window]

            chunks.append(
                ChunkRecord(
                    document_id=document_id,
                    filename=filename,
                    chunk_id=f"chunk-{chunk_index:05d}",
                    text=chunk_text,
                    page_start=min(page_numbers),
                    page_end=max(page_numbers),
                    token_count=len(window),
                )
            )

            if end == len(token_stream):
                break

            start = end - self.chunk_overlap_tokens
            chunk_index += 1

        return chunks
