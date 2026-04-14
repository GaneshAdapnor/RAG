"""Compatibility retrieval helpers layered on top of the new service stack."""

from __future__ import annotations

import time

from app.core.config import get_settings
from app.core.dependencies import get_embedding_service, get_vector_store
from app.models.schemas import RetrievedChunk


def retrieve_chunks(
    query: str,
    top_k: int | None = None,
    doc_ids: list[str] | None = None,
) -> tuple[list[RetrievedChunk], float]:
    settings = get_settings()
    vector_store = get_vector_store()
    embedding_service = get_embedding_service()

    start_time = time.perf_counter()
    query_vector = embedding_service.embed_query(query)
    results = vector_store.search(
        query_embedding=query_vector,
        top_k=top_k or settings.top_k_retrieval,
        document_ids=doc_ids,
        min_score=settings.search_min_score,
    )
    latency_ms = (time.perf_counter() - start_time) * 1000

    chunks = [
        RetrievedChunk(
            document_id=result.document_id,
            filename=result.filename,
            chunk_id=result.chunk_id,
            page_start=result.page_start,
            page_end=result.page_end,
            score=round(result.score, 4),
            excerpt=result.text[:320],
        )
        for result in results
    ]
    return chunks, latency_ms


def build_context(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return ""

    return "\n\n".join(
        (
            f"[{index}] document_id={chunk.document_id} "
            f"filename={chunk.filename} "
            f"pages={chunk.page_start}-{chunk.page_end} "
            f"score={chunk.score:.4f}\n"
            f"{chunk.excerpt}"
        )
        for index, chunk in enumerate(chunks, start=1)
    )


__all__ = ["build_context", "retrieve_chunks"]
