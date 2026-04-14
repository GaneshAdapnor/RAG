from __future__ import annotations

from functools import lru_cache

from fastapi import Depends, HTTPException, Request, status

from app.core.config import Settings, get_settings
from app.services.chunker import TextChunker
from app.services.document_ingestion import DocumentIngestionService
from app.services.document_parser import DocumentParser
from app.services.document_registry import DocumentRegistry
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
from app.services.metrics_service import MetricsService
from app.services.query_service import QueryService
from app.services.vector_store import FaissVectorStore
from app.utils.rate_limiter import TokenBucketRateLimiter


@lru_cache
def get_document_registry() -> DocumentRegistry:
    settings = get_settings()
    return DocumentRegistry(settings.documents_path)


@lru_cache
def get_metrics_service() -> MetricsService:
    return MetricsService()


@lru_cache
def get_embedding_service() -> EmbeddingService:
    settings = get_settings()
    return EmbeddingService(
        model_name=settings.embedding_model_name,
        embedding_dim=settings.embedding_dim,
        batch_size=settings.embedding_batch_size,
    )


@lru_cache
def get_vector_store() -> FaissVectorStore:
    settings = get_settings()
    return FaissVectorStore(
        index_path=settings.faiss_index_path,
        metadata_path=settings.vector_metadata_path,
        embedding_dim=settings.embedding_dim,
    )


@lru_cache
def get_document_parser() -> DocumentParser:
    return DocumentParser()


@lru_cache
def get_text_chunker() -> TextChunker:
    settings = get_settings()
    return TextChunker(
        chunk_size_tokens=settings.chunk_size_tokens,
        chunk_overlap_tokens=settings.chunk_overlap_tokens,
    )


@lru_cache
def get_llm_service() -> LLMService:
    settings = get_settings()
    return LLMService(
        api_key=settings.openai_api_key,
        model_name=settings.llm_model_name,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
        enable_extractive_fallback=settings.enable_extractive_fallback,
    )


@lru_cache
def get_ingestion_service() -> DocumentIngestionService:
    return DocumentIngestionService(
        registry=get_document_registry(),
        parser=get_document_parser(),
        chunker=get_text_chunker(),
        embedding_service=get_embedding_service(),
        vector_store=get_vector_store(),
    )


@lru_cache
def get_query_service() -> QueryService:
    settings = get_settings()
    return QueryService(
        settings=settings,
        registry=get_document_registry(),
        embedding_service=get_embedding_service(),
        vector_store=get_vector_store(),
        llm_service=get_llm_service(),
        metrics_service=get_metrics_service(),
    )


@lru_cache
def get_rate_limiter() -> TokenBucketRateLimiter:
    settings = get_settings()
    return TokenBucketRateLimiter(
        capacity=settings.rate_limit_calls,
        refill_period_seconds=settings.rate_limit_period_seconds,
    )


def get_client_identifier(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    if request.client and request.client.host:
        return request.client.host
    return "anonymous"


def enforce_rate_limit(
    request: Request,
    limiter: TokenBucketRateLimiter = Depends(get_rate_limiter),
) -> None:
    client_id = get_client_identifier(request)
    allowed, retry_after_seconds = limiter.consume(client_id)
    if allowed:
        return

    raise HTTPException(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        detail=(
            f"Rate limit exceeded for client '{client_id}'. "
            f"Retry in {retry_after_seconds:.1f} seconds."
        ),
        headers={"Retry-After": f"{retry_after_seconds:.0f}"},
    )
