from __future__ import annotations

import time

from fastapi import HTTPException, status

from app.core.config import Settings
from app.models.api import QueryRequest, QueryResponse, SourceChunkResponse
from app.models.domain import DocumentStatus
from app.services.document_registry import DocumentRegistry
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
from app.services.metrics_service import MetricsService
from app.services.vector_store import FaissVectorStore


class QueryService:
    def __init__(
        self,
        settings: Settings,
        registry: DocumentRegistry,
        embedding_service: EmbeddingService,
        vector_store: FaissVectorStore,
        llm_service: LLMService,
        metrics_service: MetricsService,
    ) -> None:
        self.settings = settings
        self.registry = registry
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.llm_service = llm_service
        self.metrics_service = metrics_service

    def answer(self, request: QueryRequest) -> QueryResponse:
        document_ids = request.document_ids or self._ready_document_ids()
        if not document_ids:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="No ready documents are available. Upload and ingest at least one document first.",
            )

        self._validate_document_scope(document_ids)

        top_k = request.top_k or self.settings.top_k_retrieval
        start_time = time.perf_counter()

        query_embedding = self.embedding_service.embed_query(request.question)
        search_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            document_ids=document_ids,
            min_score=self.settings.search_min_score,
        )

        answer, answer_model = self.llm_service.generate_answer(request.question, search_results)
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        self.metrics_service.record_query_latency(latency_ms)

        sources = [
            SourceChunkResponse(
                document_id=result.document_id,
                filename=result.filename,
                chunk_id=result.chunk_id,
                page_start=result.page_start,
                page_end=result.page_end,
                score=round(result.score, 4),
                excerpt=result.text[:320],
            )
            for result in search_results
        ]

        return QueryResponse(
            question=request.question,
            answer=answer,
            answer_model=answer_model,
            latency_ms=latency_ms,
            retrieved_chunks=len(sources),
            documents_considered=document_ids,
            sources=sources,
        )

    def _ready_document_ids(self) -> list[str]:
        ready_docs = self.registry.list_documents(status=DocumentStatus.ready)
        return [record.document_id for record in ready_docs]

    def _validate_document_scope(self, document_ids: list[str]) -> None:
        missing = []
        non_ready = []

        for document_id in document_ids:
            record = self.registry.get_document(document_id)
            if record is None:
                missing.append(document_id)
                continue
            if record.status != DocumentStatus.ready:
                non_ready.append(document_id)

        if missing:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Unknown document_ids: {', '.join(missing)}",
            )

        if non_ready:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Documents are not ready yet: {', '.join(non_ready)}",
            )
