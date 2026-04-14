from __future__ import annotations

import logging

from app.models.domain import DocumentStatus
from app.services.chunker import TextChunker
from app.services.document_parser import DocumentParser
from app.services.document_registry import DocumentRegistry
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import FaissVectorStore

logger = logging.getLogger(__name__)


class DocumentIngestionService:
    def __init__(
        self,
        registry: DocumentRegistry,
        parser: DocumentParser,
        chunker: TextChunker,
        embedding_service: EmbeddingService,
        vector_store: FaissVectorStore,
    ) -> None:
        self.registry = registry
        self.parser = parser
        self.chunker = chunker
        self.embedding_service = embedding_service
        self.vector_store = vector_store

    def ingest_document(self, document_id: str) -> None:
        record = self.registry.get_document(document_id)
        if record is None:
            logger.error("Skipping ingestion for unknown document_id=%s", document_id)
            return

        logger.info("Starting ingestion for document_id=%s filename=%s", document_id, record.filename)
        self.registry.update_status(document_id, DocumentStatus.processing)

        try:
            parse_result = self.parser.parse(record.file_path)
            chunks = self.chunker.chunk_pages(
                document_id=document_id,
                filename=record.filename,
                pages=parse_result.pages,
            )

            if not chunks:
                raise ValueError("No indexable text was extracted from the uploaded document.")

            embeddings = self.embedding_service.embed_texts([chunk.text for chunk in chunks])
            self.vector_store.add_embeddings(chunks, embeddings)

            self.registry.mark_ready(
                document_id=document_id,
                pages_extracted=len(parse_result.pages),
                chunks_created=len(chunks),
                warnings=parse_result.warnings,
            )
            logger.info(
                "Completed ingestion for document_id=%s pages=%s chunks=%s",
                document_id,
                len(parse_result.pages),
                len(chunks),
            )
        except Exception as exc:  # pragma: no cover - defensive production path
            logger.exception("Document ingestion failed for document_id=%s", document_id)
            self.registry.mark_failed(document_id=document_id, error=str(exc))
