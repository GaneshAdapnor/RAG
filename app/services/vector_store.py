from __future__ import annotations

import json
from pathlib import Path
from threading import RLock

import faiss
import numpy as np

from app.models.domain import ChunkRecord, SearchResult
from app.utils.files import atomic_write_json


class FaissVectorStore:
    """
    Flat FAISS index using inner product over L2-normalized embeddings.
    This makes inner product equivalent to cosine similarity while keeping
    implementation simple and deterministic for local deployment.
    """

    def __init__(self, index_path: Path, metadata_path: Path, embedding_dim: int) -> None:
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.embedding_dim = embedding_dim
        self._lock = RLock()
        self._index = self._load_or_create_index()
        self._metadata = self._load_metadata()
        if self._index.ntotal != len(self._metadata):
            raise RuntimeError(
                "FAISS index and vector metadata are out of sync. "
                "Rebuild the index or clear the data/index directory."
            )

    def _load_or_create_index(self) -> faiss.Index:
        if self.index_path.exists():
            return faiss.read_index(str(self.index_path))
        return faiss.IndexFlatIP(self.embedding_dim)

    def _load_metadata(self) -> list[ChunkRecord]:
        if not self.metadata_path.exists():
            return []

        raw_payload = self.metadata_path.read_text(encoding="utf-8")
        if not raw_payload.strip():
            return []

        items = json.loads(raw_payload)
        return [ChunkRecord.model_validate(item) for item in items]

    def _persist(self) -> None:
        faiss.write_index(self._index, str(self.index_path))
        payload = [record.model_dump(mode="json") for record in self._metadata]
        atomic_write_json(self.metadata_path, payload)

    def add_embeddings(self, chunks: list[ChunkRecord], embeddings: np.ndarray) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("Chunk metadata and embeddings must have the same length.")
        if embeddings.ndim != 2 or embeddings.shape[1] != self.embedding_dim:
            raise ValueError("Embedding array shape does not match the FAISS index dimension.")

        with self._lock:
            start_id = self._index.ntotal
            self._index.add(embeddings)

            for offset, chunk in enumerate(chunks):
                vector_id = int(start_id + offset)
                self._metadata.append(chunk.model_copy(update={"vector_id": vector_id}))

            self._persist()

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int,
        document_ids: list[str] | None = None,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        with self._lock:
            if self._index.ntotal == 0:
                return []

            candidate_count = self._index.ntotal if document_ids else min(self._index.ntotal, max(top_k * 5, top_k))
            scores, indices = self._index.search(query_embedding.reshape(1, -1), candidate_count)

            allowed_ids = set(document_ids or [])
            results: list[SearchResult] = []

            for score, index in zip(scores[0], indices[0]):
                if index < 0:
                    continue

                metadata = self._metadata[index]
                if allowed_ids and metadata.document_id not in allowed_ids:
                    continue
                if score < min_score:
                    continue

                results.append(
                    SearchResult(
                        document_id=metadata.document_id,
                        filename=metadata.filename,
                        chunk_id=metadata.chunk_id,
                        text=metadata.text,
                        page_start=metadata.page_start,
                        page_end=metadata.page_end,
                        score=float(score),
                    )
                )
                if len(results) == top_k:
                    break

            return results

    def stats(self) -> dict[str, int]:
        with self._lock:
            unique_document_ids = {item.document_id for item in self._metadata}
            return {
                "indexed_documents": len(unique_document_ids),
                "indexed_chunks": len(self._metadata),
            }
