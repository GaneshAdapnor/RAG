from __future__ import annotations

from threading import Lock

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    def __init__(self, model_name: str, embedding_dim: int, batch_size: int) -> None:
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.batch_size = batch_size
        self._model: SentenceTransformer | None = None
        self._lock = Lock()

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        model = self._get_model()
        embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        embeddings = embeddings.astype(np.float32)

        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {self.embedding_dim}, got {embeddings.shape[1]}"
            )

        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        embedding = self.embed_texts([query])
        return embedding[0]
