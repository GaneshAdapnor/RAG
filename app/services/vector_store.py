"""
FAISS vector store: index management and persistence.

WHY FAISS:
    - Facebook AI Similarity Search is the industry standard for local dense
      vector retrieval. It is written in C++ with Python bindings.
    - faiss-cpu handles up to ~1M vectors with sub-10ms query time on a single
      core — sufficient for a document QA service.
    - No external service required (unlike Pinecone, Weaviate, Qdrant). This
      keeps the system self-contained and free to run.

INDEX CHOICE — IndexFlatIP:
    - "Flat" = exact (brute-force) search — no approximation, no quantization.
    - "IP" = inner product distance.
    - Since all embeddings are L2-normalized (done in embedding_service.py),
      inner_product(a, b) = cos(a, b). So we get exact cosine similarity.

    WHY NOT HNSW / IVF (approximate indexes)?
    - Approximate indexes (IndexHNSWFlat, IndexIVFFlat) trade accuracy for
      speed. At < 500K vectors, exact search on CPU finishes in < 5ms per
      query — no approximation needed.
    - IVF requires training (k-means), which complicates incremental updates.
    - HNSW uses significantly more RAM (graph structure overhead).
    - We can upgrade to HNSW at 500K+ vectors with a single line change.

DISTANCE METRIC TRADE-OFFS:
    L2 (Euclidean):  Sensitive to vector magnitude. Good when vectors are not
                     normalized. IndexFlatL2.
    Cosine:          Direction-only; ignores magnitude. Best for text. Achieved
                     here via L2-normalize + IndexFlatIP.
    Dot product:     Favors longer sequences (higher magnitude). Not suitable
                     without normalization.

METADATA STORAGE:
    FAISS stores only float32 vectors, not metadata. We maintain a parallel
    JSON file where `metadata[i]` corresponds to FAISS vector at index `i`.
    This gives O(1) lookup by FAISS index position and simple persistence.
    For > 1M chunks, replace the JSON with SQLite (one line change).

THREAD SAFETY:
    The write lock (threading.Lock) ensures that concurrent ingestion requests
    don't corrupt the FAISS index. Reads (searches) are lock-free because
    FAISS IndexFlatIP.search() is thread-safe in the CPU variant.
"""

import json
import logging
import os
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ChunkMetadata:
    """
    Metadata record stored alongside each FAISS vector.

    The FAISS index position (integer 0, 1, 2, …) is the implicit key.
    `doc_id` + `chunk_id` is the application-level primary key.
    """
    doc_id: str
    chunk_id: int
    filename: str
    page: Optional[int]
    text: str


class FAISSVectorStore:
    """
    Manages a FAISS IndexFlatIP index and its associated metadata.

    Lifecycle:
        - Instantiate once at app startup (via get_vector_store()).
        - load() restores a persisted index from disk.
        - add_vectors() is called by the ingestion pipeline.
        - search() is called by the retrieval service.
        - save() persists index + metadata after each successful ingestion.
    """

    def __init__(self, dim: int, index_path: str, metadata_path: str):
        self.dim = dim
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self._write_lock = threading.Lock()

        # Initialize a fresh flat inner-product index
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(dim)
        self.metadata: List[Dict[str, Any]] = []

        logger.info(
            "FAISSVectorStore initialized (dim=%d, index_path=%s).",
            dim, self.index_path,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def load(self) -> None:
        """
        Load an existing index and metadata from disk.

        If either file is missing or corrupt, starts fresh (logged at WARNING).
        Called once during application startup.
        """
        index_exists = self.index_path.exists()
        meta_exists = self.metadata_path.exists()

        if not (index_exists and meta_exists):
            logger.warning(
                "No persisted vector store found at '%s'. Starting with empty index.",
                self.index_path,
            )
            return

        try:
            loaded_index = faiss.read_index(str(self.index_path))
            # Validate that the loaded index dimension matches configuration
            if loaded_index.d != self.dim:
                logger.error(
                    "Persisted index dimension (%d) != config dimension (%d). "
                    "Discarding stale index.",
                    loaded_index.d, self.dim,
                )
                return

            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

            self.index = loaded_index
            logger.info(
                "Loaded vector store: %d vectors from '%s'.",
                self.index.ntotal, self.index_path,
            )

            # Sanity check: metadata rows must match vector count
            if len(self.metadata) != self.index.ntotal:
                logger.error(
                    "Metadata count (%d) != FAISS vector count (%d). "
                    "Index may be corrupt. Resetting.",
                    len(self.metadata), self.index.ntotal,
                )
                self._reset()

        except Exception as exc:
            logger.error(
                "Failed to load vector store from '%s': %s. Starting fresh.",
                self.index_path, exc,
            )
            self._reset()

    def save(self) -> None:
        """
        Persist index and metadata to disk atomically.

        We write metadata to a temp file first, then rename. This prevents a
        half-written file from corrupting the store on crash or SIGKILL.
        """
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Write FAISS index
            faiss.write_index(self.index, str(self.index_path))

            # Write metadata atomically via temp file + rename
            tmp_path = self.metadata_path.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, ensure_ascii=False)
            os.replace(tmp_path, self.metadata_path)

            logger.info(
                "Vector store saved: %d vectors → '%s'.",
                self.index.ntotal, self.index_path,
            )
        except Exception as exc:
            logger.error("Failed to save vector store: %s", exc)
            raise

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def add_vectors(
        self,
        vectors: np.ndarray,
        metadata_records: List[ChunkMetadata],
    ) -> None:
        """
        Add a batch of vectors and their metadata to the index.

        Args:
            vectors: float32 array of shape (N, dim), L2-normalized.
            metadata_records: List of N ChunkMetadata objects.

        Raises:
            ValueError: If vector shape or metadata length doesn't match.
        """
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(
                f"Expected vectors shape (N, {self.dim}), got {vectors.shape}."
            )
        if len(vectors) != len(metadata_records):
            raise ValueError(
                f"vectors length ({len(vectors)}) != metadata length ({len(metadata_records)})."
            )

        vectors = vectors.astype(np.float32)

        with self._write_lock:
            self.index.add(vectors)
            self.metadata.extend([asdict(m) for m in metadata_records])
            logger.info(
                "Added %d vectors. Total in index: %d.",
                len(vectors), self.index.ntotal,
            )

    def remove_document(self, doc_id: str) -> int:
        """
        Remove all vectors belonging to a document by rebuilding the index.

        FAISS IndexFlatIP does not support in-place deletion. The standard
        approach is to rebuild — O(N) but acceptable since documents are
        rarely deleted vs. queried.

        Returns:
            Number of chunks removed.
        """
        with self._write_lock:
            original_count = self.index.ntotal

            # Identify which positions to keep
            keep_indices = [
                i for i, m in enumerate(self.metadata) if m["doc_id"] != doc_id
            ]
            removed = original_count - len(keep_indices)

            if removed == 0:
                logger.warning("remove_document: doc_id='%s' not found in index.", doc_id)
                return 0

            # Rebuild: extract surviving vectors and re-add
            if keep_indices:
                all_vectors = self._reconstruct_all_vectors()
                surviving_vectors = all_vectors[keep_indices]
                surviving_metadata = [self.metadata[i] for i in keep_indices]
            else:
                surviving_vectors = np.empty((0, self.dim), dtype=np.float32)
                surviving_metadata = []

            # Reset and repopulate
            self._reset()
            if len(surviving_vectors) > 0:
                self.index.add(surviving_vectors)
                self.metadata = surviving_metadata

            logger.info(
                "Removed %d chunks for doc_id='%s'. Index now has %d vectors.",
                removed, doc_id, self.index.ntotal,
            )
            return removed

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int,
        doc_ids: Optional[List[str]] = None,
    ) -> List[Tuple[ChunkMetadata, float]]:
        """
        Find the top-K most similar chunks to the query vector.

        Args:
            query_vector: 1D float32 array of shape (dim,).
            top_k: Number of results to return.
            doc_ids: Optional whitelist of doc_ids to filter results.
                     Applied post-search: we over-fetch and filter.

        Returns:
            List of (ChunkMetadata, cosine_similarity_score) tuples,
            ordered by descending similarity.

        HOW top_k IS CHOSEN:
            The default top_k=5 is a practical balance:
            - Too low (1–2): High miss rate when the key information spans
              multiple chunks.
            - Too high (10+): Increases LLM prompt length, adding cost and
              risk of "lost in the middle" failure (LLMs attend poorly to
              context buried in the middle of long prompts).
            - 3–5 retrieved chunks covers ≈ 1,500–2,500 characters —
              enough context for most factual questions.
        """
        if self.index.ntotal == 0:
            logger.warning("search() called on an empty index.")
            return []

        query_vector = query_vector.astype(np.float32).reshape(1, -1)

        # Over-fetch when filtering by doc_id to account for filtered-out results
        fetch_k = top_k * 5 if doc_ids else top_k
        fetch_k = min(fetch_k, self.index.ntotal)

        scores, indices = self.index.search(query_vector, fetch_k)

        results: List[Tuple[ChunkMetadata, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS sentinel for "no result"
                continue
            meta = self.metadata[idx]
            if doc_ids is not None and meta["doc_id"] not in doc_ids:
                continue

            # Clamp score to [0, 1] — floating point can produce 1.0000002 etc.
            clamped_score = float(min(max(score, 0.0), 1.0))
            results.append((ChunkMetadata(**meta), clamped_score))

            if len(results) >= top_k:
                break

        logger.debug(
            "FAISS search: top_k=%d, fetched=%d, returned=%d.",
            top_k, fetch_k, len(results),
        )
        return results

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def total_chunks(self) -> int:
        return self.index.ntotal

    @property
    def total_documents(self) -> int:
        if not self.metadata:
            return 0
        return len({m["doc_id"] for m in self.metadata})

    def get_document_chunk_count(self, doc_id: str) -> int:
        return sum(1 for m in self.metadata if m["doc_id"] == doc_id)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _reset(self) -> None:
        """Discard the current index and start fresh."""
        self.index = faiss.IndexFlatIP(self.dim)
        self.metadata = []

    def _reconstruct_all_vectors(self) -> np.ndarray:
        """
        Reconstruct all stored vectors from the index.

        IndexFlatIP inherits from IndexFlat which stores raw vectors and
        supports reconstruct(). This is only used by remove_document().
        """
        n = self.index.ntotal
        vectors = np.empty((n, self.dim), dtype=np.float32)
        for i in range(n):
            self.index.reconstruct(i, vectors[i])
        return vectors


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_vector_store: Optional[FAISSVectorStore] = None


def get_vector_store() -> FAISSVectorStore:
    """Return the application-level singleton vector store."""
    global _vector_store
    if _vector_store is None:
        _vector_store = FAISSVectorStore(
            dim=settings.EMBEDDING_DIM,
            index_path=settings.FAISS_INDEX_PATH,
            metadata_path=settings.METADATA_PATH,
        )
    return _vector_store
