"""
Embedding service using sentence-transformers.

WHY sentence-transformers / all-MiniLM-L6-v2:
    - Free, runs locally — no external API call per embedding, which avoids
      latency spikes and cost accumulation at scale.
    - all-MiniLM-L6-v2 is a 22M-parameter distilled model. It produces 384-dim
      vectors. On CPU it encodes ~2,000 sentences/second, which is sufficient
      for background document processing.
    - Benchmark (SBERT MTEB): NDCG@10 = 0.411 on BEIR — competitive with much
      larger models for information retrieval tasks.
    - Deterministic: same text always produces the same vector, so we could add
      a layer-1 cache (Redis/disk) later with no architectural change.

WHY NOT OpenAI text-embedding-ada-002:
    - Costs ~$0.10 per 1M tokens — adds up at scale.
    - Requires internet connectivity (latency + availability risk).
    - Returns 1,536-dim vectors — FAISS index would be 4× larger.
    - For a self-contained system, local embeddings are the right default.

THREAD SAFETY:
    SentenceTransformer.encode() is thread-safe in inference mode (no
    gradient tracking). We load the model once at startup (singleton pattern)
    and reuse it across all requests, avoiding the 2–5 second model load
    overhead on each call.
"""

import logging
import threading
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import settings

logger = logging.getLogger(__name__)

# Module-level singleton — loaded once, shared across threads
_model: SentenceTransformer | None = None
_model_lock = threading.Lock()


def get_embedding_model() -> SentenceTransformer:
    """
    Return the singleton SentenceTransformer model, loading it if necessary.

    Thread-safe: uses a double-checked lock so parallel requests during startup
    don't each try to load the model simultaneously (which would OOM on small
    instances).
    """
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:  # Re-check after acquiring lock
                logger.info(
                    "Loading embedding model '%s'. This may take 10–30 seconds on first run.",
                    settings.EMBEDDING_MODEL_NAME,
                )
                _model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
                logger.info(
                    "Embedding model '%s' loaded. Output dim: %d.",
                    settings.EMBEDDING_MODEL_NAME,
                    settings.EMBEDDING_DIM,
                )
    return _model


def embed_texts(texts: List[str], batch_size: int = 64) -> np.ndarray:
    """
    Generate L2-normalized embeddings for a list of texts.

    WHY NORMALIZE:
        We use FAISS IndexFlatIP (inner product). For unit-normalized vectors,
        inner product == cosine similarity. Normalization is done here (not
        inside FAISS) so that query-time embeddings follow the same pipeline
        without additional FAISS configuration.

    Args:
        texts: List of raw text strings. Must be non-empty.
        batch_size: How many texts to encode per forward pass. 64 is a safe
                    default for CPU inference with ~500-char inputs.

    Returns:
        np.ndarray of shape (len(texts), EMBEDDING_DIM), dtype=float32,
        L2-normalized row-wise.

    Raises:
        ValueError: If texts is empty.
        RuntimeError: If embedding fails (model error).
    """
    if not texts:
        raise ValueError("embed_texts received an empty list.")

    model = get_embedding_model()

    logger.debug("Encoding %d texts (batch_size=%d).", len(texts), batch_size)

    # show_progress_bar=False for production (no stdout noise)
    embeddings: np.ndarray = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2-normalize for cosine via inner product
    )

    # Ensure float32 — FAISS requires it
    embeddings = embeddings.astype(np.float32)

    logger.debug(
        "Encoded %d texts → shape %s, dtype %s.",
        len(texts), embeddings.shape, embeddings.dtype,
    )
    return embeddings


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single query string.

    Thin wrapper around embed_texts that returns a 1D vector of shape
    (EMBEDDING_DIM,) — suitable for direct FAISS search.
    """
    if not query or not query.strip():
        raise ValueError("Query text must not be empty.")

    vectors = embed_texts([query.strip()])
    return vectors[0]  # Shape: (EMBEDDING_DIM,)
