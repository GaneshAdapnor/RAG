"""
Retrieval service: converts a user query into ranked document chunks.

END-TO-END RETRIEVAL FLOW:
    1. Embed the query using the same model used for indexing.
       (Critical: using a different model for queries vs. indexing produces
        semantically incompatible vectors — a common production bug.)
    2. Search FAISS for the top-K nearest neighbours.
    3. Apply optional doc_id filtering (post-search).
    4. Build a de-duplicated, ordered context string for the LLM prompt.

SIMILARITY SCORE INTERPRETATION:
    Since all vectors are L2-normalized and we use IndexFlatIP, the scores
    are cosine similarities in [0, 1]:
        0.9–1.0 : Near-duplicate or very high semantic match
        0.7–0.9 : Strong semantic match (most useful retrievals fall here)
        0.5–0.7 : Moderate match (may be tangentially related)
        < 0.5   : Weak match (consider filtering these out)

    We apply a configurable SIMILARITY_THRESHOLD to drop low-quality results
    rather than feeding noisy context to the LLM.

RETRIEVAL FAILURE CASE (realistic example):
    Query:  "What was the net profit margin in Q3?"
    Issue:  The document has "net income as a percentage of revenue was 18.3%
            in the third quarter" — no exact phrase match for "net profit margin"
            or "Q3". The embedding of this query is semantically close but the
            chunk boundary splits the sentence:
                Chunk N:   "...revenue was 18.3% in the"
                Chunk N+1: "third quarter, compared to 21.1% in Q2..."
            Neither chunk alone has the full fact. The retrieved chunk has a
            cosine score of ~0.62, which is above threshold but the extracted
            text doesn't contain the number + quarter together.

    Fix strategies (in order of implementation effort):
        1. Increase chunk_overlap (100 chars) so the boundary sentence appears
           complete in at least one chunk.
        2. Hybrid search: combine dense (FAISS) + sparse (BM25/TF-IDF) scores.
           BM25 would boost "Q3" and "net profit" by exact keyword match.
        3. Re-ranking: after FAISS retrieval, use a cross-encoder model
           (e.g., cross-encoder/ms-marco-MiniLM-L-6-v2) to re-score chunks.
           Cross-encoders attend to both query and passage jointly, so they
           catch this semantic alignment that bi-encoders miss.
        4. Metadata filtering: store financial period tags during ingestion
           and filter by period at query time.
"""

import logging
import time
from typing import List, Optional, Tuple

from app.core.config import settings
from app.models.schemas import RetrievedChunk
from app.services.embedding_service import embed_query
from app.services.vector_store import ChunkMetadata, get_vector_store

logger = logging.getLogger(__name__)

# Minimum cosine similarity to accept a retrieved chunk.
# Chunks below this threshold are likely semantically unrelated.
SIMILARITY_THRESHOLD = 0.30


def retrieve_chunks(
    query: str,
    top_k: Optional[int] = None,
    doc_ids: Optional[List[str]] = None,
) -> Tuple[List[RetrievedChunk], float]:
    """
    Embed the query and retrieve the top-K most relevant chunks.

    This is the core retrieval function. It measures and returns its own
    latency so the API response can expose it for monitoring.

    Args:
        query: Raw natural language query string.
        top_k: Override for the default TOP_K_RETRIEVAL setting.
        doc_ids: If provided, restrict search to these document IDs.

    Returns:
        Tuple of:
          - List of RetrievedChunk, sorted by descending similarity score.
          - Retrieval latency in milliseconds (embed + FAISS search combined).

    Raises:
        RuntimeError: If the vector store is empty (no documents indexed yet).
    """
    k = top_k if top_k is not None else settings.TOP_K_RETRIEVAL

    store = get_vector_store()
    if store.total_chunks == 0:
        raise RuntimeError(
            "The vector store is empty. Please upload and process documents before querying."
        )

    retrieval_start = time.perf_counter()

    # Step 1: Embed the query
    query_vector = embed_query(query)

    # Step 2: FAISS similarity search
    raw_results = store.search(query_vector, top_k=k, doc_ids=doc_ids)

    retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000

    # Step 3: Filter by similarity threshold and convert to schema objects
    chunks: List[RetrievedChunk] = []
    for meta, score in raw_results:
        if score < SIMILARITY_THRESHOLD:
            logger.debug(
                "Dropping chunk (doc_id=%s, chunk_id=%d): score %.4f < threshold %.2f.",
                meta.doc_id, meta.chunk_id, score, SIMILARITY_THRESHOLD,
            )
            continue

        chunks.append(
            RetrievedChunk(
                doc_id=meta.doc_id,
                filename=meta.filename,
                chunk_id=meta.chunk_id,
                page=meta.page,
                text=meta.text,
                score=round(score, 4),
            )
        )

    logger.info(
        "Retrieval: query='%s...', k=%d, returned=%d chunks, latency=%.1fms.",
        query[:60], k, len(chunks), retrieval_latency_ms,
    )

    return chunks, retrieval_latency_ms


def build_context(chunks: List[RetrievedChunk]) -> str:
    """
    Assemble retrieved chunks into a formatted context string for the LLM.

    Format design:
        Each chunk is wrapped with a source header so the LLM can reference
        specific documents if asked. Chunks are separated by a visual divider
        so the LLM understands they are discrete passages, not flowing prose.

    Args:
        chunks: Retrieved and filtered chunks, ordered by relevance.

    Returns:
        Formatted context string ready for insertion into the LLM prompt.
        Returns an empty string if chunks is empty.
    """
    if not chunks:
        return ""

    parts = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.filename
        location = f"page {chunk.page}" if chunk.page is not None else "text file"
        header = f"[Source {i}: {source}, {location}, similarity={chunk.score:.3f}]"
        parts.append(f"{header}\n{chunk.text}")

    return "\n\n---\n\n".join(parts)
