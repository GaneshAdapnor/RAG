"""
Query route: the core RAG endpoint.

POST /query:
    Takes a natural language question, retrieves relevant document chunks,
    builds a grounded prompt, and returns an LLM-generated answer with
    full source attribution.

METRIC TRACKING — Latency (why chosen):
    We measure and return two separate latency metrics:
    1. retrieval_latency_ms: Time for query embedding + FAISS search.
       This isolates the vector search performance, which should be < 50ms
       for indexes < 500K vectors. If this spikes, it signals model reload
       or FAISS degradation.
    2. generation_latency_ms: Time for the OpenAI API call.
       This is network + inference time. Typical range: 500ms–3,000ms.
       Spikes here indicate OpenAI rate limits or network issues.

    WHY LATENCY over retrieval accuracy or similarity score:
    - Latency is immediately observable (no ground truth needed).
    - It directly impacts user experience — a RAG system must feel responsive.
    - Separating retrieval vs. generation latency makes debugging fast:
      if total latency is high, you can immediately tell whether it's the
      embedding model, FAISS, or the LLM.
    - Similarity scores are logged per-query for offline analysis, but are not
      the right primary operational metric (they require calibration to be
      meaningful).

    How to monitor in production:
        Log structured JSON with query, retrieval_latency_ms, generation_latency_ms,
        chunk_count, and top similarity score. Feed into Datadog/Prometheus with
        p50/p95/p99 percentile alerts on total latency > 5,000ms.
"""

import logging

from fastapi import APIRouter, HTTPException, Request

from app.models.schemas import QueryRequest, QueryResponse
from app.services.llm_service import generate_answer
from app.services.retrieval_service import build_context, retrieve_chunks
from app.utils.rate_limiter import QUERY_RATE_LIMIT, limiter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/query", tags=["Query"])


@router.post(
    "",
    response_model=QueryResponse,
    summary="Ask a question about indexed documents",
    description=(
        "Submit a natural language question. The system retrieves the most "
        "relevant document chunks and generates a grounded answer. "
        "Answers are based ONLY on indexed documents."
    ),
)
@limiter.limit(QUERY_RATE_LIMIT)
async def query_documents(
    request: Request,
    payload: QueryRequest,
) -> QueryResponse:
    """
    Full RAG pipeline: embed → retrieve → build context → generate → respond.

    Error cases handled:
    - Empty vector store: 503 Service Unavailable (no documents indexed)
    - LLM API failure: 502 Bad Gateway
    - Unexpected errors: 500 Internal Server Error
    """
    logger.info(
        "Query received: '%s...' (top_k=%s, doc_ids=%s).",
        payload.query[:60],
        payload.top_k,
        payload.doc_ids,
    )

    # ------------------------------------------------------------------
    # Step 1: Retrieve relevant chunks
    # ------------------------------------------------------------------
    try:
        chunks, retrieval_latency_ms = retrieve_chunks(
            query=payload.query,
            top_k=payload.top_k,
            doc_ids=payload.doc_ids,
        )
    except RuntimeError as exc:
        # Raised when the vector store is empty
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error("Retrieval failed for query '%s...': %s", payload.query[:60], exc)
        raise HTTPException(
            status_code=500,
            detail="An error occurred during document retrieval.",
        )

    # ------------------------------------------------------------------
    # Step 2: Build context string
    # ------------------------------------------------------------------
    context = build_context(chunks)

    # ------------------------------------------------------------------
    # Step 3: Generate answer
    # ------------------------------------------------------------------
    try:
        answer, generation_latency_ms = generate_answer(
            query=payload.query,
            context=context,
            chunks=chunks,
        )
    except RuntimeError as exc:
        # LLM API errors (rate limit, network, auth)
        raise HTTPException(
            status_code=502,
            detail=str(exc),
        )
    except Exception as exc:
        logger.error("LLM generation failed for query '%s...': %s", payload.query[:60], exc)
        raise HTTPException(
            status_code=500,
            detail="An error occurred during answer generation.",
        )

    # ------------------------------------------------------------------
    # Step 4: Structured response with latency metrics
    # ------------------------------------------------------------------
    total_ms = retrieval_latency_ms + generation_latency_ms
    logger.info(
        "Query completed: retrieved=%d chunks, retrieval=%.1fms, generation=%.1fms, total=%.1fms.",
        len(chunks), retrieval_latency_ms, generation_latency_ms, total_ms,
    )

    return QueryResponse(
        query=payload.query,
        answer=answer,
        sources=chunks,
        retrieval_latency_ms=round(retrieval_latency_ms, 2),
        generation_latency_ms=round(generation_latency_ms, 2),
    )
