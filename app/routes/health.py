"""
Health check endpoint.

GET /health is a standard operational endpoint used by:
- Kubernetes liveness/readiness probes
- Load balancers to route traffic
- Monitoring dashboards to verify the service is up

We return system stats (indexed chunks, documents) so operators can confirm
the vector store loaded correctly at startup without querying the full API.
"""

import logging

from fastapi import APIRouter

from app.core.config import settings
from app.models.schemas import HealthResponse
from app.services.vector_store import get_vector_store

logger = logging.getLogger(__name__)

router = APIRouter(tags=["System"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description=(
        "Returns service status and index statistics. "
        "Does NOT require authentication. Not rate-limited."
    ),
)
async def health_check() -> HealthResponse:
    """
    Verify the service is running and report vector store state.

    This endpoint always returns 200 OK as long as the process is alive.
    If critical components are broken (e.g., FAISS index corrupt), the
    error will appear in the logs and `indexed_chunks` will be 0.
    """
    store = get_vector_store()

    return HealthResponse(
        status="ok",
        version=settings.VERSION,
        embedding_model=settings.EMBEDDING_MODEL_NAME,
        indexed_chunks=store.total_chunks,
        indexed_documents=store.total_documents,
    )
