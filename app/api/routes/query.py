from __future__ import annotations

from fastapi import APIRouter, Depends
from starlette.concurrency import run_in_threadpool

from app.core.dependencies import enforce_rate_limit, get_query_service
from app.models.api import QueryRequest, QueryResponse
from app.services.query_service import QueryService

router = APIRouter(tags=["query"])


@router.post(
    "/query",
    response_model=QueryResponse,
    dependencies=[Depends(enforce_rate_limit)],
)
async def query_documents(
    request: QueryRequest,
    query_service: QueryService = Depends(get_query_service),
) -> QueryResponse:
    return await run_in_threadpool(query_service.answer, request)
