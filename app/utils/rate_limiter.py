"""
Rate limiting using SlowAPI (a thin wrapper over the `limits` library for FastAPI).

WHY SlowAPI:
    - Flask-Limiter's FastAPI port. Mature, well-tested. Adds rate limiting to
      FastAPI with minimal boilerplate — no middleware rewrite required.
    - Supports multiple storage backends (in-memory, Redis). We use in-memory
      (MemoryStorage) because it's zero-dependency for single-node deployments.
      For multi-replica: swap MemoryStorage for RedisStorage with one line.
    - Integrates with FastAPI's dependency injection for per-route control.

RATE LIMIT STRATEGY:
    We apply a per-IP token bucket: 10 requests per 60 seconds (configured in
    settings). This prevents:
    - Accidental hammering from a misbehaving client
    - Denial-of-wallet attacks on the OpenAI API endpoint

    The /upload endpoint is limited separately at 5/minute because ingestion
    is CPU-intensive (embedding generation) and should not be spammed.

    The /health endpoint is NOT rate-limited — it's a lightweight probe used
    by load balancers and should always be reachable.

TOKEN BUCKET vs FIXED WINDOW:
    SlowAPI uses a sliding window (not fixed window) by default. This prevents
    the "thundering herd" problem at window boundaries where clients can send
    2× the limit by timing requests to straddle the reset point.

IDENTIFYING CLIENTS:
    We use X-Forwarded-For header (first IP in the chain) for clients behind a
    proxy/load balancer, falling back to request.client.host. This matches the
    real client IP rather than the proxy IP.

    Security note: In production, validate that X-Forwarded-For is set by a
    trusted proxy and cannot be spoofed by clients.
"""

import logging
from typing import Callable

from fastapi import Request, Response
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.config import settings

logger = logging.getLogger(__name__)


def get_client_ip(request: Request) -> str:
    """
    Extract the real client IP, respecting X-Forwarded-For.

    Priority:
        1. X-Forwarded-For header (first IP — the original client)
        2. X-Real-IP header (set by some proxies)
        3. request.client.host (direct connection)
    """
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # X-Forwarded-For: client, proxy1, proxy2 — take the leftmost
        client_ip = forwarded_for.split(",")[0].strip()
        return client_ip

    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()

    return get_remote_address(request)


# Module-level limiter instance — registered with the FastAPI app in main.py
limiter = Limiter(
    key_func=get_client_ip,
    default_limits=[],  # No default limit — applied per-route
    storage_uri="memory://",  # In-process storage; swap to "redis://..." for multi-replica
)

# Convenience limit strings derived from settings
# Format: "{count}/{period}" — e.g. "10/minute"
QUERY_RATE_LIMIT = f"{settings.RATE_LIMIT_CALLS}/minute"
UPLOAD_RATE_LIMIT = "5/minute"   # More conservative — embedding is CPU-intensive


def rate_limit_exceeded_handler(request: Request, exc: Exception) -> Response:
    """
    Custom handler for 429 Too Many Requests.

    Returns a JSON body consistent with the API's ErrorResponse schema rather
    than the default HTML error page that SlowAPI generates.
    """
    import json
    from starlette.responses import JSONResponse

    logger.warning(
        "Rate limit exceeded for client %s on %s %s.",
        get_client_ip(request),
        request.method,
        request.url.path,
    )

    return JSONResponse(
        status_code=429,
        content={
            "detail": "Rate limit exceeded. Please slow down your requests.",
            "error_code": "RATE_LIMIT_EXCEEDED",
        },
        headers={"Retry-After": "60"},
    )
