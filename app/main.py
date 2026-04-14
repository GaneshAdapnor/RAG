"""
FastAPI application entry point.

WHY FastAPI over Flask:
    1. Native async/await: FastAPI is built on Starlette (ASGI). Each request
       is handled by an async coroutine, so I/O-bound work (OpenAI API calls,
       file reads) doesn't block the event loop. Flask is WSGI — synchronous
       by default, requiring thread pools or gevent hacks for concurrency.
    2. Automatic OpenAPI docs: /docs (Swagger UI) and /redoc are generated from
       the code with zero configuration. Essential for API evaluation.
    3. Pydantic integration: Request/response validation is declared in the
       function signature. No manual request parsing.
    4. BackgroundTasks: Built into FastAPI's request lifecycle. A background
       task runs after the response is sent — no external queue needed for
       simple async work like document ingestion.
    5. Type safety: Python type hints are enforced by Pydantic at runtime,
       catching serialization bugs early.

LIFESPAN (startup/shutdown):
    FastAPI's @asynccontextmanager lifespan pattern replaces the older
    @app.on_event("startup") decorator (deprecated in FastAPI 0.103+).
    We use it to:
    - Load the FAISS index from disk before accepting traffic (avoiding the
      race condition where a query arrives before the index is ready)
    - Warm up the embedding model (avoids cold-start latency on first request)
    - Gracefully log shutdown
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.core.config import settings
from app.core.logging_config import setup_logging
from app.routes import health, query, upload
from app.services.embedding_service import get_embedding_model
from app.services.vector_store import get_vector_store
from app.utils.rate_limiter import limiter, rate_limit_exceeded_handler

# Initialize logging first — before any other imports log anything
setup_logging(level="INFO")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan: startup and shutdown hooks
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Manage application-level resources across the server lifecycle.

    Startup:
        1. Load the FAISS index from disk (or start fresh if not found).
        2. Pre-load the embedding model into memory.

    Shutdown:
        3. Log graceful shutdown (could flush metrics, close DB connections, etc.)

    Both steps run before the first request is served, ensuring the API is
    fully ready when it starts accepting traffic.
    """
    # --- Startup ---
    logger.info("Starting %s v%s", settings.PROJECT_NAME, settings.VERSION)

    # Load persisted vector store
    store = get_vector_store()
    store.load()
    logger.info(
        "Vector store ready: %d chunks across %d documents.",
        store.total_chunks,
        store.total_documents,
    )

    # Warm up the embedding model (downloads on first run if not cached)
    get_embedding_model()
    logger.info("Embedding model warm-up complete.")

    logger.info("Application startup complete. Ready to accept requests.")

    yield  # Server is running — handle requests

    # --- Shutdown ---
    logger.info("Shutting down %s. Goodbye.", settings.PROJECT_NAME)


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=(
        "Production-quality Retrieval-Augmented Generation (RAG) Question Answering API. "
        "Upload PDF/TXT documents and ask questions grounded in their content."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

# CORS — adjust allowed_origins for production deployments
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Tighten to specific domains in production
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Rate limiting middleware — must be added before routes are registered
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)


# ---------------------------------------------------------------------------
# Global exception handler — catch-all for unhandled exceptions
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Catch any unhandled exception and return a structured 500 response.

    Without this, FastAPI returns a plain HTML error page for unhandled
    exceptions. We want consistent JSON across all error types.
    """
    logger.error(
        "Unhandled exception on %s %s: %s",
        request.method,
        request.url.path,
        exc,
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An unexpected internal error occurred.",
            "error_code": "INTERNAL_SERVER_ERROR",
        },
    )


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------

app.include_router(health.router)
app.include_router(upload.router)
app.include_router(query.router)


# ---------------------------------------------------------------------------
# Root redirect
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root() -> JSONResponse:
    """Redirect root to API docs."""
    return JSONResponse(
        content={
            "message": f"Welcome to {settings.PROJECT_NAME}",
            "version": settings.VERSION,
            "docs": "/docs",
            "health": "/health",
        }
    )
