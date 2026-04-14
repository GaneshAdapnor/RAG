from __future__ import annotations

from fastapi import FastAPI

from app.core.config import get_settings
from app.core.dependencies import get_document_registry, get_vector_store
from app.routes.health import router as health_router
from app.routes.query import router as query_router
from app.routes.upload import router as upload_router
from app.utils.logging import configure_logging


def create_application() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title=settings.project_name,
        version=settings.version,
        description=(
            "Production-style Retrieval-Augmented Generation API with explicit "
            "document ingestion, FAISS-based retrieval, and grounded answer generation."
        ),
    )

    @app.on_event("startup")
    def startup() -> None:
        settings.ensure_directories()
        get_document_registry()
        get_vector_store()

    app.include_router(health_router, prefix=settings.api_prefix)
    app.include_router(upload_router, prefix=settings.api_prefix)
    app.include_router(query_router, prefix=settings.api_prefix)

    return app


app = create_application()
