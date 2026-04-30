"""
Legal API — application entry point.

Registers all routers, middleware, and exception handlers.
Manages startup/shutdown lifecycle: Weaviate connect, Neo4j connect.

OCR is NOT initialized at startup.
The DoclingOCRClient uses lazy initialization — it activates
on the first document ingestion request, not at application boot.
This keeps startup time fast regardless of OCR model availability.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.middleware.correlation.correlation_middleware import CorrelationMiddleware
from src.api.middleware.error_handler.error_handler_middleware import app_exception_handler
from src.api.routes.chat.chat_routes import router as chat_router
from src.api.routes.conversation.conversation_routes import router as conversation_router
from src.api.routes.document.document_routes import router as document_router
from src.api.routes.health.health_routes import router as health_router
from src.api.routes.model.model_routes import router as model_router
from src.api.routes.prompt.prompt_routes import router as prompt_router
from src.core.config.settings import settings
from src.core.exceptions.base import AppBaseError
from src.core.logging.logger import get_logger
from src.db_models import register_all_models

# Ensure all ORM models are loaded and SQLAlchemy mappers configured
# before any database operation. This is explicit and deterministic
# unlike relying on import side effects from router imports.
register_all_models()

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan — startup and shutdown logic.

    Startup:
        1. Connect to Weaviate and ensure collection schema exists.
        2. Connect to Neo4j.
        (OCR is NOT initialized here — see DoclingOCRClient for lazy init.)

    Shutdown:
        1. Close Weaviate connection.
        2. Close Neo4j driver.
        3. Close Redis connection pool.
    """
    logger.info("legal API starting up...")

    from src.providers.infrastructure.clients import (
        get_cache_client,
        get_graph_client,
        get_vector_client,
    )

    # ── Weaviate ─────────────────────────────────────────────
    try:
        vector_client = get_vector_client()
        vector_client.connect()  # type: ignore[attr-defined]
        logger.info("Weaviate connected.")
    except Exception as exc:
        logger.error(
            "Weaviate connection failed at startup — vector search unavailable.",
            extra={"error": str(exc)},
            exc_info=True,
        )

    # ── Neo4j ─────────────────────────────────────────────────
    try:
        graph_client = get_graph_client()
        await graph_client.connect()  # type: ignore[attr-defined]
        logger.info("Neo4j connected.")
    except Exception as exc:
        logger.error(
            "Neo4j connection failed at startup — graph queries unavailable.",
            extra={"error": str(exc)},
            exc_info=True,
        )

    # OCR: no initialization here.
    # DoclingOCRClient activates lazily on first extract_text() call.
    logger.info(
        "OCR pipeline: lazy mode — will initialize on first document ingestion request."
    )

    logger.info("legal API startup complete.")
    yield

    # ── Shutdown ─────────────────────────────────────────────
    logger.info("legal API shutting down...")
    try:
        get_vector_client().close()  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        await get_graph_client().close()  # type: ignore[attr-defined]
    except Exception:
        pass
    try:
        await get_cache_client().close()  # type: ignore[attr-defined]
    except Exception:
        pass
    logger.info("legal API shutdown complete.")


def create_app() -> FastAPI:
    """Construct and configure the FastAPI application."""
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.APP_VERSION,
        description="Domain-Native Indonesian Legal GraphRAG Platform",
        docs_url="/docs" if settings.is_development else None,
        redoc_url="/redoc" if settings.is_development else None,
        lifespan=lifespan,
    )

    # ── Middleware (order matters: outermost added last) ──────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.is_development else [],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(CorrelationMiddleware)

    # ── Exception handlers ───────────────────────────────────
    app.add_exception_handler(AppBaseError, app_exception_handler)
    app.add_exception_handler(Exception, app_exception_handler)

    # ── Routers ──────────────────────────────────────────────
    app.include_router(health_router)
    app.include_router(model_router, prefix="/api/v1")
    app.include_router(prompt_router, prefix="/api/v1")
    app.include_router(conversation_router, prefix="/api/v1")
    app.include_router(document_router, prefix="/api/v1")
    app.include_router(chat_router, prefix="/api/v1")

    return app


app = create_app()