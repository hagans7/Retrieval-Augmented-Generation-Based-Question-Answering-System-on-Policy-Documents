"""
Global exception handler middleware.

Maps all AppBaseError subclasses to appropriate HTTP status codes
and wraps them in the standard ErrorResponse envelope.
Unexpected exceptions are logged with full stack trace and return 500.
"""
from __future__ import annotations
import traceback
from fastapi import Request
from fastapi.responses import JSONResponse
from src.core.exceptions.base import AppBaseError
from src.core.exceptions.not_found import NotFoundError
from src.core.exceptions.domain import (
    ModelNotActiveError, ValidationError, PersistenceError, AgentExecutionError,
)
from src.core.exceptions.infrastructure import (
    LLMInferenceError, VectorSearchError, GraphQueryError,
    EmbeddingError, RerankerError, StorageError, OCRError,
)
from src.core.logging.logger import get_logger

logger = get_logger(__name__)

_STATUS_MAP: dict[type, int] = {
    NotFoundError: 404,
    ModelNotActiveError: 422,
    ValidationError: 422,
    PersistenceError: 503,
    AgentExecutionError: 503,
    LLMInferenceError: 503,
    VectorSearchError: 503,
    GraphQueryError: 503,
    EmbeddingError: 503,
    RerankerError: 503,
    StorageError: 503,
    OCRError: 503,
}


async def app_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Map known AppBaseError to HTTP response. Unknown exceptions → 500."""
    correlation_id = getattr(request.state, "correlation_id", "-")

    if isinstance(exc, AppBaseError):
        status_code = 500
        for exc_type, code in _STATUS_MAP.items():
            if isinstance(exc, exc_type):
                status_code = code
                break
        logger.warning(
            "Application error",
            extra={"error_type": type(exc).__name__, "error": exc.message, "correlation_id": correlation_id},
            exc_info=True
        )
        return JSONResponse(
            status_code=status_code,
            content={
                "success": False,
                "data": None,
                "error": {
                    "code": type(exc).__name__.upper().replace("ERROR", "_ERROR"),
                    "message": exc.message,
                },
            },
        )

    # Unhandled exception
    logger.error(
        "Unhandled exception",
        extra={"correlation_id": correlation_id, "error": str(exc)},
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "data": None,
            "error": {"code": "INTERNAL_SERVER_ERROR", "message": "An unexpected error occurred."},
        },
    )
