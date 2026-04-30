"""
Health check routes.

/health/liveness  — is the FastAPI process alive? (no dependency checks)
/health/readiness — are critical dependencies ready?

OCR readiness note:
    The OCR client uses lazy initialization — it is NOT ready at startup.
    An OCR status of false simply means no document has been processed yet;
    it does NOT mean the system is unhealthy or unable to serve chat requests.
    This is expected behavior and is documented in the readiness response.
"""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/liveness")
async def liveness():
    """Return 200 if the FastAPI process is running. No dependency checks."""
    return {"status": "ok"}


@router.get("/readiness")
async def readiness():
    """
    Check critical dependencies and return 200 only if all pass.

    Returns 503 with per-check details if any critical check fails.
    OCR is non-critical at readiness time (lazy init) and reported
    separately with an explanatory note.
    """
    from src.providers.infrastructure.database import check_db_connectivity
    from src.providers.infrastructure.clients import get_cache_client, get_ocr_client

    checks: dict[str, bool] = {}
    notes: dict[str, str] = {}

    # PostgreSQL — critical
    checks["database"] = await check_db_connectivity()

    # Redis — critical (Celery broker)
    try:
        cache = get_cache_client()
        checks["redis"] = await cache.ping()  # type: ignore[attr-defined]
    except Exception:
        checks["redis"] = False

    # OCR — non-critical at readiness time (lazy init)
    try:
        ocr = get_ocr_client()
        checks["ocr_pipeline"] = ocr.is_ready()
        if not checks["ocr_pipeline"]:
            notes["ocr_pipeline"] = (
                "Not yet activated. OCR initializes lazily on the first "
                "document ingestion request. Chat endpoints are fully available."
            )
    except Exception:
        checks["ocr_pipeline"] = False

    # Only DB and Redis are critical for overall readiness
    critical_ok = checks["database"] and checks["redis"]
    status_code = 200 if critical_ok else 503

    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ready" if critical_ok else "not_ready",
            "checks": checks,
            "notes": notes,
        },
    )