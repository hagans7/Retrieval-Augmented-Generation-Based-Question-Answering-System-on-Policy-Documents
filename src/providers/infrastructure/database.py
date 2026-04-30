"""
Database infrastructure provider.

Creates the async SQLAlchemy engine and session factory once at startup.
The get_db_session() generator is used as a FastAPI Depends() dependency.
"""
from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.core.config.settings import settings
from src.core.logging.logger import get_logger

logger = get_logger(__name__)

# Engine created once at module import — shared across all requests.
# pool_pre_ping=True reconnects on stale connections after DB restart.
_engine = create_async_engine(
    settings.DATABASE_URL,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_pre_ping=settings.DB_POOL_PRE_PING,
    echo=settings.DEBUG,
)

_AsyncSessionFactory = async_sessionmaker(
    bind=_engine,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI Depends() provider for an async database session.

    Yields one AsyncSession per request and guarantees it is closed
    when the request completes, whether it succeeds or raises.

    Usage:
        async def route(db: AsyncSession = Depends(get_db_session)):
            ...
    """
    async with _AsyncSessionFactory() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def check_db_connectivity() -> bool:
    """
    Ping the database to confirm connectivity.

    Used by the readiness health check endpoint.

    Returns:
        True if DB responds; False otherwise.
    """
    try:
        from sqlalchemy import text
        async with _AsyncSessionFactory() as session:
            await session.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        logger.warning("DB connectivity check failed", extra={"error": str(exc)})
        return False
