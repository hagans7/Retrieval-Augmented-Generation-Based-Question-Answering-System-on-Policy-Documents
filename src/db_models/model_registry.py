"""
ORM Model Registry — centralized, explicit loading of all SQLAlchemy models.

PURPOSE:
    SQLAlchemy uses a mapper registry to resolve string-based relationships
    (e.g. relationship("UserORM", ...)). For configure_mappers() to succeed,
    ALL models that participate in ANY relationship must be imported before
    any ORM operation is performed.

    In FastAPI this happens automatically because main.py imports routers
    which transitively import all models. In Celery workers, each task
    imports only what it needs — so the mapper registry is incomplete.

    This module is the single, explicit solution: import it once at the
    start of any process that uses SQLAlchemy (Celery tasks, scripts, tests).

USAGE:
    # At the top of celery_app.py, alembic env.py, test conftest.py:
    from src.db_models.model_registry import register_all_models
    register_all_models()

WHAT IT DOES:
    - Imports every ORM class into the SQLAlchemy mapper registry
    - Calls configure_mappers() explicitly to resolve all relationships
    - Is idempotent — safe to call multiple times (noop after first call)

NEVER:
    - Use lazy imports (inside functions) for ORM classes
    - Rely on import order side effects for mapper configuration
"""
from __future__ import annotations

import logging

_registered = False
logger = logging.getLogger(__name__)


def register_all_models() -> None:
    """
    Import every ORM model and configure SQLAlchemy mappers.

    Idempotent — subsequent calls are no-ops.
    Must be called before any session.execute(), session.add(), or
    relationship traversal in any process that uses SQLAlchemy.
    """
    global _registered
    if _registered:
        return

    logger.debug("ORM registry: loading all models...")

    # Base must be imported first to initialize the DeclarativeBase registry
    from src.db_models.base.base import Base  # noqa: F401

    # Import every ORM class so SQLAlchemy can register their mappers
    # and resolve all string-based relationships (e.g. relationship("UserORM"))
    from src.db_models.user.user_orm import UserORM  # noqa: F401
    from src.db_models.conversation.conversation_orm import ConversationORM  # noqa: F401
    from src.db_models.message.message_orm import MessageORM  # noqa: F401
    from src.db_models.document.document_orm import DocumentORM  # noqa: F401
    from src.db_models.system_prompt.system_prompt_orm import SystemPromptORM  # noqa: F401
    from src.db_models.available_model.available_model_orm import AvailableModelORM  # noqa: F401

    # Explicitly configure all mappers now that every model is loaded.
    # This resolves all string-based relationship() references.
    # Without this, SQLAlchemy resolves lazily on first use — which
    # can deadlock in async/Celery contexts.
    from sqlalchemy.orm import configure_mappers
    configure_mappers()

    _registered = True
    logger.debug(
        "ORM registry: all models loaded and mappers configured. "
        f"Tables: {list(Base.metadata.tables.keys())}"
    )