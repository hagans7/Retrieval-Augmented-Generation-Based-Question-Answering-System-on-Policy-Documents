"""
ORM models package.

Import register_all_models() from here to ensure all SQLAlchemy
mappers are configured before any database operation.

Usage (Celery, scripts, tests):
    from src.db_models import register_all_models
    register_all_models()
"""
from src.db_models.model_registry import register_all_models

__all__ = ["register_all_models"]