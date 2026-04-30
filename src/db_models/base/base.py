"""
SQLAlchemy declarative base for all ORM models.

All ORM classes inherit from Base. Alembic imports Base via env.py
to autogenerate migration scripts from ORM model changes.

Usage:
    from src.db_models.base.base import Base
"""

from __future__ import annotations

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """
    Shared declarative base for all ORM models.

    Using a single Base instance ensures all tables are registered
    in one MetaData object, which Alembic needs for autogenerate.
    """
    pass
