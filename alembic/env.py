"""
Alembic migration environment configuration.

Uses synchronous psycopg2 connection for migration execution.
Imports all ORM models so Alembic can detect schema changes for autogenerate.
"""

from __future__ import annotations

import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool
from alembic import context

# Use centralized model registry — ensures all ORM classes are loaded
# and configure_mappers() is called before Alembic inspects Base.metadata
from src.db_models import register_all_models
from src.db_models.base.base import Base
register_all_models()

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def get_sync_url() -> str:
    """
    Convert async DATABASE_URL to sync URL for Alembic.

    Alembic's migration runner uses synchronous psycopg2.
    Replace +asyncpg with +psycopg2 for compatibility.
    """
    url = os.environ.get("DATABASE_URL", "")
    return url.replace("+asyncpg", "+psycopg2")


def run_migrations_offline() -> None:
    """Run migrations in offline mode (generate SQL without DB connection)."""
    url = get_sync_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations with a live DB connection."""
    configuration = config.get_section(config.config_ini_section) or {}
    configuration["sqlalchemy.url"] = get_sync_url()

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()