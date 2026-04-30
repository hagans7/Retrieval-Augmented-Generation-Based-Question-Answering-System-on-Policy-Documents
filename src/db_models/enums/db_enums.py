"""
SQLAlchemy Enum type definitions for all database columns.

All enum types are defined here — never inline within ORM model files.
This ensures consistency: if an enum value changes, only this file changes.

Alembic will create the PostgreSQL ENUM types from these definitions.
"""

from __future__ import annotations

import sqlalchemy as sa

# Stored in messages.role column
MessageRoleEnum = sa.Enum(
    "user",
    "assistant",
    name="message_role",
    create_type=True,
)

# Stored in documents.ingestion_status column
IngestionStatusEnum = sa.Enum(
    "pending",
    "processing",
    "completed",
    "failed",
    name="ingestion_status",
    create_type=True,
)

# Stored in users.role column — prepared for future auth enforcement
UserRoleEnum = sa.Enum(
    "user",
    "admin",
    "supervisor",
    name="user_role",
    create_type=True,
)
