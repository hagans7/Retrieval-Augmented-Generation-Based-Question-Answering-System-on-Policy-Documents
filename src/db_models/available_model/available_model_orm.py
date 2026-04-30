"""
SQLAlchemy ORM model for the available_models table.

Stores the catalog of LLM models users can select from.
Written via seeding/admin only — never written by application requests.
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func

from src.db_models.base.base import Base


class AvailableModelORM(Base):
    __tablename__ = "available_models"

    __table_args__ = (
        sa.Index("ix_available_models_is_active", "is_active"),
        sa.Index("ix_available_models_provider", "provider"),
    )

    model_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True
    )
    model_name: Mapped[str] = mapped_column(
        sa.Text, nullable=False, unique=True
    )
    provider: Mapped[str] = mapped_column(sa.Text, nullable=False)
    display_name: Mapped[str] = mapped_column(sa.Text, nullable=False)
    description: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    context_window: Mapped[int | None] = mapped_column(sa.Integer, nullable=True)
    is_active: Mapped[bool] = mapped_column(
        sa.Boolean, nullable=False, default=True, server_default=sa.true()
    )
    created_at: Mapped[sa.DateTime] = mapped_column(
        sa.TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )
