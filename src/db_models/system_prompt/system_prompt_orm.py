"""
SQLAlchemy ORM model for the system_prompts table.

Stores user-defined system prompts (persona, domain context injections).
user_id is nullable while auth is deferred.
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from src.db_models.base.base import Base


class SystemPromptORM(Base):
    __tablename__ = "system_prompts"

    __table_args__ = (
        sa.Index("ix_system_prompts_user_id", "user_id"),
        sa.Index("ix_system_prompts_is_default", "is_default"),
        sa.Index("ix_system_prompts_is_active", "is_active"),
    )

    system_prompt_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True
    )
    user_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False),
        sa.ForeignKey("users.user_id", ondelete="SET NULL"),
        nullable=True,
    )
    name: Mapped[str] = mapped_column(sa.Text, nullable=False)
    content: Mapped[str] = mapped_column(sa.Text, nullable=False)
    is_default: Mapped[bool] = mapped_column(
        sa.Boolean, nullable=False, default=False, server_default=sa.false()
    )
    is_active: Mapped[bool] = mapped_column(
        sa.Boolean, nullable=False, default=True, server_default=sa.true()
    )
    created_at: Mapped[sa.DateTime] = mapped_column(
        sa.TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[sa.DateTime] = mapped_column(
        sa.TIMESTAMP(timezone=True), nullable=False,
        server_default=func.now(), onupdate=func.now()
    )

    user = relationship("UserORM", back_populates="system_prompts", lazy="select")
