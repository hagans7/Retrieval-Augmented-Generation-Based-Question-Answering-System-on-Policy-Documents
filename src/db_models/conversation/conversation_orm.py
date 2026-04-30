"""
SQLAlchemy ORM model for the conversations table.
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from src.db_models.base.base import Base


class ConversationORM(Base):
    __tablename__ = "conversations"

    __table_args__ = (
        sa.Index("ix_conversations_user_id", "user_id"),
        sa.Index("ix_conversations_created_at", "created_at"),
        sa.Index("ix_conversations_is_active", "is_active"),
    )

    conversation_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True
    )
    user_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False),
        sa.ForeignKey("users.user_id", ondelete="SET NULL"),
        nullable=True,
    )
    title: Mapped[str] = mapped_column(
        sa.Text, nullable=False, default="New Conversation"
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

    user = relationship("UserORM", back_populates="conversations", lazy="select")
    messages = relationship(
        "MessageORM",
        back_populates="conversation",
        lazy="select",
        order_by="MessageORM.created_at",
    )
