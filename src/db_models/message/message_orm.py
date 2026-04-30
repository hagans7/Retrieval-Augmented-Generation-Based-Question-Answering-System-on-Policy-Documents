"""
SQLAlchemy ORM model for the messages table.

Stores only user questions and assistant responses.
Intermediate agent steps are tracked in Langfuse, not here.
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from src.db_models.base.base import Base
from src.db_models.enums.db_enums import MessageRoleEnum


class MessageORM(Base):
    __tablename__ = "messages"

    __table_args__ = (
        sa.Index("ix_messages_conversation_id", "conversation_id"),
        sa.Index("ix_messages_created_at", "created_at"),
        sa.Index("ix_messages_system_prompt_id", "system_prompt_id"),
        sa.Index("ix_messages_model_id", "model_id"),
    )

    message_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True
    )
    conversation_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        sa.ForeignKey("conversations.conversation_id", ondelete="CASCADE"),
        nullable=False,
    )
    role: Mapped[str] = mapped_column(MessageRoleEnum, nullable=False)
    content: Mapped[str] = mapped_column(sa.Text, nullable=False)
    system_prompt_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False),
        sa.ForeignKey("system_prompts.system_prompt_id", ondelete="SET NULL"),
        nullable=True,
    )
    model_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False),
        sa.ForeignKey("available_models.model_id", ondelete="SET NULL"),
        nullable=True,
    )
    token_count: Mapped[int | None] = mapped_column(sa.Integer, nullable=True)
    created_at: Mapped[sa.DateTime] = mapped_column(
        sa.TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )

    conversation = relationship("ConversationORM", back_populates="messages", lazy="select")
