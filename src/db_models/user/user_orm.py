"""
SQLAlchemy ORM model for the users table.

Prepared for future auth enforcement. Currently not enforced in middleware.
The static DEFAULT_USER_ID row is seeded during initial migration.
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from src.db_models.base.base import Base
from src.db_models.enums.db_enums import UserRoleEnum


class UserORM(Base):
    __tablename__ = "users"

    user_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True
    )
    role: Mapped[str] = mapped_column(
        UserRoleEnum, nullable=False, default="user", server_default="user"
    )
    is_active: Mapped[bool] = mapped_column(
        sa.Boolean, nullable=False, default=True, server_default=sa.true()
    )
    created_at: Mapped[sa.DateTime] = mapped_column(
        sa.TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )

    # Relationships — lazy loaded, not eager
    conversations = relationship("ConversationORM", back_populates="user", lazy="select")
    system_prompts = relationship("SystemPromptORM", back_populates="user", lazy="select")
    documents = relationship("DocumentORM", back_populates="user", lazy="select")
