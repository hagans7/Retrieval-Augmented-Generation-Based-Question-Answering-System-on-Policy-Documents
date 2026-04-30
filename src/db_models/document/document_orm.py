"""
SQLAlchemy ORM model for the documents table.

Stores metadata for all uploaded documents.
The physical file lives in object storage (MinIO/S3) at storage_key.
Chunks live in Weaviate. Entities live in Neo4j.
"""

from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from src.db_models.base.base import Base
from src.db_models.enums.db_enums import IngestionStatusEnum


class DocumentORM(Base):
    __tablename__ = "documents"

    __table_args__ = (
        sa.Index("ix_documents_user_id", "user_id"),
        sa.Index("ix_documents_ingestion_status", "ingestion_status"),
        sa.Index("ix_documents_created_at", "created_at"),
    )

    document_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False), primary_key=True
    )
    user_id: Mapped[str | None] = mapped_column(
        UUID(as_uuid=False),
        sa.ForeignKey("users.user_id", ondelete="SET NULL"),
        nullable=True,
    )
    file_name: Mapped[str] = mapped_column(sa.Text, nullable=False)
    file_type: Mapped[str] = mapped_column(sa.Text, nullable=False)
    storage_key: Mapped[str] = mapped_column(sa.Text, nullable=False, unique=True)
    file_size_bytes: Mapped[int | None] = mapped_column(sa.BigInteger, nullable=True)
    ingestion_status: Mapped[str] = mapped_column(
        IngestionStatusEnum, nullable=False, default="pending", server_default="pending"
    )
    error_message: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    chunk_count: Mapped[int | None] = mapped_column(sa.Integer, nullable=True)
    created_at: Mapped[sa.DateTime] = mapped_column(
        sa.TIMESTAMP(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[sa.DateTime] = mapped_column(
        sa.TIMESTAMP(timezone=True), nullable=False,
        server_default=func.now(), onupdate=func.now()
    )

    user = relationship("UserORM", back_populates="documents", lazy="select")
