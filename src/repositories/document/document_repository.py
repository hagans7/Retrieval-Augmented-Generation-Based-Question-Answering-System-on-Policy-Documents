"""
Document repository.

No begin() — uses implicit autobegin transaction from SQLAlchemy 2.0.

Soft delete: documents table has no is_active column.
Deletion is represented by ingestion_status='failed' with error_message='Deleted by user.'
This keeps the document record visible for audit but excluded from active document lists.
"""
from __future__ import annotations
import uuid
from sqlalchemy import select, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.exceptions.domain import PersistenceError
from src.core.exceptions.not_found import DocumentNotFoundError
from src.core.logging.logger import get_logger
from src.db_models.document.document_orm import DocumentORM
from src.entities.document.document import Document
from src.interfaces.repositories.base_document_repository import BaseDocumentRepository
from src.repositories.document.document_mapper import to_entity

_DELETED_STATUSES = {"failed"}


class DocumentRepository(BaseDocumentRepository):
    def __init__(self, db_session: AsyncSession) -> None:
        self._session = db_session
        self._logger = get_logger(__name__)

    async def create(self, user_id, file_name, file_type, storage_key, file_size_bytes) -> Document:
        orm = DocumentORM(
            document_id=str(uuid.uuid4()),
            user_id=user_id,
            file_name=file_name,
            file_type=file_type,
            storage_key=storage_key,
            file_size_bytes=file_size_bytes,
            ingestion_status="pending",
        )
        try:
            self._session.add(orm)
            await self._session.commit()
            await self._session.refresh(orm)
            return to_entity(orm)
        except SQLAlchemyError as exc:
            await self._session.rollback()
            raise PersistenceError(
                message="Failed to create document record.",
                context={"error": str(exc)},
            ) from exc

    async def get_by_id(self, document_id: str) -> Document | None:
        try:
            stmt = select(DocumentORM).where(DocumentORM.document_id == document_id)
            result = await self._session.execute(stmt)
            orm = result.scalar_one_or_none()
            return to_entity(orm) if orm else None
        except SQLAlchemyError as exc:
            raise PersistenceError(
                message=f"Failed to retrieve document '{document_id}'.",
                context={"document_id": document_id, "error": str(exc)},
            ) from exc

    async def update_status(
        self, document_id, status, error_message=None, chunk_count=None
    ) -> Document:
        values: dict = {"ingestion_status": status}
        if error_message is not None:
            values["error_message"] = error_message
        if chunk_count is not None:
            values["chunk_count"] = chunk_count
        try:
            stmt = (
                update(DocumentORM)
                .where(DocumentORM.document_id == document_id)
                .values(**values)
                .returning(DocumentORM)
            )
            result = await self._session.execute(stmt)
            orm = result.scalar_one_or_none()
            if not orm:
                raise DocumentNotFoundError(document_id)
            await self._session.commit()
            return to_entity(orm)
        except DocumentNotFoundError:
            raise
        except SQLAlchemyError as exc:
            await self._session.rollback()
            raise PersistenceError(
                message=f"Failed to update status for document '{document_id}'.",
                context={"document_id": document_id, "error": str(exc)},
            ) from exc

    async def get_all_by_user(self, user_id, limit, offset) -> list[Document]:
        """
        Return all non-deleted documents for a user.
        Excludes documents with status='failed' that have error_message='Deleted by user.'
        """
        try:
            stmt = (
                select(DocumentORM)
                .where(
                    DocumentORM.user_id == user_id,
                    # Exclude user-deleted records
                    ~(
                        (DocumentORM.ingestion_status == "failed")
                        & (DocumentORM.error_message == "Deleted by user.")
                    ),
                )
                .order_by(DocumentORM.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            result = await self._session.execute(stmt)
            return [to_entity(r) for r in result.scalars().all()]
        except SQLAlchemyError as exc:
            raise PersistenceError(
                message="Failed to list documents.",
                context={"user_id": user_id, "error": str(exc)},
            ) from exc

    async def soft_delete(self, document_id: str) -> None:
        """
        Soft-delete a document by marking it as failed with 'Deleted by user.' message.

        This preserves the audit trail while hiding it from active document lists.
        """
        try:
            stmt = (
                update(DocumentORM)
                .where(DocumentORM.document_id == document_id)
                .values(ingestion_status="failed", error_message="Deleted by user.")
            )
            result = await self._session.execute(stmt)
            if result.rowcount == 0:
                raise DocumentNotFoundError(document_id)
            await self._session.commit()
        except DocumentNotFoundError:
            raise
        except SQLAlchemyError as exc:
            await self._session.rollback()
            raise PersistenceError(
                message=f"Failed to delete document '{document_id}'.",
                context={"document_id": document_id, "error": str(exc)},
            ) from exc