"""Abstract base for document repositories."""
from __future__ import annotations
from abc import ABC, abstractmethod
from src.entities.document.document import Document


class BaseDocumentRepository(ABC):
    """Contract for document metadata persistence."""

    @abstractmethod
    async def create(
        self,
        user_id: str | None,
        file_name: str,
        file_type: str,
        storage_key: str,
        file_size_bytes: int | None,
    ) -> Document:
        """Create a new document record with status=pending."""
        ...

    @abstractmethod
    async def get_by_id(self, document_id: str) -> Document | None:
        """Return document by UUID, or None if not found."""
        ...

    @abstractmethod
    async def update_status(
        self,
        document_id: str,
        status: str,
        error_message: str | None = None,
        chunk_count: int | None = None,
    ) -> Document:
        """
        Update the ingestion status of a document.

        Args:
            document_id: Target document UUID.
            status: New status string (pending/processing/completed/failed).
            error_message: Set when status is 'failed'.
            chunk_count: Set when status is 'completed'.
        """
        ...

    @abstractmethod
    async def get_all_by_user(
        self,
        user_id: str | None,
        limit: int,
        offset: int,
    ) -> list[Document]:
        """Return all active documents for a user, newest first."""
        ...

    @abstractmethod
    async def soft_delete(self, document_id: str) -> None:
        """
        Soft-delete a document by setting is_active=False.

        Raises:
            DocumentNotFoundError: If document does not exist.
            PersistenceError: If update fails.
        """
        ...