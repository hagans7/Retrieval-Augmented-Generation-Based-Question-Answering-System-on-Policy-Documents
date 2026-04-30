"""Abstract base for all object storage clients."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseStorageClient(ABC):
    """Contract for all S3-compatible object storage clients."""

    @abstractmethod
    async def upload_file(
        self,
        key: str,
        data: bytes,
        content_type: str,
    ) -> str:
        """
        Upload bytes to object storage at the given key.

        Args:
            key: Storage path key (e.g. "pdf/doc-uuid/filename.pdf").
            data: File bytes to upload.
            content_type: MIME type (e.g. "application/pdf").

        Returns:
            The storage key used for upload.

        Raises:
            StorageError: If the upload fails.
        """
        ...

    @abstractmethod
    async def download_file(self, key: str) -> bytes:
        """
        Download file bytes from object storage.

        Args:
            key: Storage path key.

        Returns:
            File bytes.

        Raises:
            StorageError: If the download fails or key does not exist.
        """
        ...

    @abstractmethod
    async def delete_file(self, key: str) -> None:
        """
        Delete a file from object storage.

        Args:
            key: Storage path key to delete.

        Raises:
            StorageError: If the deletion fails.
        """
        ...

    @abstractmethod
    async def generate_presigned_url(
        self,
        key: str,
        expiry_seconds: int = 3600,
    ) -> str:
        """
        Generate a time-limited presigned URL for direct file access.

        Args:
            key: Storage path key.
            expiry_seconds: URL validity duration. Default 1 hour.

        Returns:
            Presigned URL string.

        Raises:
            StorageError: If URL generation fails.
        """
        ...
