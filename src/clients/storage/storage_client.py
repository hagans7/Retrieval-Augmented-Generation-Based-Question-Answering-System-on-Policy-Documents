"""
S3-compatible storage client — implements BaseStorageClient.

Works with MinIO for development and AWS S3 for production.
Switch by changing STORAGE_ENDPOINT env var — no code changes needed.
"""

from __future__ import annotations

import aiobotocore.session

from src.clients.storage.key_builder import StorageKeyBuilder
from src.clients.storage.url_signer import UrlSigner
from src.core.exceptions.infrastructure import StorageError
from src.core.logging.logger import get_logger
from src.interfaces.clients.base_storage_client import BaseStorageClient


class S3CompatibleStorageClient(BaseStorageClient):
    """
    Async S3-compatible object storage client.

    Args:
        endpoint: Storage endpoint URL (e.g. "http://minio:9000").
        access_key: S3 access key.
        secret_key: S3 secret key.
        bucket: Target bucket name.
        use_ssl: Whether to use HTTPS.
    """

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        use_ssl: bool = False,
    ) -> None:
        self._endpoint = endpoint
        self._access_key = access_key
        self._secret_key = secret_key
        self._bucket = bucket
        self._use_ssl = use_ssl
        self._session = aiobotocore.session.get_session()
        self._logger = get_logger(__name__)

    def _client_context(self):
        """Return an aiobotocore S3 client async context manager."""
        return self._session.create_client(
            "s3",
            endpoint_url=self._endpoint,
            aws_access_key_id=self._access_key,
            aws_secret_access_key=self._secret_key,
            use_ssl=self._use_ssl,
        )

    async def upload_file(
        self,
        key: str,
        data: bytes,
        content_type: str,
    ) -> str:
        """
        Upload bytes to object storage.

        Returns:
            The storage key used.

        Raises:
            StorageError: If the upload fails.
        """
        try:
            async with self._client_context() as s3:
                await s3.put_object(
                    Bucket=self._bucket,
                    Key=key,
                    Body=data,
                    ContentType=content_type,
                )
            self._logger.info(
                "File uploaded to storage",
                extra={"key": key, "bucket": self._bucket, "size_bytes": len(data)},
            )
            return key
        except Exception as exc:
            raise StorageError(
                message=f"Failed to upload file to storage key '{key}'.",
                context={"key": key, "error": str(exc)},
            ) from exc

    async def download_file(self, key: str) -> bytes:
        """
        Download file bytes from object storage.

        Raises:
            StorageError: If the download fails.
        """
        try:
            async with self._client_context() as s3:
                response = await s3.get_object(Bucket=self._bucket, Key=key)
                async with response["Body"] as stream:
                    return await stream.read()
        except Exception as exc:
            raise StorageError(
                message=f"Failed to download file from storage key '{key}'.",
                context={"key": key, "error": str(exc)},
            ) from exc

    async def delete_file(self, key: str) -> None:
        """
        Delete a file from object storage.

        Raises:
            StorageError: If the deletion fails.
        """
        try:
            async with self._client_context() as s3:
                await s3.delete_object(Bucket=self._bucket, Key=key)
            self._logger.info("File deleted from storage", extra={"key": key})
        except Exception as exc:
            raise StorageError(
                message=f"Failed to delete file at storage key '{key}'.",
                context={"key": key, "error": str(exc)},
            ) from exc

    async def generate_presigned_url(
        self,
        key: str,
        expiry_seconds: int = 3600,
    ) -> str:
        """
        Generate a presigned URL for direct file access.

        Raises:
            StorageError: If URL generation fails.
        """
        async with self._client_context() as s3:
            signer = UrlSigner(s3, self._bucket)
            return await signer.generate(key, expiry_seconds)
