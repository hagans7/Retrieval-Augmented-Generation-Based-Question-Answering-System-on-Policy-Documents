"""
URL signer — generates presigned URLs for direct object storage access.

Presigned URLs allow temporary direct access to private files
without routing the download through the application server.
"""

from __future__ import annotations

from src.core.exceptions.infrastructure import StorageError
from src.core.logging.logger import get_logger

logger = get_logger(__name__)


class UrlSigner:
    """
    Generates presigned URLs using an aiobotocore S3 client.

    Args:
        s3_client: An active aiobotocore S3 client (async context manager).
        bucket: The target S3 bucket name.
    """

    def __init__(self, s3_client, bucket: str) -> None:
        self._s3 = s3_client
        self._bucket = bucket

    async def generate(self, key: str, expiry_seconds: int) -> str:
        """
        Generate a presigned GET URL for a storage object.

        Args:
            key: The storage key of the object.
            expiry_seconds: URL validity duration in seconds.

        Returns:
            Presigned URL string.

        Raises:
            StorageError: If URL generation fails.
        """
        try:
            url = await self._s3.generate_presigned_url(
                "get_object",
                Params={"Bucket": self._bucket, "Key": key},
                ExpiresIn=expiry_seconds,
            )
            return url
        except Exception as exc:
            logger.error(
                "Presigned URL generation failed",
                extra={"key": key, "error": str(exc)},
                exc_info=True,
            )
            raise StorageError(
                message=f"Failed to generate presigned URL for key '{key}'.",
                context={"key": key, "error": str(exc)},
            ) from exc
