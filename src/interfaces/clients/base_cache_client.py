"""Abstract base for all cache clients."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseCacheClient(ABC):
    """
    Contract for all cache clients (Redis, etc.).

    CacheError raised by implementations is NON-CRITICAL.
    Service layer catches CacheError, logs a WARNING, and continues
    without cache (graceful degradation).
    """

    @abstractmethod
    async def get(self, key: str) -> str | None:
        """
        Retrieve a value from the cache.

        Args:
            key: Cache key to look up.

        Returns:
            Cached value string, or None if key does not exist.

        Raises:
            CacheError: If the cache is unreachable.
        """
        ...

    @abstractmethod
    async def set(self, key: str, value: str, ttl_seconds: int) -> None:
        """
        Store a value in the cache with a TTL.

        Args:
            key: Cache key.
            value: String value to cache.
            ttl_seconds: Time-to-live in seconds.

        Raises:
            CacheError: If the cache is unreachable.
        """
        ...

    @abstractmethod
    async def delete(self, key: str) -> None:
        """
        Delete a key from the cache.

        Args:
            key: Cache key to delete.

        Raises:
            CacheError: If the cache is unreachable.
        """
        ...

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """
        Check whether a key exists in the cache.

        Args:
            key: Cache key to check.

        Returns:
            True if the key exists and has not expired.

        Raises:
            CacheError: If the cache is unreachable.
        """
        ...
