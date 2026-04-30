"""
Redis cache client — implements BaseCacheClient.

All redis.RedisError exceptions are caught here and normalized
to CacheError. CacheError is NON-CRITICAL — service layer
catches it, logs a WARNING, and continues without cache.
"""

from __future__ import annotations

import redis.asyncio as aioredis

from src.core.exceptions.infrastructure import CacheError
from src.core.logging.logger import get_logger
from src.interfaces.clients.base_cache_client import BaseCacheClient


class RedisCacheClient(BaseCacheClient):
    """
    Async Redis cache client.

    Args:
        url: Redis connection URL (e.g. "redis://localhost:6379/0").
    """

    def __init__(self, url: str) -> None:
        self._url = url
        self._client: aioredis.Redis = aioredis.from_url(
            url,
            encoding="utf-8",
            decode_responses=True,
        )
        self._logger = get_logger(__name__)

    async def get(self, key: str) -> str | None:
        """
        Retrieve a cached value by key.

        Returns:
            Cached string value, or None if the key does not exist.

        Raises:
            CacheError: If Redis is unreachable.
        """
        try:
            return await self._client.get(key)
        except aioredis.RedisError as exc:
            raise CacheError(
                message=f"Cache GET failed for key '{key}'.",
                context={"key": key, "error": str(exc)},
            ) from exc

    async def set(self, key: str, value: str, ttl_seconds: int) -> None:
        """
        Store a value in the cache with a TTL.

        Raises:
            CacheError: If Redis is unreachable.
        """
        try:
            await self._client.set(key, value, ex=ttl_seconds)
        except aioredis.RedisError as exc:
            raise CacheError(
                message=f"Cache SET failed for key '{key}'.",
                context={"key": key, "ttl_seconds": ttl_seconds, "error": str(exc)},
            ) from exc

    async def delete(self, key: str) -> None:
        """
        Delete a key from the cache.

        Raises:
            CacheError: If Redis is unreachable.
        """
        try:
            await self._client.delete(key)
        except aioredis.RedisError as exc:
            raise CacheError(
                message=f"Cache DELETE failed for key '{key}'.",
                context={"key": key, "error": str(exc)},
            ) from exc

    async def exists(self, key: str) -> bool:
        """
        Check if a key exists and has not expired.

        Raises:
            CacheError: If Redis is unreachable.
        """
        try:
            result = await self._client.exists(key)
            return bool(result)
        except aioredis.RedisError as exc:
            raise CacheError(
                message=f"Cache EXISTS check failed for key '{key}'.",
                context={"key": key, "error": str(exc)},
            ) from exc

    async def ping(self) -> bool:
        """Return True if Redis responds to PING. Used by health check."""
        try:
            return await self._client.ping()
        except aioredis.RedisError:
            return False

    async def close(self) -> None:
        """Close the Redis connection pool."""
        await self._client.aclose()
