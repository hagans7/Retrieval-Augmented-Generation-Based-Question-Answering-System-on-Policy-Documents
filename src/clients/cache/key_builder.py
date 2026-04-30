"""
Cache key builder — namespaced key construction for Redis.

All cache keys are built here to ensure consistent namespacing
and prevent key collisions between different domain objects.
"""

from __future__ import annotations


class CacheKeyBuilder:
    """
    Builds namespaced cache keys for different domain objects.

    All keys follow the pattern: {namespace}:{identifier}
    This prevents collision across domains and makes key scanning predictable.
    """

    @staticmethod
    def conversation_history(conversation_id: str) -> str:
        """Key for caching conversation message history."""
        return f"conversation:{conversation_id}:history"

    @staticmethod
    def model_catalog() -> str:
        """Key for caching the full list of active models."""
        return "catalog:models:active"

    @staticmethod
    def system_prompt(system_prompt_id: str) -> str:
        """Key for caching a single system prompt."""
        return f"prompt:{system_prompt_id}"
