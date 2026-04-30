"""
Abstract base for all LLM clients.

Implementing this interface allows swapping LLM providers
(OpenRouter, OpenAI, local) without changing service layer code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator


class BaseLLMClient(ABC):
    """Contract for all LLM provider clients."""

    @abstractmethod
    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        model: str,
        tools: list[dict] | None = None,
    ) -> AsyncIterator[str]:
        """
        Stream a completion from the LLM, yielding tokens as they arrive.

        Args:
            messages: List of message dicts in OpenAI format: [{"role": ..., "content": ...}].
            model: Model name string to pass to the provider API.
            tools: Optional list of tool definition dicts (OpenAI tool format).

        Yields:
            String chunks as they arrive from the provider SSE stream.

        Raises:
            LLMInferenceError: If the request fails after all retries.
        """
        ...

    @abstractmethod
    async def generate(
        self,
        messages: list[dict[str, str]],
        model: str,
        tools: list[dict] | None = None,
    ) -> str:
        """
        Generate a completion from the LLM, returning the full response.

        Args:
            messages: List of message dicts in OpenAI format.
            model: Model name string.
            tools: Optional list of tool definition dicts.

        Returns:
            Full response string.

        Raises:
            LLMInferenceError: If the request fails after all retries.
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Verify connectivity to the LLM provider.

        Returns:
            True if the provider is reachable and responding.

        Raises:
            LLMInferenceError: If the health check itself fails unexpectedly.
        """
        ...
