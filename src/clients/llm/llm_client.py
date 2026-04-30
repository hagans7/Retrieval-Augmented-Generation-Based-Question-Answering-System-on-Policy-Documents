"""
LLM client — entry point implementing BaseLLMClient.

Composes request_builder, response_parser, stream_processor,
and retry_handler to provide a clean generate() interface.
All httpx and provider-specific logic is isolated here.
"""

from __future__ import annotations

from typing import AsyncIterator

import httpx

from src.clients.llm.request_builder import LLMRequestBuilder
from src.clients.llm.response_parser import LLMResponseParser
from src.clients.llm.retry_handler import build_llm_retry
from src.clients.llm.stream_processor import LLMStreamProcessor
from src.core.exceptions.infrastructure import LLMInferenceError
from src.core.logging.logger import get_logger
from src.interfaces.clients.base_llm_client import BaseLLMClient


class OpenAICompatibleLLMClient(BaseLLMClient):
    """
    LLM client for any OpenAI-compatible API (OpenRouter, OpenAI, etc.).

    Configure the provider by setting LLM_BASE_URL, LLM_API_KEY,
    and LLM_DEFAULT_MODEL in environment variables.

    Args:
        base_url: Provider API base URL.
        api_key: Provider API key.
        default_model: Fallback model name.
        timeout: HTTP request timeout in seconds.
        max_retries: Maximum retry attempts on transient failures.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        default_model: str,
        timeout: int = 60,
        max_retries: int = 3,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._request_builder = LLMRequestBuilder(default_model=default_model)
        self._logger = get_logger(__name__)
        self._retry = build_llm_retry(max_retries)

    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        model: str,
        tools: list[dict] | None = None,
    ) -> AsyncIterator[str]:
        """
        Stream tokens from the LLM provider via SSE.

        Yields:
            String token chunks as they arrive.

        Raises:
            LLMInferenceError: After all retries are exhausted.
        """
        body = self._request_builder.build(
            messages=messages, model=model, stream=True, tools=tools
        )
        headers = self._request_builder.build_headers(self._api_key)

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                async with client.stream(
                    "POST",
                    f"{self._base_url}/chat/completions",
                    json=body,
                    headers=headers,
                ) as response:
                    response.raise_for_status()
                    async for token in LLMStreamProcessor.process(response):
                        yield token
        except httpx.HTTPStatusError as exc:
            raise LLMInferenceError(
                message=f"LLM provider returned HTTP {exc.response.status_code}.",
                context={"status_code": exc.response.status_code, "model": model},
            ) from exc
        except httpx.RequestError as exc:
            raise LLMInferenceError(
                message="LLM provider request failed.",
                context={"error": str(exc), "model": model},
            ) from exc

    async def generate(
        self,
        messages: list[dict[str, str]],
        model: str,
        tools: list[dict] | None = None,
    ) -> str:
        """
        Generate a full completion (non-stream) from the LLM provider.

        Applies retry logic for transient failures.

        Returns:
            Full response string.

        Raises:
            LLMInferenceError: After all retries are exhausted.
        """
        body = self._request_builder.build(
            messages=messages, model=model, stream=False, tools=tools
        )
        headers = self._request_builder.build_headers(self._api_key)

        @self._retry
        async def _call() -> str:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self._base_url}/chat/completions",
                    json=body,
                    headers=headers,
                )
                response.raise_for_status()
                return LLMResponseParser.extract_content(response.json())

        try:
            return await _call()
        except httpx.HTTPStatusError as exc:
            raise LLMInferenceError(
                message=f"LLM provider returned HTTP {exc.response.status_code}.",
                context={"status_code": exc.response.status_code, "model": model},
            ) from exc
        except httpx.RequestError as exc:
            raise LLMInferenceError(
                message="LLM provider request failed after retries.",
                context={"error": str(exc), "model": model},
            ) from exc

    async def generate_with_tool_call(
        self,
        messages: list[dict[str, str]],
        model: str,
        tools: list[dict],
    ) -> dict | None:
        """
        Generate a completion expecting a tool call response.

        Returns the parsed tool call arguments dict, or None if
        the model responded with plain text instead of a tool call.

        Args:
            messages: Conversation messages.
            model: Model name.
            tools: Tool definitions that the model may call.

        Returns:
            Parsed tool arguments dict, or None.

        Raises:
            LLMInferenceError: If the request fails.
        """
        body = self._request_builder.build(
            messages=messages, model=model, stream=False, tools=tools
        )
        headers = self._request_builder.build_headers(self._api_key)

        @self._retry
        async def _call() -> dict | None:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    f"{self._base_url}/chat/completions",
                    json=body,
                    headers=headers,
                )
                response.raise_for_status()
                return LLMResponseParser.extract_tool_call(response.json())

        try:
            return await _call()
        except httpx.HTTPStatusError as exc:
            raise LLMInferenceError(
                message=f"LLM tool call returned HTTP {exc.response.status_code}.",
                context={"status_code": exc.response.status_code},
            ) from exc

    async def health_check(self) -> bool:
        """Return True if the provider API is reachable."""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(
                    f"{self._base_url}/models",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                )
                return response.status_code == 200
        except Exception:
            return False
