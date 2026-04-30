"""
LLM stream processor — parses SSE chunks from the LLM provider.

Responsible for iterating the raw SSE byte stream, parsing each
JSON chunk, and yielding the incremental token strings to the caller.
"""

from __future__ import annotations

import json
from typing import AsyncIterator

import httpx

from src.core.exceptions.infrastructure import LLMInferenceError
from src.core.logging.logger import get_logger

logger = get_logger(__name__)

_DONE_SIGNAL = "[DONE]"
_DATA_PREFIX = "data: "


class LLMStreamProcessor:
    """
    Processes an httpx streaming response from an OpenAI-compatible provider.

    Iterates SSE lines, filters data events, parses JSON chunks,
    and yields delta content strings until [DONE] is received.
    """

    @staticmethod
    async def process(response: httpx.Response) -> AsyncIterator[str]:
        """
        Yield token strings from a streaming LLM response.

        Args:
            response: An open httpx streaming response with text/event-stream content.

        Yields:
            String chunks as they arrive from the provider.

        Raises:
            LLMInferenceError: If the stream is interrupted or malformed.
        """
        try:
            async for line in response.aiter_lines():
                line = line.strip()

                if not line or not line.startswith(_DATA_PREFIX):
                    continue

                data = line[len(_DATA_PREFIX):]

                if data == _DONE_SIGNAL:
                    break

                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    # Partial or malformed chunk — skip silently
                    continue

                choices = chunk.get("choices", [])
                if not choices:
                    continue

                delta = choices[0].get("delta", {})
                content = delta.get("content")

                if content:
                    yield content

        except httpx.StreamError as exc:
            raise LLMInferenceError(
                message="LLM stream was interrupted.",
                context={"error": str(exc)},
            ) from exc
