"""
LLM response parser — extracts content from non-stream API responses.

Handles extraction of the generated text from OpenAI-compatible
JSON responses. Also handles tool call extraction for agent nodes.
"""

from __future__ import annotations

import json

from src.core.exceptions.infrastructure import LLMInferenceError
from src.core.logging.logger import get_logger

logger = get_logger(__name__)


class LLMResponseParser:
    """
    Parses non-stream responses from OpenAI-compatible LLM APIs.

    Handles both plain text responses and tool call responses.
    """

    @staticmethod
    def extract_content(response_json: dict) -> str:
        """
        Extract the text content from a chat completion response.

        Args:
            response_json: Parsed JSON response dict from the LLM API.

        Returns:
            The generated text string from choices[0].message.content.

        Raises:
            LLMInferenceError: If the response structure is unexpected
                               or content cannot be extracted.
        """
        try:
            choices = response_json.get("choices", [])
            if not choices:
                raise LLMInferenceError(
                    message="LLM response contained no choices.",
                    context={"response_keys": list(response_json.keys())},
                )
            message = choices[0].get("message", {})
            content = message.get("content")
            if content is None:
                raise LLMInferenceError(
                    message="LLM response message had no content.",
                    context={"message_keys": list(message.keys())},
                )
            return content
        except LLMInferenceError:
            raise
        except Exception as exc:
            raise LLMInferenceError(
                message="Failed to parse LLM response.",
                context={"error": str(exc)},
            ) from exc

    @staticmethod
    def extract_tool_call(response_json: dict) -> dict | None:
        """
        Extract tool call arguments from a response, if present.

        Returns None if the response is plain text (no tool calls).

        Args:
            response_json: Parsed JSON response dict.

        Returns:
            Parsed tool call arguments dict, or None if no tool call.

        Raises:
            LLMInferenceError: If tool call arguments cannot be parsed.
        """
        try:
            choices = response_json.get("choices", [])
            if not choices:
                return None
            message = choices[0].get("message", {})
            tool_calls = message.get("tool_calls")
            if not tool_calls:
                return None
            # Take first tool call
            arguments_str = tool_calls[0].get("function", {}).get("arguments", "{}")
            return json.loads(arguments_str)
        except json.JSONDecodeError as exc:
            raise LLMInferenceError(
                message="Failed to parse tool call arguments as JSON.",
                context={"error": str(exc)},
            ) from exc
        except Exception as exc:
            raise LLMInferenceError(
                message="Failed to extract tool call from response.",
                context={"error": str(exc)},
            ) from exc
