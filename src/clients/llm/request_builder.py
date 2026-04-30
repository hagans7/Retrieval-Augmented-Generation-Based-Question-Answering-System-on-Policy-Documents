"""
LLM request builder — constructs OpenAI-compatible request bodies.

Responsible only for building the JSON payload sent to the LLM provider.
Handles differences between stream and non-stream requests.
Handles optional tool definitions inclusion.
"""

from __future__ import annotations


class LLMRequestBuilder:
    """
    Builds request body dicts for OpenAI-compatible chat completions API.

    Constructed with provider-level defaults; overridable per request.

    Args:
        default_model: Fallback model name when no per-request model is given.
        temperature: Sampling temperature (0.0 = deterministic).
        max_tokens: Maximum tokens to generate per response.
    """

    def __init__(
        self,
        default_model: str,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> None:
        self._default_model = default_model
        self._temperature = temperature
        self._max_tokens = max_tokens

    def build(
        self,
        messages: list[dict[str, str]],
        model: str | None,
        stream: bool,
        tools: list[dict] | None = None,
    ) -> dict:
        """
        Build the complete request body for a chat completion call.

        Args:
            messages: List of message dicts: [{"role": ..., "content": ...}].
            model: Model name to use. Falls back to default if None.
            stream: If True, adds "stream": true to the payload.
            tools: Optional list of tool definition dicts.

        Returns:
            Complete request body dict ready for JSON serialization.
        """
        body: dict = {
            "model": model or self._default_model,
            "messages": messages,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "stream": stream,
        }

        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"

        return body

    def build_headers(self, api_key: str) -> dict[str, str]:
        """
        Build the required HTTP headers for the provider API.

        Includes Authorization and HTTP-Referer (required by OpenRouter).

        Args:
            api_key: Provider API key.

        Returns:
            Headers dict.
        """
        return {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://legal.ai",
            "X-Title": "legal Legal AI",
        }
