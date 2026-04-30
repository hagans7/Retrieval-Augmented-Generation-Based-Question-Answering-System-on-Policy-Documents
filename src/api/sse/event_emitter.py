"""
SSE event emitter.

Formats event type + payload into the SSE wire format:
    event: {type}\\ndata: {json}\\n\\n

All SSE string formatting is isolated here.
"""
from __future__ import annotations
import json
from collections.abc import AsyncIterator


class SSEEventEmitter:
    """Formats and yields SSE-spec strings for StreamingResponse."""

    @staticmethod
    def format(event_type: str, payload: dict) -> str:
        """
        Format one SSE event as a wire-format string.

        Args:
            event_type: The SSE event name.
            payload: Dict to JSON-serialize as event data.

        Returns:
            SSE-formatted string ending with double newline.
        """
        data = json.dumps(payload, ensure_ascii=False, default=str)
        return f"event: {event_type}\ndata: {data}\n\n"

    @staticmethod
    async def stream_from_service(
        service_iterator: AsyncIterator[tuple[str, dict]],
    ) -> AsyncIterator[str]:
        """
        Convert service-level (event_type, payload) tuples to SSE strings.

        Args:
            service_iterator: Async iterator yielding (event_type, payload) tuples.

        Yields:
            SSE-formatted strings.
        """
        async for event_type, payload in service_iterator:
            yield SSEEventEmitter.format(event_type, payload)
