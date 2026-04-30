"""
SSE event payload builder.

Constructs flat payload dicts for every SSE event type.
Separates payload construction from SSE string formatting.
"""
from __future__ import annotations
from datetime import datetime, timezone


class SSEEventBuilder:
    """Builds payload dicts for each SSE event type."""

    @staticmethod
    def conversation_started(conversation_id: str, model_name: str, system_prompt_id: str | None) -> dict:
        return {
            "conversation_id": conversation_id,
            "model_name": model_name,
            "system_prompt_id": system_prompt_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    @staticmethod
    def plan_created(steps: list[str], estimated_complexity: str = "medium") -> dict:
        return {"steps": steps, "estimated_complexity": estimated_complexity}

    @staticmethod
    def tool_called(tool_name: str, parameters: dict) -> dict:
        return {"tool_name": tool_name, "parameters": parameters}

    @staticmethod
    def evidence_found(source_count: int, chunk_ids: list[str]) -> dict:
        return {"source_count": source_count, "chunk_ids": chunk_ids}

    @staticmethod
    def generation_started() -> dict:
        return {"timestamp": datetime.now(timezone.utc).isoformat()}

    @staticmethod
    def token(content: str) -> dict:
        return {"content": content}

    @staticmethod
    def done(payload: dict) -> dict:
        return payload

    @staticmethod
    def error(code: str, message: str) -> dict:
        return {"code": code, "message": message}
