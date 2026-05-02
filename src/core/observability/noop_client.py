"""
No-op observability client — used when observability is disabled.

Returned by the provider when LANGFUSE keys are absent or when running tests.
All methods are silent no-ops that return None.
Service layer code works identically regardless of which implementation is active.
"""
from __future__ import annotations

from typing import Any

from src.interfaces.clients.base_observability_client import BaseObservabilityClient


class NoOpObservabilityClient(BaseObservabilityClient):
    """Silent no-op: all calls succeed without doing anything."""

    @property
    def enabled(self) -> bool:
        return False

    def start_trace(self, name: str, session_id: str = "", user_id: str = "",
                    input: dict | None = None, metadata: dict | None = None,
                    tags: list[str] | None = None) -> Any:
        return None

    def end_trace(self, trace: Any, output: dict | None = None) -> None:
        pass

    def start_span(self, trace: Any, name: str, input: dict | None = None,
                   metadata: dict | None = None) -> Any:
        return None

    def end_span(self, span: Any, output: dict | None = None,
                 metadata: dict | None = None, level: str = "DEFAULT") -> None:
        pass

    def start_generation(self, trace: Any, name: str, model: str = "",
                         input: Any = None, metadata: dict | None = None) -> Any:
        return None

    def end_generation(self, generation: Any, output: Any = None,
                       usage: dict | None = None, metadata: dict | None = None) -> None:
        pass

    def log_event(self, trace: Any, name: str, input: dict | None = None,
                  output: dict | None = None, metadata: dict | None = None,
                  level: str = "DEFAULT") -> None:
        pass

    def score_trace(self, trace: Any, name: str, value: float, comment: str = "") -> None:
        pass

    def flush(self) -> None:
        pass