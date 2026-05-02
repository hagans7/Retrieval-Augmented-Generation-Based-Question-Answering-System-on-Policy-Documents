"""
Abstract base for all observability clients.

Implementing this interface allows swapping observability backends
(Langfuse, Langsmith, Phoenix, no-op) without changing agent_runner code.

All methods return the created object (trace/span/generation) so callers
can pass it to subsequent calls. Implementations must never raise exceptions
from observability calls — observability must never break the main flow.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseObservabilityClient(ABC):
    """Contract for all observability/tracing backends."""

    @property
    @abstractmethod
    def enabled(self) -> bool:
        """Return True if this client is actively sending data."""
        ...

    # ── Trace ─────────────────────────────────────────────────────────────────

    @abstractmethod
    def start_trace(
        self,
        name: str,
        session_id: str = "",
        user_id: str = "",
        input: dict | None = None,
        metadata: dict | None = None,
        tags: list[str] | None = None,
    ) -> Any:
        """
        Start a top-level trace representing one end-to-end request.

        Returns:
            Trace handle passed to subsequent span/generation/event calls.
        """
        ...

    @abstractmethod
    def end_trace(self, trace: Any, output: dict | None = None) -> None:
        """Finalize the trace with output data."""
        ...

    # ── Span ──────────────────────────────────────────────────────────────────

    @abstractmethod
    def start_span(
        self,
        trace: Any,
        name: str,
        input: dict | None = None,
        metadata: dict | None = None,
    ) -> Any:
        """
        Start a span representing one unit of work within a trace.

        Returns:
            Span handle passed to end_span().
        """
        ...

    @abstractmethod
    def end_span(
        self,
        span: Any,
        output: dict | None = None,
        metadata: dict | None = None,
        level: str = "DEFAULT",
    ) -> None:
        """Close a span. level: DEFAULT | DEBUG | WARNING | ERROR."""
        ...

    # ── LLM Generation ────────────────────────────────────────────────────────

    @abstractmethod
    def start_generation(
        self,
        trace: Any,
        name: str,
        model: str = "",
        input: Any = None,
        metadata: dict | None = None,
    ) -> Any:
        """
        Start a generation span for one LLM call.

        Generations are special spans that capture model, input, output,
        and token usage — visible in LLM cost dashboards.

        Returns:
            Generation handle passed to end_generation().
        """
        ...

    @abstractmethod
    def end_generation(
        self,
        generation: Any,
        output: Any = None,
        usage: dict | None = None,
        metadata: dict | None = None,
    ) -> None:
        """
        Close a generation.

        Args:
            usage: dict with keys: input, output, total (token counts).
        """
        ...

    # ── Event ─────────────────────────────────────────────────────────────────

    @abstractmethod
    def log_event(
        self,
        trace: Any,
        name: str,
        input: dict | None = None,
        output: dict | None = None,
        metadata: dict | None = None,
        level: str = "DEFAULT",
    ) -> None:
        """Log a point-in-time event (no duration) on the trace."""
        ...

    # ── Score ─────────────────────────────────────────────────────────────────

    @abstractmethod
    def score_trace(
        self,
        trace: Any,
        name: str,
        value: float,
        comment: str = "",
    ) -> None:
        """Attach a numeric quality score to a trace."""
        ...

    # ── Flush ─────────────────────────────────────────────────────────────────

    @abstractmethod
    def flush(self) -> None:
        """Flush pending events to the backend. Call at end of each request."""
        ...