"""
Langfuse v2 observability client — concrete implementation of BaseObservabilityClient.

Wraps the Langfuse SDK. All methods suppress exceptions internally so
observability failures never propagate to the main request flow.

Provider selection (in providers/infrastructure/clients.py):
    If LANGFUSE keys are set → LangfuseObservabilityClient
    Otherwise               → NoOpObservabilityClient

This class is never instantiated directly by services.
Use src.providers.get_observability_client() which returns BaseObservabilityClient.
"""
from __future__ import annotations

import contextlib
from typing import Any

from src.core.logging.logger import get_logger
from src.interfaces.clients.base_observability_client import BaseObservabilityClient

logger = get_logger(__name__)


class LangfuseObservabilityClient(BaseObservabilityClient):
    """
    Langfuse v2 backend for observability tracing.

    Initialized with public/secret key and host URL.
    All public methods suppress exceptions — observability must never
    break the main request flow.
    """

    def __init__(self, public_key: str, secret_key: str, host: str) -> None:
        from langfuse import Langfuse  # type: ignore[import]
        self._client = Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
        )
        logger.info("Langfuse observability enabled", extra={"host": host})

    @property
    def enabled(self) -> bool:
        return True

    def start_trace(
        self,
        name: str,
        session_id: str = "",
        user_id: str = "",
        input: dict | None = None,
        metadata: dict | None = None,
        tags: list[str] | None = None,
    ) -> Any:
        with contextlib.suppress(Exception):
            return self._client.trace(
                name=name,
                session_id=session_id or None,
                user_id=user_id or None,
                input=input or {},
                metadata=metadata or {},
                tags=tags or [],
            )
        return None

    def end_trace(self, trace: Any, output: dict | None = None) -> None:
        if trace is None:
            return
        with contextlib.suppress(Exception):
            trace.update(output=output or {})

    def start_span(
        self,
        trace: Any,
        name: str,
        input: dict | None = None,
        metadata: dict | None = None,
    ) -> Any:
        if trace is None:
            return None
        with contextlib.suppress(Exception):
            return trace.span(name=name, input=input or {}, metadata=metadata or {})
        return None

    def end_span(
        self,
        span: Any,
        output: dict | None = None,
        metadata: dict | None = None,
        level: str = "DEFAULT",
    ) -> None:
        if span is None:
            return
        with contextlib.suppress(Exception):
            span.end(output=output or {}, metadata=metadata or {}, level=level)

    def start_generation(
        self,
        trace: Any,
        name: str,
        model: str = "",
        input: Any = None,
        metadata: dict | None = None,
    ) -> Any:
        if trace is None:
            return None
        with contextlib.suppress(Exception):
            return trace.generation(
                name=name, model=model, input=input, metadata=metadata or {}
            )
        return None

    def end_generation(
        self,
        generation: Any,
        output: Any = None,
        usage: dict | None = None,
        metadata: dict | None = None,
    ) -> None:
        if generation is None:
            return
        with contextlib.suppress(Exception):
            generation.end(output=output, usage=usage, metadata=metadata or {})

    def log_event(
        self,
        trace: Any,
        name: str,
        input: dict | None = None,
        output: dict | None = None,
        metadata: dict | None = None,
        level: str = "DEFAULT",
    ) -> None:
        if trace is None:
            return
        with contextlib.suppress(Exception):
            trace.event(
                name=name,
                input=input or {},
                output=output or {},
                metadata=metadata or {},
                level=level,
            )

    def score_trace(
        self,
        trace: Any,
        name: str,
        value: float,
        comment: str = "",
    ) -> None:
        if trace is None:
            return
        with contextlib.suppress(Exception):
            self._client.score(
                trace_id=trace.id,
                name=name,
                value=value,
                comment=comment,
            )

    def flush(self) -> None:
        with contextlib.suppress(Exception):
            self._client.flush()