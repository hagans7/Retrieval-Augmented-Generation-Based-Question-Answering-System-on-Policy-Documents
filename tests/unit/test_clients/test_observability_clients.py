"""
Unit tests for observability client implementations.

Tests cover:
- NoOpObservabilityClient: all methods are silent, returns None/False
- BaseObservabilityClient: interface contract (ABC cannot be instantiated)
- LangfuseObservabilityClient: skipped when langfuse not configured

No live Langfuse connection required for these tests.
Live contract tests are in tests/contracts/test_observability_contract.py.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from src.interfaces.clients.base_observability_client import BaseObservabilityClient
from src.core.observability.noop_client import NoOpObservabilityClient


# ── BaseObservabilityClient interface ─────────────────────────────────────────

def test_base_is_abstract():
    """ABC cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseObservabilityClient()  # type: ignore[abstract]


def test_noop_implements_interface():
    """NoOpObservabilityClient must satisfy the ABC contract."""
    client = NoOpObservabilityClient()
    assert isinstance(client, BaseObservabilityClient)


# ── NoOpObservabilityClient behavior ─────────────────────────────────────────

class TestNoOpObservabilityClient:
    def setup_method(self):
        self.client = NoOpObservabilityClient()

    def test_enabled_is_false(self):
        assert self.client.enabled is False

    def test_start_trace_returns_none(self):
        result = self.client.start_trace("chat_request", session_id="conv-1")
        assert result is None

    def test_end_trace_does_not_raise(self):
        self.client.end_trace(None, output={"answer": "test"})

    def test_start_span_returns_none(self):
        result = self.client.start_span(None, "retrieval", input={"query": "bpjs"})
        assert result is None

    def test_end_span_does_not_raise(self):
        self.client.end_span(None, output={"count": 3})

    def test_start_generation_returns_none(self):
        result = self.client.start_generation(None, "router_llm", model="qwen/qwen-turbo")
        assert result is None

    def test_end_generation_does_not_raise(self):
        self.client.end_generation(None, output={"verdict": "sufficient"})

    def test_log_event_does_not_raise(self):
        self.client.log_event(None, "auditor_decision",
                              output={"verdict": "sufficient", "score": 0.9})

    def test_score_trace_does_not_raise(self):
        self.client.score_trace(None, "evidence_score", value=0.85)

    def test_flush_does_not_raise(self):
        self.client.flush()

    def test_full_request_lifecycle_no_error(self):
        """Simulate a complete agent request lifecycle — no method should raise."""
        trace = self.client.start_trace("chat_request", session_id="s1",
                                        input={"query": "bpjs"})
        gen = self.client.start_generation(trace, "router_llm", model="qwen")
        self.client.end_generation(gen, output={"query_type": "retrieval"})

        span = self.client.start_span(trace, "retrieval", input={"query": "bpjs"})
        self.client.end_span(span, output={"evidence_count": 1})

        gen2 = self.client.start_generation(trace, "auditor_llm", model="qwen")
        self.client.end_generation(gen2, output={"verdict": "sufficient"})
        self.client.log_event(trace, "auditor_decision",
                              output={"verdict": "sufficient", "score": 0.8})

        gen3 = self.client.start_generation(trace, "generator_llm", model="qwen")
        self.client.end_generation(gen3, output={"answer": "BPJS adalah..."})

        self.client.end_trace(trace, output={"answer_preview": "BPJS adalah..."})
        self.client.score_trace(trace, "top_evidence_score", value=0.8)
        self.client.flush()


# ── Provider selection logic ───────────────────────────────────────────────────

class TestObservabilityProvider:
    def test_returns_noop_when_keys_absent(self):
        """
        Provider returns NoOpObservabilityClient when LANGFUSE keys are empty.

        settings is imported at module level in clients.py, so we patch it there.
        lru_cache must be cleared before and after to isolate the test.
        """
        from src.providers.infrastructure import clients as clients_module
        clients_module.get_observability_client.cache_clear()

        with patch("src.providers.infrastructure.clients.settings") as mock_settings:
            mock_settings.LANGFUSE_SECRET_KEY = ""
            mock_settings.LANGFUSE_PUBLIC_KEY = ""
            mock_settings.LANGFUSE_HOST = "http://localhost:3000"

            client = clients_module.get_observability_client()

            assert isinstance(client, NoOpObservabilityClient), (
                f"Expected NoOpObservabilityClient, got {type(client).__name__}"
            )
            assert client.enabled is False

        clients_module.get_observability_client.cache_clear()

    def test_returns_base_interface(self):
        """Provider always returns BaseObservabilityClient regardless of implementation."""
        from src.providers.infrastructure import clients as clients_module
        clients_module.get_observability_client.cache_clear()

        client = clients_module.get_observability_client()

        assert isinstance(client, BaseObservabilityClient)
        clients_module.get_observability_client.cache_clear()