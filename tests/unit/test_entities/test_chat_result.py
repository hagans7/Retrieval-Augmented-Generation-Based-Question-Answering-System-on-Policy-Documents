
"""
Unit tests for ChatResult entity.

ChatResult is a pure Python dataclass — no external dependencies.
Tests verify field defaults, helpers, and SSE payload serialization.
"""
from __future__ import annotations

from src.entities.chat_result.chat_result import ChatResult

SOURCE_ITEM = {
    "chunk_id": "abc-123",
    "document_id": "doc-456",
    "score": 0.87,
    "preview": "Mahasiswa diwajibkan mendaftar BPJS Kesehatan.",
}


def make_result(**kwargs) -> ChatResult:
    defaults = dict(conversation_id="conv-1", message_id="msg-1", answer="Test answer")
    return ChatResult(**{**defaults, **kwargs})


# ── has_sources ────────────────────────────────────────────────────────────────

def test_has_sources_true_with_structured_item():
    assert make_result(sources=[SOURCE_ITEM]).has_sources() is True


def test_has_sources_false_when_empty():
    assert make_result(sources=[]).has_sources() is False


# ── to_sse_done_payload ────────────────────────────────────────────────────────

def test_sse_payload_has_required_keys():
    result = make_result(sources=[SOURCE_ITEM])
    payload = result.to_sse_done_payload()
    for key in ("conversation_id", "message_id", "answer", "sources", "model_name", "created_at"):
        assert key in payload, f"Missing key: {key}"


def test_sse_payload_sources_is_list_of_dicts():
    result = make_result(sources=[SOURCE_ITEM])
    payload = result.to_sse_done_payload()
    assert isinstance(payload["sources"], list)
    assert isinstance(payload["sources"][0], dict)


def test_sse_payload_source_item_has_all_fields():
    result = make_result(sources=[SOURCE_ITEM])
    source = result.to_sse_done_payload()["sources"][0]
    assert source["chunk_id"] == "abc-123"
    assert source["document_id"] == "doc-456"
    assert source["score"] == 0.87
    assert "BPJS" in source["preview"]


def test_sse_payload_created_at_is_string():
    result = make_result()
    payload = result.to_sse_done_payload()
    assert isinstance(payload["created_at"], str)


def test_sse_payload_empty_sources():
    result = make_result(sources=[])
    payload = result.to_sse_done_payload()
    assert payload["sources"] == []