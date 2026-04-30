"""Unit tests for ChatResult entity."""
from src.entities.chat_result.chat_result import ChatResult


def make_result(**kwargs) -> ChatResult:
    defaults = dict(conversation_id="conv-1", message_id="msg-1", answer="Test answer")
    return ChatResult(**{**defaults, **kwargs})


def test_has_sources_true():
    assert make_result(sources=["chunk-1"]).has_sources() is True


def test_has_sources_false():
    assert make_result(sources=[]).has_sources() is False


def test_to_sse_done_payload_has_required_keys():
    result = make_result()
    payload = result.to_sse_done_payload()
    for key in ("conversation_id", "message_id", "answer", "sources", "model_name", "created_at"):
        assert key in payload
