"""Unit tests for LLM request builder."""
from src.clients.llm.request_builder import LLMRequestBuilder


def make_builder() -> LLMRequestBuilder:
    return LLMRequestBuilder(default_model="qwen/qwen3-6b-plus:free")


def test_build_uses_provided_model():
    body = make_builder().build(messages=[], model="gpt-4", stream=False)
    assert body["model"] == "gpt-4"


def test_build_falls_back_to_default_model():
    body = make_builder().build(messages=[], model=None, stream=False)
    assert body["model"] == "qwen/qwen3-6b-plus:free"


def test_build_stream_flag():
    body = make_builder().build(messages=[], model="m", stream=True)
    assert body["stream"] is True


def test_build_includes_tools_when_provided():
    tools = [{"type": "function", "function": {"name": "test"}}]
    body = make_builder().build(messages=[], model="m", stream=False, tools=tools)
    assert "tools" in body
    assert body["tool_choice"] == "auto"


def test_build_excludes_tools_when_none():
    body = make_builder().build(messages=[], model="m", stream=False, tools=None)
    assert "tools" not in body


def test_build_headers_has_authorization():
    headers = make_builder().build_headers("sk-test")
    assert headers["Authorization"] == "Bearer sk-test"
