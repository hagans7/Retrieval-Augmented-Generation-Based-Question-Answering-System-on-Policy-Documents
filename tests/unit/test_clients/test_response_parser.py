"""Unit tests for LLM response parser."""
import pytest
from src.clients.llm.response_parser import LLMResponseParser
from src.core.exceptions.infrastructure import LLMInferenceError


def test_extract_content_success():
    response = {"choices": [{"message": {"content": "Hello"}}]}
    assert LLMResponseParser.extract_content(response) == "Hello"


def test_extract_content_no_choices_raises():
    with pytest.raises(LLMInferenceError):
        LLMResponseParser.extract_content({"choices": []})


def test_extract_tool_call_returns_none_when_absent():
    response = {"choices": [{"message": {"content": "plain"}}]}
    assert LLMResponseParser.extract_tool_call(response) is None


def test_extract_tool_call_parses_arguments():
    import json
    args = {"query_type": "retrieval"}
    response = {
        "choices": [{
            "message": {
                "tool_calls": [{"function": {"arguments": json.dumps(args)}}]
            }
        }]
    }
    assert LLMResponseParser.extract_tool_call(response) == args
