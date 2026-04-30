"""Unit tests for Message entity."""
import pytest
from src.entities.message.message import Message


def make_message(**kwargs) -> Message:
    defaults = dict(message_id="msg-1", conversation_id="conv-1", role="user", content="Hello")
    return Message(**{**defaults, **kwargs})


def test_is_from_user():
    assert make_message(role="user").is_from_user() is True

def test_is_from_assistant():
    assert make_message(role="assistant").is_from_assistant() is True

def test_to_llm_format():
    msg = make_message(role="user", content="What is Pasal 5?")
    fmt = msg.to_llm_format()
    assert fmt == {"role": "user", "content": "What is Pasal 5?"}

def test_to_llm_format_assistant():
    msg = make_message(role="assistant", content="Pasal 5 membahas...")
    assert msg.to_llm_format()["role"] == "assistant"
