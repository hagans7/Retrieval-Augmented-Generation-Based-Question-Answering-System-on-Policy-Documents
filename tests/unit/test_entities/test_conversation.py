"""Unit tests for Conversation entity."""
import pytest
from src.entities.conversation.conversation import Conversation
from src.entities.message.message import Message


def make_conv(**kwargs) -> Conversation:
    return Conversation(conversation_id="conv-1", title="Test", **kwargs)


def make_message(role="user", content="hi") -> Message:
    return Message(message_id="m1", conversation_id="conv-1", role=role, content=content)


def test_is_empty_when_no_messages():
    assert make_conv().is_empty() is True


def test_is_not_empty_with_messages():
    conv = make_conv()
    conv.messages = [make_message()]
    assert conv.is_empty() is False


def test_get_recent_messages_respects_limit():
    conv = make_conv()
    conv.messages = [make_message(content=str(i)) for i in range(10)]
    recent = conv.get_recent_messages(3)
    assert len(recent) == 3
    assert recent[-1].content == "9"


def test_get_last_assistant_message():
    conv = make_conv()
    conv.messages = [
        make_message(role="user", content="Q"),
        make_message(role="assistant", content="A"),
    ]
    last = conv.get_last_assistant_message()
    assert last is not None
    assert last.content == "A"


def test_get_last_assistant_message_none_when_no_assistant():
    conv = make_conv()
    conv.messages = [make_message(role="user")]
    assert conv.get_last_assistant_message() is None
