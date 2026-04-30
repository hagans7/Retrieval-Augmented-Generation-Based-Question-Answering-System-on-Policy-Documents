"""
Integration tests for POST /api/v1/chat endpoint.

conversation_id is now optional (null → auto-create).
Tests updated to reflect this behavior.
"""
from __future__ import annotations
from datetime import datetime, timezone
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock
import pytest
import uuid

from src.entities.chat_result.chat_result import ChatResult

CONV_ID = str(uuid.uuid4())

FAKE_RESULT = ChatResult(
    conversation_id=CONV_ID,
    message_id=str(uuid.uuid4()),
    answer="Wanprestasi adalah kegagalan memenuhi kewajiban kontrak.",
    sources=["chunk-1"],
    model_name="qwen/qwen-turbo",
    system_prompt_id=None,
    created_at=datetime.now(timezone.utc),
)


@pytest.fixture
def client():
    from src.main import app
    from src.providers import get_process_chat_service

    mock_service = AsyncMock()
    mock_service.execute.return_value = FAKE_RESULT

    app.dependency_overrides[get_process_chat_service] = lambda: mock_service
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    app.dependency_overrides.clear()


def test_chat_non_stream_returns_200(client):
    response = client.post(
        "/api/v1/chat",
        json={
            "conversation_id": CONV_ID,
            "message": "Apa itu wanprestasi?",
            "stream": False,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert "answer" in body["data"]
    assert body["data"]["answer"] == FAKE_RESULT.answer


def test_chat_returns_conversation_id(client):
    response = client.post(
        "/api/v1/chat",
        json={
            "conversation_id": CONV_ID,
            "message": "Test question",
            "stream": False,
        },
    )
    assert response.json()["data"]["conversation_id"] == CONV_ID


def test_chat_rejects_empty_message(client):
    """Empty or whitespace-only message should return 422."""
    response = client.post(
        "/api/v1/chat",
        json={
            "conversation_id": CONV_ID,
            "message": "   ",
            "stream": False,
        },
    )
    assert response.status_code == 422


def test_chat_allows_null_conversation_id(client):
    """
    conversation_id=null is valid — service auto-creates conversation.
    Should return 200, not 422.
    """
    response = client.post(
        "/api/v1/chat",
        json={
            "conversation_id": None,
            "message": "Hello, auto-create a conversation for me.",
            "stream": False,
        },
    )
    assert response.status_code == 200
    assert response.json()["data"]["conversation_id"] == CONV_ID


def test_chat_omitting_conversation_id_also_valid(client):
    """
    Omitting conversation_id entirely also triggers auto-create (defaults to None).
    """
    response = client.post(
        "/api/v1/chat",
        json={"message": "Hello"},
    )
    assert response.status_code == 200


def test_health_liveness_returns_200(client):
    response = client.get("/health/liveness")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"