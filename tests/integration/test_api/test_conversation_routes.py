"""
Integration tests for /api/v1/conversations endpoints.

Why patch instead of dependency_overrides:
    conversation_routes.py calls get_conversation_repo(db) as a plain
    function after receiving db from Depends(get_db_session).
    FastAPI's dependency_overrides only intercepts Depends() declarations.
    Since get_conversation_repo is NOT declared as Depends() in the route
    parameters, overrides have no effect there.

    Solution: patch the function at its import site in the route module
    so all calls to get_conversation_repo(...) return the mock repo.
"""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import uuid
from fastapi.testclient import TestClient

from src.entities.conversation.conversation import Conversation
from src.entities.message.message import Message

# ── Constants ─────────────────────────────────────────────────
CONV_ID = str(uuid.uuid4())
MSG_ID = str(uuid.uuid4())

FAKE_CONV = Conversation(
    conversation_id=CONV_ID,
    title="Test Conversation",
    user_id=None,
    is_active=True,
    created_at=datetime.now(timezone.utc),
    updated_at=datetime.now(timezone.utc),
)

FAKE_MSG = Message(
    message_id=MSG_ID,
    conversation_id=CONV_ID,
    role="user",
    content="Apa itu wanprestasi?",
    created_at=datetime.now(timezone.utc),
)

# ── Fixtures ───────────────────────────────────────────────────

@pytest.fixture
def mock_conv_repo():
    repo = AsyncMock()
    repo.create.return_value = FAKE_CONV
    repo.get_by_id.return_value = FAKE_CONV
    repo.get_all_by_user.return_value = [FAKE_CONV]
    repo.update_title.return_value = FAKE_CONV
    repo.soft_delete.return_value = None
    return repo


@pytest.fixture
def mock_msg_repo():
    repo = AsyncMock()
    repo.get_history.return_value = [FAKE_MSG]
    return repo


@pytest.fixture
def client(mock_conv_repo, mock_msg_repo):
    """
    TestClient with patched repository functions.

    Patches get_conversation_repo and get_message_repo at the module
    where they are called in conversation_routes.py. This intercepts the
    direct function calls (not Depends declarations) inside each route handler.

    Also provides a minimal async-generator override for get_db_session
    so FastAPI can resolve the db dependency without a real DB connection.
    """
    from src.main import app
    from src.providers.infrastructure.database import get_db_session

    async def _mock_db_session():
        """Minimal async generator that yields a no-op mock session."""
        yield MagicMock()

    # Override get_db_session so routes receive a mock db (not a real connection)
    app.dependency_overrides[get_db_session] = _mock_db_session

    route_module = "src.api.routes.conversation.conversation_routes"

    with (
        patch(f"{route_module}.get_conversation_repo", return_value=mock_conv_repo),
        patch(f"{route_module}.get_message_repo", return_value=mock_msg_repo),
    ):
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c

    app.dependency_overrides.clear()


# ── Tests ──────────────────────────────────────────────────────

def test_create_conversation_returns_201(client):
    response = client.post("/api/v1/conversations", json={"title": "New Chat"})
    assert response.status_code == 201
    assert response.json()["data"]["title"] == "Test Conversation"


def test_list_conversations_returns_200(client):
    response = client.get("/api/v1/conversations")
    assert response.status_code == 200
    body = response.json()
    assert "data" in body
    assert "pagination" in body


def test_get_conversation_returns_200(client):
    response = client.get(f"/api/v1/conversations/{CONV_ID}")
    assert response.status_code == 200
    assert response.json()["data"]["conversation_id"] == CONV_ID


def test_update_conversation_returns_200(client):
    response = client.patch(
        f"/api/v1/conversations/{CONV_ID}",
        json={"title": "Judul Baru"},
    )
    assert response.status_code == 200
    assert response.json()["success"] is True


def test_delete_conversation_returns_204(client):
    response = client.delete(f"/api/v1/conversations/{CONV_ID}")
    assert response.status_code == 204


def test_list_messages_returns_paginated(client):
    response = client.get(f"/api/v1/conversations/{CONV_ID}/messages")
    assert response.status_code == 200
    body = response.json()
    assert "pagination" in body
    assert len(body["data"]) == 1
    assert body["data"][0]["role"] == "user"


def test_create_conversation_default_title(client):
    """Empty title should fall back to 'New Conversation'."""
    response = client.post("/api/v1/conversations", json={})
    # Request is valid — title has a default value in the schema
    assert response.status_code == 201