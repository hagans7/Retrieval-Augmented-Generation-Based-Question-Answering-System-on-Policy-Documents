"""
Integration tests for /api/v1/prompts endpoints.
Uses FastAPI TestClient with dependency overrides.
"""
from __future__ import annotations
from datetime import datetime
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock
import pytest

from src.entities.system_prompt.system_prompt import SystemPrompt

FAKE_PROMPT = SystemPrompt(
    system_prompt_id="p-1",
    name="Test Prompt",
    content="Kamu adalah asisten hukum.",
    is_default=False,
    is_active=True,
    created_at=datetime.utcnow(),
    updated_at=datetime.utcnow(),
)


@pytest.fixture
def client():
    from src.main import app
    from src.providers import get_manage_prompt_service

    mock_service = AsyncMock()
    mock_service.execute_create.return_value = FAKE_PROMPT
    mock_service.execute_list.return_value = [FAKE_PROMPT]
    mock_service.execute_get.return_value = FAKE_PROMPT
    mock_service.execute_update.return_value = FAKE_PROMPT
    mock_service.execute_set_default.return_value = None
    mock_service.execute_delete.return_value = None

    app.dependency_overrides[get_manage_prompt_service] = lambda: mock_service
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    app.dependency_overrides.clear()


def test_create_prompt_returns_201(client):
    response = client.post(
        "/api/v1/prompts",
        json={"name": "New Prompt", "content": "Be helpful."},
    )
    assert response.status_code == 201
    assert response.json()["data"]["system_prompt_id"] == "p-1"


def test_list_prompts_returns_200(client):
    response = client.get("/api/v1/prompts")
    assert response.status_code == 200
    assert len(response.json()["data"]) == 1


def test_get_prompt_returns_200(client):
    response = client.get("/api/v1/prompts/p-1")
    assert response.status_code == 200
    assert response.json()["data"]["name"] == "Test Prompt"


def test_update_prompt_returns_200(client):
    response = client.patch(
        "/api/v1/prompts/p-1",
        json={"name": "Updated Name"},
    )
    assert response.status_code == 200


def test_delete_prompt_returns_204(client):
    response = client.delete("/api/v1/prompts/p-1")
    assert response.status_code == 204


def test_create_prompt_rejects_empty_name(client):
    """Empty name should return 422 from Pydantic validator."""
    response = client.post(
        "/api/v1/prompts",
        json={"name": "  ", "content": "Content"},
    )
    assert response.status_code == 422
