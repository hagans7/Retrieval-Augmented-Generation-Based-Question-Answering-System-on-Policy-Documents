"""
Integration tests for GET /api/v1/models endpoints.

Uses FastAPI TestClient with dependency overrides so no DB is needed.
"""
from __future__ import annotations
from datetime import datetime
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, patch
import pytest

from src.entities.available_model.available_model import AvailableModel

FAKE_MODEL = AvailableModel(
    model_id="m-1",
    model_name="qwen/qwen3-6b-plus:free",
    provider="openrouter",
    display_name="Qwen 3 6B (Free)",
    is_active=True,
    created_at=datetime.utcnow(),
)


@pytest.fixture
def client():
    """TestClient with mocked service dependencies."""
    from src.main import app
    from src.providers import get_resolve_model_service

    mock_service = AsyncMock()
    mock_service.execute_get_all.return_value = [FAKE_MODEL]
    mock_service.execute_resolve.return_value = FAKE_MODEL

    app.dependency_overrides[get_resolve_model_service] = lambda: mock_service
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c
    app.dependency_overrides.clear()


def test_list_models_returns_200(client):
    response = client.get("/api/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert len(data["data"]) == 1
    assert data["data"][0]["model_name"] == "qwen/qwen3-6b-plus:free"


def test_get_model_by_id_returns_200(client):
    response = client.get("/api/v1/models/m-1")
    assert response.status_code == 200
    assert response.json()["data"]["model_id"] == "m-1"


def test_list_models_response_has_required_fields(client):
    response = client.get("/api/v1/models")
    model = response.json()["data"][0]
    for field in ("model_id", "model_name", "provider", "display_name", "is_active"):
        assert field in model, f"Missing field: {field}"
