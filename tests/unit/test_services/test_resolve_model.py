"""Unit tests for ResolveModelService."""
import pytest
from unittest.mock import AsyncMock
from src.services.resolve_model.resolve_model import ResolveModelService
from src.entities.available_model.available_model import AvailableModel
from src.core.exceptions.not_found import ModelNotFoundError
from src.core.exceptions.domain import ModelNotActiveError
from datetime import datetime


def make_model(is_active=True) -> AvailableModel:
    return AvailableModel(
        model_id="model-1", model_name="qwen/qwen3", provider="openrouter",
        display_name="Qwen 3", is_active=is_active, created_at=datetime.utcnow(),
    )


@pytest.mark.asyncio
async def test_execute_resolve_returns_model():
    repo = AsyncMock()
    repo.get_by_id.return_value = make_model()
    svc = ResolveModelService(model_repo=repo)
    result = await svc.execute_resolve("model-1")
    assert result.model_id == "model-1"


@pytest.mark.asyncio
async def test_execute_resolve_raises_not_found():
    repo = AsyncMock()
    repo.get_by_id.return_value = None
    svc = ResolveModelService(model_repo=repo)
    with pytest.raises(ModelNotFoundError):
        await svc.execute_resolve("nonexistent")


@pytest.mark.asyncio
async def test_execute_resolve_raises_not_active():
    repo = AsyncMock()
    repo.get_by_id.return_value = make_model(is_active=False)
    svc = ResolveModelService(model_repo=repo)
    with pytest.raises(ModelNotActiveError):
        await svc.execute_resolve("model-1")
