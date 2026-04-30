"""Unit tests for ManagePromptService."""
import pytest
from unittest.mock import AsyncMock
from datetime import datetime
from src.services.manage_prompt.manage_prompt import ManagePromptService
from src.entities.system_prompt.system_prompt import SystemPrompt
from src.core.exceptions.domain import ValidationError
from src.core.exceptions.not_found import SystemPromptNotFoundError


def make_prompt(is_default=False) -> SystemPrompt:
    return SystemPrompt(
        system_prompt_id="p-1", name="Test", content="Test content",
        is_default=is_default, is_active=True,
        created_at=datetime.utcnow(), updated_at=datetime.utcnow(),
    )


@pytest.mark.asyncio
async def test_create_raises_on_empty_name():
    repo = AsyncMock()
    svc = ManagePromptService(system_prompt_repo=repo)
    with pytest.raises(ValidationError):
        await svc.execute_create(user_id=None, name="  ", content="content")


@pytest.mark.asyncio
async def test_delete_raises_on_default_prompt():
    repo = AsyncMock()
    repo.get_by_id.return_value = make_prompt(is_default=True)
    svc = ManagePromptService(system_prompt_repo=repo)
    with pytest.raises(ValidationError):
        await svc.execute_delete("p-1", user_id=None)


@pytest.mark.asyncio
async def test_delete_raises_not_found():
    repo = AsyncMock()
    repo.get_by_id.return_value = None
    svc = ManagePromptService(system_prompt_repo=repo)
    with pytest.raises(SystemPromptNotFoundError):
        await svc.execute_delete("p-1", user_id=None)
