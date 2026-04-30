"""
ManagePromptService — CRUD operations for user-defined system prompts.

All operations are scoped to a single service class because they belong
to the same domain: user prompt management. Each public method handles
one operation. Business rules (e.g. cannot delete the only default prompt)
are enforced here, not in the repository or the route handler.
"""
from __future__ import annotations

from src.core.exceptions.domain import ValidationError
from src.core.exceptions.not_found import SystemPromptNotFoundError
from src.core.logging.logger import get_logger
from src.entities.system_prompt.system_prompt import SystemPrompt
from src.interfaces.repositories.base_system_prompt_repository import BaseSystemPromptRepository


class ManagePromptService:
    """
    Service for all system prompt lifecycle operations.

    Args:
        system_prompt_repo: Repository for system_prompts table.
    """

    def __init__(self, system_prompt_repo: BaseSystemPromptRepository) -> None:
        self._repo = system_prompt_repo
        self._logger = get_logger(__name__)

    async def execute_create(
        self,
        user_id: str | None,
        name: str,
        content: str,
        is_default: bool = False,
    ) -> SystemPrompt:
        """
        Create a new system prompt.

        Raises:
            ValidationError: If name or content is empty.
        """
        if not name.strip():
            raise ValidationError(field="name", reason="Name cannot be empty.")
        if not content.strip():
            raise ValidationError(field="content", reason="Content cannot be empty.")
        return await self._repo.create(user_id=user_id, name=name.strip(), content=content.strip(), is_default=is_default)

    async def execute_list(self, user_id: str | None) -> list[SystemPrompt]:
        """Return all active prompts for the given user."""
        return await self._repo.get_all_by_user(user_id)

    async def execute_get(self, system_prompt_id: str, user_id: str | None) -> SystemPrompt:
        """
        Return a specific prompt.

        Raises:
            SystemPromptNotFoundError: If not found or inactive.
        """
        prompt = await self._repo.get_by_id(system_prompt_id)
        if prompt is None:
            raise SystemPromptNotFoundError(system_prompt_id)
        return prompt

    async def execute_update(
        self,
        system_prompt_id: str,
        user_id: str | None,
        name: str | None,
        content: str | None,
    ) -> SystemPrompt:
        """
        Partially update name and/or content.

        Raises:
            SystemPromptNotFoundError: If not found.
            ValidationError: If provided values are empty strings.
        """
        if name is not None and not name.strip():
            raise ValidationError(field="name", reason="Name cannot be empty.")
        if content is not None and not content.strip():
            raise ValidationError(field="content", reason="Content cannot be empty.")
        return await self._repo.update(system_prompt_id, name, content)

    async def execute_set_default(self, system_prompt_id: str, user_id: str | None) -> None:
        """
        Set a prompt as the default, unsetting all others atomically.

        Raises:
            SystemPromptNotFoundError: If the prompt does not exist.
        """
        prompt = await self._repo.get_by_id(system_prompt_id)
        if prompt is None:
            raise SystemPromptNotFoundError(system_prompt_id)
        await self._repo.set_as_default(system_prompt_id, user_id)

    async def execute_delete(self, system_prompt_id: str, user_id: str | None) -> None:
        """
        Soft-delete a prompt.

        Cannot delete the default prompt — user must set another as default first.

        Raises:
            SystemPromptNotFoundError: If not found.
            ValidationError: If attempting to delete the default prompt.
        """
        prompt = await self._repo.get_by_id(system_prompt_id)
        if prompt is None:
            raise SystemPromptNotFoundError(system_prompt_id)
        if prompt.is_default:
            raise ValidationError(
                field="system_prompt_id",
                reason="Cannot delete the default prompt. Set another prompt as default first.",
            )
        await self._repo.soft_delete(system_prompt_id)
