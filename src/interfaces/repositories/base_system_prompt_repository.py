"""Abstract base for system prompt repositories."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.entities.system_prompt.system_prompt import SystemPrompt


class BaseSystemPromptRepository(ABC):
    """Contract for all system prompt persistence implementations."""

    @abstractmethod
    async def create(
        self,
        user_id: str | None,
        name: str,
        content: str,
        is_default: bool,
    ) -> SystemPrompt:
        """
        Create a new system prompt.

        If is_default is True, atomically unsets is_default on all
        other prompts for this user in the same transaction.

        Args:
            user_id: Owner UUID. Nullable.
            name: Descriptive name.
            content: Prompt text content.
            is_default: Whether to set as the user's default prompt.

        Returns:
            The newly created SystemPrompt entity.

        Raises:
            PersistenceError: If the insert fails.
        """
        ...

    @abstractmethod
    async def get_by_id(self, system_prompt_id: str) -> SystemPrompt | None:
        """Return active prompt by ID, or None."""
        ...

    @abstractmethod
    async def get_all_by_user(self, user_id: str | None) -> list[SystemPrompt]:
        """Return all active prompts for a user."""
        ...

    @abstractmethod
    async def get_default_for_user(self, user_id: str | None) -> SystemPrompt | None:
        """Return the default prompt for a user, or None if none set."""
        ...

    @abstractmethod
    async def update(
        self,
        system_prompt_id: str,
        name: str | None,
        content: str | None,
    ) -> SystemPrompt:
        """
        Partially update a system prompt's name and/or content.

        Raises:
            SystemPromptNotFoundError: If the prompt does not exist.
            PersistenceError: If the update fails.
        """
        ...

    @abstractmethod
    async def set_as_default(
        self,
        system_prompt_id: str,
        user_id: str | None,
    ) -> None:
        """
        Atomically set one prompt as default, unset all others for the user.

        Raises:
            SystemPromptNotFoundError: If the prompt does not exist.
            PersistenceError: If the update fails.
        """
        ...

    @abstractmethod
    async def soft_delete(self, system_prompt_id: str) -> None:
        """
        Soft-delete a prompt by setting is_active to False.

        Raises:
            SystemPromptNotFoundError: If the prompt does not exist.
            PersistenceError: If the update fails.
        """
        ...
