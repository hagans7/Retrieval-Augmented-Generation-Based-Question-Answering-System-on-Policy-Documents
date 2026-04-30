"""Abstract base for message repositories."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.entities.message.message import Message


class BaseMessageRepository(ABC):
    """Contract for all message persistence implementations."""

    @abstractmethod
    async def save_user_message(
        self,
        conversation_id: str,
        content: str,
        system_prompt_id: str | None,
        model_id: str | None,
    ) -> Message:
        """
        Persist a user-role message.

        Args:
            conversation_id: Parent conversation UUID.
            content: The user's message text.
            system_prompt_id: Active system prompt ID (nullable).
            model_id: Model ID (nullable — not used for user messages).

        Returns:
            The persisted Message entity.

        Raises:
            PersistenceError: If the insert fails.
        """
        ...

    @abstractmethod
    async def save_assistant_message(
        self,
        conversation_id: str,
        content: str,
        system_prompt_id: str | None,
        model_id: str | None,
        token_count: int | None,
    ) -> Message:
        """
        Persist an assistant-role message (final generated answer).

        Args:
            conversation_id: Parent conversation UUID.
            content: The assistant's response text.
            system_prompt_id: System prompt active during generation.
            model_id: Model used for generation.
            token_count: Estimated tokens used, if available.

        Returns:
            The persisted Message entity.

        Raises:
            PersistenceError: If the insert fails.
        """
        ...

    @abstractmethod
    async def get_history(
        self,
        conversation_id: str,
        limit: int,
    ) -> list[Message]:
        """
        Retrieve the N most recent messages for a conversation, oldest-first.

        The returned list is ready for direct injection into LLM context
        after calling message.to_llm_format() on each element.

        Args:
            conversation_id: UUID of the target conversation.
            limit: Maximum number of messages to return.

        Returns:
            List of Message entities ordered by created_at ASC.
            Returns empty list if no messages exist — never raises NotFoundError.

        Raises:
            PersistenceError: If the query fails unexpectedly.
        """
        ...
