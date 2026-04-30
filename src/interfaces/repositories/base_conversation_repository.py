"""Abstract base for conversation repositories."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.entities.conversation.conversation import Conversation


class BaseConversationRepository(ABC):
    """Contract for all conversation persistence implementations."""

    @abstractmethod
    async def create(self, user_id: str | None, title: str) -> Conversation:
        """
        Create a new conversation record.

        Args:
            user_id: Owner UUID. Pass None for the default anonymous user.
            title: Initial conversation title.

        Returns:
            The newly created Conversation entity.

        Raises:
            PersistenceError: If the insert fails.
        """
        ...

    @abstractmethod
    async def get_by_id(self, conversation_id: str) -> Conversation | None:
        """
        Retrieve an active conversation by its ID.

        Args:
            conversation_id: UUID of the conversation.

        Returns:
            Conversation entity if found and active, None otherwise.

        Raises:
            PersistenceError: If the query fails unexpectedly.
        """
        ...

    @abstractmethod
    async def get_all_by_user(
        self,
        user_id: str | None,
        limit: int,
        offset: int,
    ) -> list[Conversation]:
        """
        List all active conversations for a user, newest first.

        Args:
            user_id: Owner UUID. Pass None to query by null user_id.
            limit: Maximum results to return.
            offset: Number of results to skip for pagination.

        Returns:
            List of Conversation entities, ordered by created_at DESC.

        Raises:
            PersistenceError: If the query fails.
        """
        ...

    @abstractmethod
    async def update_title(
        self,
        conversation_id: str,
        title: str,
    ) -> Conversation:
        """
        Update the title of an existing conversation.

        Args:
            conversation_id: UUID of the conversation to update.
            title: New title string.

        Returns:
            Updated Conversation entity.

        Raises:
            ConversationNotFoundError: If no active conversation has this ID.
            PersistenceError: If the update fails.
        """
        ...

    @abstractmethod
    async def soft_delete(self, conversation_id: str) -> None:
        """
        Soft-delete a conversation by setting is_active to False.

        Args:
            conversation_id: UUID of the conversation to archive.

        Raises:
            ConversationNotFoundError: If no active conversation has this ID.
            PersistenceError: If the update fails.
        """
        ...
