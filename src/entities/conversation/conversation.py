"""
Conversation entity — domain object for a chat session.

Pure Python dataclass. Zero external dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from src.entities.message.message import Message


@dataclass
class Conversation:
    """
    Represents a single chat session between a user and the system.

    The messages list is populated by the repository when full history
    is needed. It defaults to empty — not all operations require messages.

    Fields:
        conversation_id: Unique UUID identifier.
        user_id: Owner of the conversation. Nullable while auth is deferred.
        title: Display title, can be updated by the user.
        is_active: Soft delete flag. False means the conversation is archived.
        created_at: UTC-aware timestamp of creation.
        updated_at: UTC-aware timestamp of last update.
        messages: Ordered list of messages (oldest first). Default empty.
    """

    conversation_id: str
    title: str
    is_active: bool = True
    user_id: str | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    messages: list[Message] = field(default_factory=list)

    def is_empty(self) -> bool:
        """
        Return True if no messages have been exchanged yet.

        Returns:
            True when the messages list is empty.
        """
        return len(self.messages) == 0

    def get_recent_messages(self, limit: int) -> list[Message]:
        """
        Return the N most recent messages, ordered oldest-first.

        Used to build the LLM context window. Caller converts each
        message to LLM format via message.to_llm_format().

        Args:
            limit: Maximum number of messages to return.

        Returns:
            Up to `limit` most recent messages, oldest first.
            Returns all messages if len(messages) <= limit.
        """
        if limit <= 0:
            return []
        return self.messages[-limit:]

    def get_last_assistant_message(self) -> Message | None:
        """
        Return the most recent assistant message, or None if none exists.

        Useful for display purposes and audit checks.

        Returns:
            The last assistant Message, or None.
        """
        for message in reversed(self.messages):
            if message.is_from_assistant():
                return message
        return None
