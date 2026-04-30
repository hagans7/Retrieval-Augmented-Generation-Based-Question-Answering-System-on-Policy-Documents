"""
SystemPrompt entity — domain object for user-defined system prompts.

These prompts are injected into the generator agent's context window
alongside the agent's internal prompt, providing persona/domain context.
Pure Python dataclass. Zero external dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class SystemPrompt:
    """
    Represents a user-defined system prompt stored in the database.

    This is NOT the internal agent prompt (those live in src/prompts/agents/).
    This is the user-facing persona/context injection, e.g.:
        "Jadilah asisten AI dalam aspek hukum perdata Indonesia."

    Fields:
        system_prompt_id: Unique UUID identifier.
        name: Descriptive name shown in the UI.
        content: The actual prompt text to inject into generation context.
        is_default: Whether this is the user's default prompt.
        is_active: Soft delete flag.
        user_id: Owner. Nullable while auth is deferred.
        created_at: UTC-aware creation timestamp.
        updated_at: UTC-aware last update timestamp.
    """

    system_prompt_id: str
    name: str
    content: str
    is_default: bool = False
    is_active: bool = True
    user_id: str | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def is_owned_by(self, user_id: str) -> bool:
        """
        Return True if this prompt belongs to the given user.

        While auth is deferred, the service layer bypasses this check.
        Once auth is enforced, this method is used in ownership validation.

        Args:
            user_id: The user ID to check ownership against.

        Returns:
            True if self.user_id matches user_id, or if user_id is None
            (anonymous/default user during auth-deferred phase).
        """
        if self.user_id is None:
            return True
        return self.user_id == user_id

    def truncate_content(self, max_chars: int = 100) -> str:
        """
        Return a truncated preview of the prompt content.

        Used for listing endpoints — never for LLM injection.

        Args:
            max_chars: Maximum characters to return.

        Returns:
            Content truncated to max_chars with ellipsis if needed.
        """
        if len(self.content) <= max_chars:
            return self.content
        return self.content[:max_chars] + "..."
