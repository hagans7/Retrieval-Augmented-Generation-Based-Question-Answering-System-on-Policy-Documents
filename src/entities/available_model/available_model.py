"""
AvailableModel entity — domain object for an LLM model in the catalog.

Users select from this catalog per-request. The API key and default model
remain in environment variables — users only choose from pre-configured options.
Pure Python dataclass. Zero external dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class AvailableModel:
    """
    Represents a model available for selection in the chat interface.

    The model_name is the technical identifier sent to the LLM provider API.
    The display_name is what users see in the UI.

    Fields:
        model_id: Unique UUID identifier.
        model_name: Technical name for API calls (e.g. "qwen/qwen3-6b-plus:free").
        provider: Provider name (e.g. "openrouter").
        display_name: Human-readable name for the UI.
        description: Optional short description of capabilities.
        context_window: Maximum context window in tokens, if known.
        is_active: Whether this model is available for selection.
        created_at: UTC-aware creation timestamp.
    """

    model_id: str
    model_name: str
    provider: str
    display_name: str
    is_active: bool = True
    description: str | None = None
    context_window: int | None = None
    created_at: datetime = field(default_factory=datetime.utcnow)

    def is_available(self) -> bool:
        """
        Return True if this model can be selected by a user.

        Alias for is_active — more expressive in service layer contexts.

        Returns:
            True when is_active is True.
        """
        return self.is_active
