"""
Not-found exception variants for all domain entities.

All exceptions here map to HTTP 404 in the API layer.
Catch NotFoundError to handle any resource-not-found case generically,
or catch a specific subclass for targeted handling.
"""

from __future__ import annotations

from src.core.exceptions.base import AppBaseError


class NotFoundError(AppBaseError):
    """Base class for all resource-not-found errors."""


class ConversationNotFoundError(NotFoundError):
    """
    Raised when a conversation does not exist or has been soft-deleted.

    Args:
        conversation_id: The ID that was looked up.
    """

    def __init__(self, conversation_id: str) -> None:
        super().__init__(
            message=f"Conversation '{conversation_id}' not found.",
            context={"conversation_id": conversation_id},
        )


class MessageNotFoundError(NotFoundError):
    """
    Raised when a message does not exist.

    Args:
        message_id: The ID that was looked up.
    """

    def __init__(self, message_id: str) -> None:
        super().__init__(
            message=f"Message '{message_id}' not found.",
            context={"message_id": message_id},
        )


class SystemPromptNotFoundError(NotFoundError):
    """
    Raised when a system prompt does not exist or has been soft-deleted.

    Args:
        system_prompt_id: The ID that was looked up.
    """

    def __init__(self, system_prompt_id: str) -> None:
        super().__init__(
            message=f"System prompt '{system_prompt_id}' not found.",
            context={"system_prompt_id": system_prompt_id},
        )


class ModelNotFoundError(NotFoundError):
    """
    Raised when an available model does not exist in the catalog.

    Args:
        model_id: The ID that was looked up.
    """

    def __init__(self, model_id: str) -> None:
        super().__init__(
            message=f"Model '{model_id}' not found in the catalog.",
            context={"model_id": model_id},
        )


class DocumentNotFoundError(NotFoundError):
    """
    Raised when a document record does not exist.

    Args:
        document_id: The ID that was looked up.
    """

    def __init__(self, document_id: str) -> None:
        super().__init__(
            message=f"Document '{document_id}' not found.",
            context={"document_id": document_id},
        )
