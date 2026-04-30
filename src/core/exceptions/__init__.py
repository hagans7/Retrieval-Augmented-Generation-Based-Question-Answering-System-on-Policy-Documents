"""
Exception package — re-exports all application exceptions.

Import exceptions from here for clean, single-source imports:
    from src.core.exceptions import ConversationNotFoundError, LLMInferenceError
"""

from src.core.exceptions.base import AppBaseError
from src.core.exceptions.not_found import (
    ConversationNotFoundError,
    DocumentNotFoundError,
    MessageNotFoundError,
    ModelNotFoundError,
    NotFoundError,
    SystemPromptNotFoundError,
)
from src.core.exceptions.infrastructure import (
    CacheError,
    EmbeddingError,
    GraphQueryError,
    LLMInferenceError,
    OCRError,
    RerankerError,
    StorageError,
    VectorSearchError,
)
from src.core.exceptions.domain import (
    AgentExecutionError,
    AuditFailedError,
    DuplicateResourceError,
    ModelNotActiveError,
    PersistenceError,
    PlanningError,
    ValidationError,
)

__all__ = [
    "AppBaseError",
    "NotFoundError",
    "ConversationNotFoundError",
    "MessageNotFoundError",
    "SystemPromptNotFoundError",
    "ModelNotFoundError",
    "DocumentNotFoundError",
    "LLMInferenceError",
    "VectorSearchError",
    "GraphQueryError",
    "EmbeddingError",
    "RerankerError",
    "StorageError",
    "OCRError",
    "CacheError",
    "PersistenceError",
    "DuplicateResourceError",
    "AgentExecutionError",
    "PlanningError",
    "AuditFailedError",
    "ValidationError",
    "ModelNotActiveError",
]
