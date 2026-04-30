"""
Repository provider functions.

Each function returns a repository interface type, never the concrete class.
Session is injected via FastAPI's Depends() mechanism.
"""
from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from src.interfaces.repositories.base_available_model_repository import BaseAvailableModelRepository
from src.interfaces.repositories.base_conversation_repository import BaseConversationRepository
from src.interfaces.repositories.base_document_repository import BaseDocumentRepository
from src.interfaces.repositories.base_message_repository import BaseMessageRepository
from src.interfaces.repositories.base_system_prompt_repository import BaseSystemPromptRepository


def get_conversation_repo(db: AsyncSession) -> BaseConversationRepository:
    from src.repositories.conversation.conversation_repository import ConversationRepository
    return ConversationRepository(db)


def get_message_repo(db: AsyncSession) -> BaseMessageRepository:
    from src.repositories.message.message_repository import MessageRepository
    return MessageRepository(db)


def get_system_prompt_repo(db: AsyncSession) -> BaseSystemPromptRepository:
    from src.repositories.system_prompt.system_prompt_repository import SystemPromptRepository
    return SystemPromptRepository(db)


def get_available_model_repo(db: AsyncSession) -> BaseAvailableModelRepository:
    from src.repositories.available_model.available_model_repository import AvailableModelRepository
    return AvailableModelRepository(db)


def get_document_repo(db: AsyncSession) -> BaseDocumentRepository:
    from src.repositories.document.document_repository import DocumentRepository
    return DocumentRepository(db)
