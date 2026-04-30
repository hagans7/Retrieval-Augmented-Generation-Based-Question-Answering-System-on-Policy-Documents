"""
Context builder — resolves all runtime context needed by the chat agent.

Receives an already-resolved conversation_id (never None here — ProcessChatService
handles the auto-create before calling resolve_all).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.core.config.settings import settings
from src.core.exceptions.domain import ModelNotActiveError
from src.core.exceptions.not_found import ConversationNotFoundError, ModelNotFoundError
from src.core.logging.logger import get_logger
from src.interfaces.repositories.base_available_model_repository import BaseAvailableModelRepository
from src.interfaces.repositories.base_conversation_repository import BaseConversationRepository
from src.interfaces.repositories.base_message_repository import BaseMessageRepository
from src.interfaces.repositories.base_system_prompt_repository import BaseSystemPromptRepository

logger = get_logger(__name__)


@dataclass
class ChatContext:
    """All resolved context needed to run the chat agent."""
    conversation_id: str
    user_message: str
    model_name: str
    model_id: str | None
    user_system_prompt: str
    system_prompt_id: str | None
    chat_history: list[dict] = field(default_factory=list)
    user_id: str | None = None


class ContextBuilder:
    def __init__(
        self,
        conversation_repo: BaseConversationRepository,
        message_repo: BaseMessageRepository,
        system_prompt_repo: BaseSystemPromptRepository,
        model_repo: BaseAvailableModelRepository,
    ) -> None:
        self._conv_repo = conversation_repo
        self._msg_repo = message_repo
        self._prompt_repo = system_prompt_repo
        self._model_repo = model_repo

    async def resolve_all(
        self,
        conversation_id: str,
        user_message: str,
        model_id: str | None,
        system_prompt_id: str | None,
    ) -> ChatContext:
        """
        Resolve all context for a chat request.

        conversation_id must be a valid UUID string at this point —
        ProcessChatService auto-creates if needed before calling here.

        Raises:
            ConversationNotFoundError: If conversation does not exist.
            ModelNotFoundError: If specified model does not exist.
            ModelNotActiveError: If specified model is inactive.
        """
        # ── Resolve conversation ─────────────────────────────
        conversation = await self._conv_repo.get_by_id(conversation_id)
        if conversation is None:
            raise ConversationNotFoundError(conversation_id)

        # ── Resolve model ────────────────────────────────────
        resolved_model_name = settings.LLM_DEFAULT_MODEL
        resolved_model_id: str | None = None
        if model_id:
            model = await self._model_repo.get_by_id(model_id)
            if model is None:
                raise ModelNotFoundError(model_id)
            if not model.is_available():
                raise ModelNotActiveError(model_id)
            resolved_model_name = model.model_name
            resolved_model_id = model.model_id

        # ── Resolve system prompt ─────────────────────────────
        resolved_prompt_content = ""
        resolved_prompt_id: str | None = None
        if system_prompt_id:
            prompt = await self._prompt_repo.get_by_id(system_prompt_id)
            if prompt:
                resolved_prompt_content = prompt.content
                resolved_prompt_id = prompt.system_prompt_id
        else:
            default_prompt = await self._prompt_repo.get_default_for_user(conversation.user_id)
            if default_prompt:
                resolved_prompt_content = default_prompt.content
                resolved_prompt_id = default_prompt.system_prompt_id

        # ── Load chat history ─────────────────────────────────
        history_messages = await self._msg_repo.get_history(
            conversation_id=conversation_id,
            limit=settings.CHAT_HISTORY_MAX_TURNS * 2,
        )
        chat_history = [m.to_llm_format() for m in history_messages]

        return ChatContext(
            conversation_id=conversation_id,
            user_message=user_message,
            model_name=resolved_model_name,
            model_id=resolved_model_id,
            user_system_prompt=resolved_prompt_content,
            system_prompt_id=resolved_prompt_id,
            chat_history=chat_history,
            user_id=conversation.user_id,
        )