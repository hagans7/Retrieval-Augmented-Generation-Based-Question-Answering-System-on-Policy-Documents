"""
Unit tests for ProcessChatService.

All external dependencies are replaced with AsyncMock instances.
No database, network, or LLM calls are made in these tests.
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from src.services.process_chat.process_chat import ProcessChatService
from src.entities.chat_result.chat_result import ChatResult
from src.entities.conversation.conversation import Conversation
from src.entities.message.message import Message
from src.entities.available_model.available_model import AvailableModel
from src.entities.system_prompt.system_prompt import SystemPrompt
from src.core.exceptions.not_found import ConversationNotFoundError


# ── Helpers ──────────────────────────────────────────────────

def make_conversation(conversation_id: str = "conv-1") -> Conversation:
    return Conversation(
        conversation_id=conversation_id,
        title="Test Conversation",
        user_id="user-1",
        is_active=True,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )


def make_message(role: str = "assistant", content: str = "Test answer") -> Message:
    return Message(
        message_id="msg-new",
        conversation_id="conv-1",
        role=role,
        content=content,
        created_at=datetime.utcnow(),
    )


def make_model() -> AvailableModel:
    return AvailableModel(
        model_id="model-1",
        model_name="qwen/qwen3-6b-plus:free",
        provider="openrouter",
        display_name="Qwen 3",
        is_active=True,
        created_at=datetime.utcnow(),
    )


def make_prompt() -> SystemPrompt:
    return SystemPrompt(
        system_prompt_id="prompt-1",
        name="Default",
        content="Kamu adalah asisten hukum.",
        is_default=True,
        is_active=True,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )


def make_service(
    conversation=None,
    saved_message=None,
    final_answer="Test answer from agent",
) -> ProcessChatService:
    """Build a fully-mocked ProcessChatService."""
    conversation_repo = AsyncMock()
    conversation_repo.get_by_id.return_value = conversation or make_conversation()

    message_repo = AsyncMock()
    message_repo.save_user_message.return_value = make_message(role="user", content="Q")
    message_repo.save_assistant_message.return_value = saved_message or make_message()
    message_repo.get_history.return_value = []

    system_prompt_repo = AsyncMock()
    system_prompt_repo.get_by_id.return_value = make_prompt()
    system_prompt_repo.get_default_for_user.return_value = make_prompt()

    model_repo = AsyncMock()
    model_repo.get_by_id.return_value = make_model()
    model_repo.get_all_active.return_value = [make_model()]

    llm_client = AsyncMock()
    llm_client.generate.return_value = final_answer
    llm_client.generate_with_tool_call.return_value = {
        "query_type": "simple",
        "reasoning": "test",
        "requires_document_filter": False,
        "answer": final_answer,
        "citations": [],
        "confidence": "high",
    }

    vector_client = AsyncMock()
    vector_client.hybrid_search.return_value = []

    graph_client = AsyncMock()
    graph_client.multi_hop_query.return_value = []

    embedding_client = AsyncMock()
    embedding_client.embed_single.return_value = [0.1, 0.2, 0.3]

    reranker_client = AsyncMock()
    reranker_client.rerank.return_value = []

    cache_client = AsyncMock()
    cache_client.get.return_value = None

    return ProcessChatService(
        conversation_repo=conversation_repo,
        message_repo=message_repo,
        system_prompt_repo=system_prompt_repo,
        model_repo=model_repo,
        llm_client=llm_client,
        vector_client=vector_client,
        graph_client=graph_client,
        embedding_client=embedding_client,
        reranker_client=reranker_client,
        cache_client=cache_client,
    )


# ── Tests ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_execute_non_stream_returns_chat_result():
    """Non-stream mode should return a ChatResult entity."""
    svc = make_service()
    result = await svc.execute(
        conversation_id="conv-1",
        user_message="Apa itu Pasal 1338?",
        model_id=None,
        system_prompt_id=None,
        stream=False,
        correlation_id="test-cid",
    )
    assert isinstance(result, ChatResult)
    assert result.conversation_id == "conv-1"
    assert result.answer != ""


@pytest.mark.asyncio
async def test_execute_persists_user_message():
    """Service must save the user message before invoking the agent."""
    svc = make_service()
    await svc.execute(
        conversation_id="conv-1",
        user_message="Test question",
        model_id=None,
        system_prompt_id=None,
        stream=False,
    )
    svc._message_repo.save_user_message.assert_called_once()


@pytest.mark.asyncio
async def test_execute_persists_assistant_message():
    """Service must save the assistant response after agent completes."""
    svc = make_service()
    await svc.execute(
        conversation_id="conv-1",
        user_message="Test question",
        model_id=None,
        system_prompt_id=None,
        stream=False,
    )
    svc._message_repo.save_assistant_message.assert_called_once()


@pytest.mark.asyncio
async def test_execute_raises_when_conversation_not_found():
    """Should raise ConversationNotFoundError if conversation doesn't exist."""
    svc = make_service()
    svc._context_builder._conv_repo.get_by_id.return_value = None

    with pytest.raises(ConversationNotFoundError):
        await svc.execute(
            conversation_id="nonexistent",
            user_message="Hello",
            model_id=None,
            system_prompt_id=None,
            stream=False,
        )


@pytest.mark.asyncio
async def test_execute_stream_returns_async_iterator():
    """Stream mode should return an async iterable, not a ChatResult."""
    svc = make_service()
    result = await svc.execute(
        conversation_id="conv-1",
        user_message="Test question",
        model_id=None,
        system_prompt_id=None,
        stream=True,
    )
    # Result must be async-iterable (not a ChatResult)
    assert hasattr(result, "__aiter__") or hasattr(result, "__anext__")
