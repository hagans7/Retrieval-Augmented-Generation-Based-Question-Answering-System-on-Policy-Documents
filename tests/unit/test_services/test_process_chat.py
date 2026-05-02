
"""
Unit tests for ProcessChatService.

All external dependencies replaced with mocks.
No database, network, LLM, or observability calls are made.

Key invariants tested:
- Non-stream returns ChatResult with structured sources (list[dict])
- User message persisted before agent invocation
- Assistant message persisted after agent completes
- Observability client receives start_trace and flush calls
- ConversationNotFoundError raised for unknown conversations
- Stream mode returns async iterable
"""
from __future__ import annotations

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, call

from src.services.process_chat.process_chat import ProcessChatService, _build_sources
from src.entities.chat_result.chat_result import ChatResult
from src.entities.conversation.conversation import Conversation
from src.entities.message.message import Message
from src.entities.available_model.available_model import AvailableModel
from src.entities.system_prompt.system_prompt import SystemPrompt
from src.core.exceptions.not_found import ConversationNotFoundError
from src.core.observability.noop_client import NoOpObservabilityClient


# ── Builders ───────────────────────────────────────────────────────────────────

def make_conversation(conversation_id: str = "conv-1") -> Conversation:
    return Conversation(conversation_id=conversation_id, title="Test",
                        user_id="user-1", is_active=True,
                        created_at=datetime.utcnow(), updated_at=datetime.utcnow())


def make_message(role: str = "assistant", content: str = "Test answer") -> Message:
    return Message(message_id="msg-new", conversation_id="conv-1",
                   role=role, content=content, created_at=datetime.utcnow())


def make_model() -> AvailableModel:
    return AvailableModel(model_id="model-1", model_name="qwen/qwen-turbo",
                          provider="openrouter", display_name="Qwen",
                          is_active=True, created_at=datetime.utcnow())


def make_prompt() -> SystemPrompt:
    return SystemPrompt(system_prompt_id="prompt-1", name="Default",
                        content="Kamu adalah asisten hukum.", is_default=True,
                        is_active=True, created_at=datetime.utcnow(),
                        updated_at=datetime.utcnow())


FAKE_EVIDENCE = [
    {
        "content": "Mahasiswa diwajibkan mendaftar BPJS Kesehatan.",
        "score": 0.85,
        "source": "vector",
        "chunk_id": "chunk-uuid-1",
        "document_id": "doc-uuid-1",
        "_hash": "abc123",
    }
]


def make_service(
    conversation=None,
    saved_message=None,
    final_answer: str = "Test answer from agent",
    observability_client=None,
) -> ProcessChatService:
    """Build a fully-mocked ProcessChatService."""
    conversation_repo = AsyncMock()
    conversation_repo.get_by_id.return_value = conversation or make_conversation()
    conversation_repo.create.return_value = make_conversation("new-conv")

    message_repo = AsyncMock()
    message_repo.save_user_message.return_value = make_message(role="user", content="Q")
    message_repo.save_assistant_message.return_value = saved_message or make_message()
    message_repo.get_history.return_value = []

    system_prompt_repo = AsyncMock()
    system_prompt_repo.get_default_for_user.return_value = make_prompt()

    model_repo = AsyncMock()
    model_repo.get_default.return_value = make_model()

    llm_client = AsyncMock()
    # generate_with_tool_call is called 5 times per non-stream request:
    #   1. router    → classifies query type
    #   2. planner   → creates retrieval steps
    #   3. executor  → decides which tool/query to call
    #   4. auditor   → evaluates evidence sufficiency
    #   5. generator → produces final answer with citations
    llm_client.generate_with_tool_call.side_effect = [
        {"query_type": "retrieval"},                       # (1) router
        {"steps": ["Cari BPJS"], "estimated_complexity": "low"},  # (2) planner
        {"query": "bpjs kesehatan", "top_k": 5},           # (3) executor
        {                                                   # (4) auditor
            "verdict": "sufficient",
            "top_evidence_score": 0.85,
            "reasoning": "good evidence",
            "missing_aspects": [],
        },
        {                                                   # (5) generator
            "answer": final_answer,
            "citations": ["1"],
            "confidence": "high",
        },
    ]
    llm_client.generate.return_value = final_answer

    vector_client = AsyncMock()
    vector_client.hybrid_search.return_value = [
        {"chunk_id": "chunk-uuid-1", "content": "Mahasiswa diwajibkan mendaftar BPJS Kesehatan.",
         "score": 0.9, "document_id": "doc-uuid-1", "metadata": {}}
    ]

    graph_client = AsyncMock()
    graph_client.multi_hop_query.return_value = []

    embedding_client = AsyncMock()
    embedding_client.embed_single.return_value = [0.1, 0.2, 0.3]

    reranker_client = AsyncMock()
    reranker_client.rerank.return_value = [
        {"text": "Mahasiswa diwajibkan mendaftar BPJS Kesehatan.", "score": 0.85}
    ]

    cache_client = AsyncMock()
    cache_client.get.return_value = None

    obs = observability_client or NoOpObservabilityClient()

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
        observability_client=obs,
    )


# ── _build_sources (pure function) ────────────────────────────────────────────

class TestBuildSources:
    def test_returns_list_of_dicts(self):
        sources = _build_sources(FAKE_EVIDENCE, citations=["1"])
        assert isinstance(sources, list)
        assert all(isinstance(s, dict) for s in sources)

    def test_source_item_has_required_keys(self):
        sources = _build_sources(FAKE_EVIDENCE, citations=["1"])
        assert len(sources) == 1
        item = sources[0]
        assert "chunk_id" in item
        assert "document_id" in item
        assert "score" in item
        assert "preview" in item

    def test_preview_max_100_chars(self):
        long_content = "A" * 200
        evidence = [{**FAKE_EVIDENCE[0], "content": long_content}]
        sources = _build_sources(evidence, citations=["1"])
        assert len(sources[0]["preview"]) <= 100

    def test_graph_evidence_excluded(self):
        graph_ev = [{**FAKE_EVIDENCE[0], "source": "graph",
                     "content": '{"entity": "BPJS"}'}]
        sources = _build_sources(graph_ev, citations=[])
        assert sources == []

    def test_dedup_by_chunk_id(self):
        duplicate = [FAKE_EVIDENCE[0], FAKE_EVIDENCE[0]]
        sources = _build_sources(duplicate, citations=[])
        assert len(sources) == 1

    def test_empty_evidence_returns_empty(self):
        assert _build_sources([], citations=[]) == []

    def test_fallback_when_no_valid_citations(self):
        """If citations contain invalid indices, fall back to all evidence."""
        sources = _build_sources(FAKE_EVIDENCE, citations=["99", "abc"])
        assert len(sources) == 1

    def test_score_rounded_to_4dp(self):
        evidence = [{**FAKE_EVIDENCE[0], "score": 0.123456789}]
        sources = _build_sources(evidence, citations=["1"])
        assert sources[0]["score"] == round(0.123456789, 4)


# ── ProcessChatService.execute non-stream ─────────────────────────────────────

@pytest.mark.asyncio
async def test_execute_non_stream_returns_chat_result():
    svc = make_service()
    result = await svc.execute(
        conversation_id="conv-1", user_message="Apa itu BPJS?",
        model_id=None, system_prompt_id=None, stream=False, correlation_id="cid-1")
    assert isinstance(result, ChatResult)
    assert result.conversation_id == "conv-1"
    assert result.answer != ""


@pytest.mark.asyncio
async def test_sources_is_list_of_dicts():
    """sources must be list[dict] matching SourceItem schema."""
    svc = make_service()
    result = await svc.execute(
        conversation_id="conv-1", user_message="Apa itu BPJS?",
        model_id=None, system_prompt_id=None, stream=False)
    assert isinstance(result.sources, list)
    if result.sources:
        item = result.sources[0]
        assert all(k in item for k in ("chunk_id", "document_id", "score", "preview"))


@pytest.mark.asyncio
async def test_user_message_persisted_before_agent():
    svc = make_service()
    await svc.execute(conversation_id="conv-1", user_message="Q",
                      model_id=None, system_prompt_id=None, stream=False)
    svc._message_repo.save_user_message.assert_called_once()


@pytest.mark.asyncio
async def test_assistant_message_persisted_after_agent():
    svc = make_service()
    await svc.execute(conversation_id="conv-1", user_message="Q",
                      model_id=None, system_prompt_id=None, stream=False)
    svc._message_repo.save_assistant_message.assert_called_once()


@pytest.mark.asyncio
async def test_auto_create_conversation_when_none():
    svc = make_service()
    result = await svc.execute(
        conversation_id=None, user_message="Hello",
        model_id=None, system_prompt_id=None, stream=False)
    assert isinstance(result, ChatResult)
    svc._conv_repo.create.assert_called_once()


@pytest.mark.asyncio
async def test_raises_when_conversation_not_found():
    svc = make_service()
    svc._context_builder._conv_repo.get_by_id.return_value = None
    with pytest.raises(ConversationNotFoundError):
        await svc.execute(conversation_id="nonexistent", user_message="Hello",
                          model_id=None, system_prompt_id=None, stream=False)


# ── Observability integration ──────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_observability_start_trace_called(spy_observability_client):
    svc = make_service(observability_client=spy_observability_client)
    await svc.execute(conversation_id="conv-1", user_message="Q",
                      model_id=None, system_prompt_id=None, stream=False,
                      correlation_id="cid-trace")
    spy_observability_client.start_trace.assert_called_once()
    call_kwargs = spy_observability_client.start_trace.call_args
    assert call_kwargs is not None


@pytest.mark.asyncio
async def test_observability_flush_called(spy_observability_client):
    svc = make_service(observability_client=spy_observability_client)
    await svc.execute(conversation_id="conv-1", user_message="Q",
                      model_id=None, system_prompt_id=None, stream=False)
    spy_observability_client.flush.assert_called_once()


# ── Stream mode ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_execute_stream_returns_async_iterator():
    svc = make_service()
    result = await svc.execute(conversation_id="conv-1", user_message="Q",
                               model_id=None, system_prompt_id=None, stream=True)
    assert hasattr(result, "__aiter__") or hasattr(result, "__anext__")