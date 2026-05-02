"""
Shared pytest fixtures for all test categories.

Provides:
- async_session: In-memory SQLite async session for repository tests
- mock_*: AsyncMock/MagicMock for all infrastructure clients
- Consistent UUIDs for entity creation across test files

Fixture naming contract:
    mock_<thing>_client  → returns AsyncMock matching the Base<Thing>Client interface
    All return types match the interface contract, not the concrete class.
"""
from __future__ import annotations

from src.db_models import register_all_models
register_all_models()  # must run before any ORM usage

import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock

# ── Shared fake IDs ────────────────────────────────────────────────────────────
FAKE_CONVERSATION_ID = str(uuid.uuid4())
FAKE_MESSAGE_ID      = str(uuid.uuid4())
FAKE_PROMPT_ID       = str(uuid.uuid4())
FAKE_MODEL_ID        = str(uuid.uuid4())
FAKE_DOCUMENT_ID     = str(uuid.uuid4())
FAKE_CHUNK_ID        = str(uuid.uuid4())
FAKE_USER_ID         = str(uuid.uuid4())


# ── Infrastructure client mocks ────────────────────────────────────────────────

@pytest.fixture
def mock_llm_client():
    """BaseLLMClient mock — returns structured tool-call results."""
    client = AsyncMock()
    client.generate.return_value = "Mocked LLM response"
    client.generate_with_tool_call.return_value = {
        "query_type": "retrieval",
        "reasoning": "test",
        "requires_document_filter": False,
    }
    client.generate_stream = AsyncMock(return_value=_async_token_gen())
    client.health_check.return_value = True
    return client


async def _async_token_gen():
    """Async generator yielding fake tokens."""
    for token in ["Halo", " dunia", "!"]:
        yield token


@pytest.fixture
def mock_vector_client():
    """BaseVectorStoreClient mock — returns list[dict] matching result_mapper output."""
    client = AsyncMock()
    client.hybrid_search.return_value = [
        {
            "chunk_id": FAKE_CHUNK_ID,
            "content": "Test legal content about BPJS",
            "score": 0.9,
            "document_id": FAKE_DOCUMENT_ID,
            "metadata": {},
        }
    ]
    client.upsert_chunks.return_value = None
    client.delete_by_document_id.return_value = None
    return client


@pytest.fixture
def mock_graph_client():
    """BaseGraphClient mock."""
    client = AsyncMock()
    client.multi_hop_query.return_value = []
    client.upsert_entities.return_value = None
    client.upsert_relations.return_value = None
    return client


@pytest.fixture
def mock_embedding_client():
    """BaseEmbeddingClient mock."""
    client = AsyncMock()
    client.embed.return_value = [[0.1, 0.2, 0.3]]
    client.embed_single.return_value = [0.1, 0.2, 0.3]
    return client


@pytest.fixture
def mock_reranker_client():
    """
    BaseRerankerClient mock.

    Returns list[dict] — matches ScoreProcessor.process() output format.
    Each dict: {text: str, score: float}
    """
    client = AsyncMock()
    client.rerank.return_value = [
        {"text": "Test legal content about BPJS", "score": 0.95},
    ]
    return client


@pytest.fixture
def mock_cache_client():
    """BaseCacheClient mock."""
    client = AsyncMock()
    client.get.return_value = None
    client.set.return_value = None
    client.delete.return_value = None
    client.exists.return_value = False
    return client


@pytest.fixture
def mock_observability_client():
    """
    BaseObservabilityClient mock — NoOpObservabilityClient for unit tests.

    Returns the real NoOpObservabilityClient (not a MagicMock) so that
    type checking passes and call signatures are validated correctly.
    In unit tests, we only care that observability calls don't crash the service.
    For spy behavior, wrap it: MagicMock(wraps=NoOpObservabilityClient()).
    """
    from src.core.observability.noop_client import NoOpObservabilityClient
    return NoOpObservabilityClient()


@pytest.fixture
def spy_observability_client():
    """
    Spy wrapper around NoOpObservabilityClient.

    Use when a test needs to assert that specific observability methods
    were called (e.g. start_trace, flush) without actually sending data.
    """
    from src.core.observability.noop_client import NoOpObservabilityClient
    return MagicMock(wraps=NoOpObservabilityClient())