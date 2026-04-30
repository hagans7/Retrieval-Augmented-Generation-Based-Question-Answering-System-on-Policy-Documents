"""
Shared pytest fixtures for all test categories.

Provides:
- async_session: In-memory SQLite async session for repository tests
- mock clients: AsyncMock implementations for all infrastructure clients
- default IDs: Consistent UUIDs for entity creation
"""
from __future__ import annotations

from src.db_models import register_all_models
register_all_models()  # ensure ORM mappers configured before tests
import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock

FAKE_CONVERSATION_ID = str(uuid.uuid4())
FAKE_MESSAGE_ID = str(uuid.uuid4())
FAKE_PROMPT_ID = str(uuid.uuid4())
FAKE_MODEL_ID = str(uuid.uuid4())
FAKE_DOCUMENT_ID = str(uuid.uuid4())
FAKE_USER_ID = str(uuid.uuid4())


@pytest.fixture
def mock_llm_client():
    client = AsyncMock()
    client.generate.return_value = "Mocked LLM response"
    client.generate_with_tool_call.return_value = {
        "query_type": "retrieval",
        "reasoning": "test",
        "requires_document_filter": False,
    }
    return client


@pytest.fixture
def mock_vector_client():
    client = AsyncMock()
    client.hybrid_search.return_value = [
        {"chunk_id": str(uuid.uuid4()), "content": "Test legal content", "score": 0.9, "document_id": FAKE_DOCUMENT_ID, "metadata": {}}
    ]
    client.upsert_chunks.return_value = None
    client.delete_by_document_id.return_value = None
    return client


@pytest.fixture
def mock_graph_client():
    client = AsyncMock()
    client.multi_hop_query.return_value = []
    client.upsert_entities.return_value = None
    client.upsert_relations.return_value = None
    return client


@pytest.fixture
def mock_embedding_client():
    client = AsyncMock()
    client.embed.return_value = [[0.1, 0.2, 0.3]]
    client.embed_single.return_value = [0.1, 0.2, 0.3]
    return client


@pytest.fixture
def mock_reranker_client():
    client = AsyncMock()
    client.rerank.return_value = [("Test legal content", 0.95)]
    return client


@pytest.fixture
def mock_cache_client():
    client = AsyncMock()
    client.get.return_value = None
    client.set.return_value = None
    client.delete.return_value = None
    client.exists.return_value = False
    return client