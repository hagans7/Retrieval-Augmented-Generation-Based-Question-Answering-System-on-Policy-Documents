"""
Contract tests for BaseVectorStoreClient (Weaviate).

Reads from settings (.env), not os.environ directly.

SKIP: if WEAVIATE_URL points to Docker internal hostname.

To run locally (after exposing port 8080 in docker-compose):
    pytest tests/contracts/test_vector_store_client_contract.py -v
"""
from __future__ import annotations

import uuid
import pytest


def _should_skip() -> tuple[bool, str]:
    try:
        from src.core.config.settings import settings
        url = settings.WEAVIATE_URL
        if not url:
            return True, "WEAVIATE_URL is empty in .env"
        # Docker internal hostnames aren't resolvable from host machine
        if "weaviate" in url and "localhost" not in url:
            return True, (
                f"WEAVIATE_URL='{url}' is a Docker internal hostname. "
                "Set WEAVIATE_URL=http://localhost:8080 to run from host."
            )
        return False, ""
    except Exception as exc:
        return True, f"Could not load settings: {exc}"


_SKIP, _REASON = _should_skip()
pytestmark = pytest.mark.skipif(_SKIP, reason=_REASON)

TEST_COLLECTION = "ContractTestChunk"


@pytest.fixture
def vector_client():
    from src.clients.vector_store.vector_store_client import WeaviateVectorStoreClient
    from src.core.config.settings import settings
    client = WeaviateVectorStoreClient(
        weaviate_url=settings.WEAVIATE_URL,
        collection_name=TEST_COLLECTION,
    )
    client.connect()
    yield client
    try:
        client._client.collections.delete(TEST_COLLECTION)
    except Exception:
        pass
    client.close()


@pytest.mark.asyncio
async def test_upsert_and_search_returns_results(vector_client):
    doc_id = str(uuid.uuid4())
    vector = [0.1] * 768
    await vector_client.upsert_chunks([{
        "chunk_id": str(uuid.uuid4()),
        "content": "Pasal 1338 menyatakan bahwa perjanjian mengikat para pihak.",
        "document_id": doc_id,
        "vector": vector,
        "chunk_index": 0,
        "file_name": "test.pdf",
        "file_type": "pdf",
        "section_type": "pasal",
        "parent_section": "BAB I",
        "token_count": 20,
    }])
    results = await vector_client.hybrid_search(
        query_vector=vector, query_text="Pasal 1338", top_k=5
    )
    assert isinstance(results, list)
    assert len(results) > 0


@pytest.mark.asyncio
async def test_delete_by_document_id_removes_chunks(vector_client):
    doc_id = str(uuid.uuid4())
    vector = [0.0] * 768
    await vector_client.upsert_chunks([{
        "chunk_id": str(uuid.uuid4()),
        "content": "Test content to delete",
        "document_id": doc_id,
        "vector": vector,
        "chunk_index": 0,
        "file_name": "test.pdf",
        "file_type": "pdf",
        "section_type": "test",
        "parent_section": "",
        "token_count": 5,
    }])
    await vector_client.delete_by_document_id(doc_id)
    results = await vector_client.hybrid_search(
        query_vector=vector, query_text="Test content to delete", top_k=5,
        filters={"document_id": doc_id},
    )
    doc_results = [r for r in results if r.get("document_id") == doc_id]
    assert len(doc_results) == 0