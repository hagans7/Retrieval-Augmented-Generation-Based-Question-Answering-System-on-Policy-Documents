"""
Contract tests for BaseEmbeddingClient (OpenAI-compatible format).

Reads configuration from src.core.config.settings — which loads from .env file
via pydantic-settings. Does NOT read from os.environ directly, to avoid
picking up stale system environment values from previous Docker sessions.

SKIP conditions:
  - EMBEDDING_BASE_URL points to a Docker internal hostname (tei-embedding)
  - LLM_API_KEY is empty (needed for OpenRouter auth)

To run:
    pytest tests/contracts/test_embedding_client_contract.py -v

Requirements:
  - EMBEDDING_BASE_URL=https://openrouter.ai/api/v1 in .env
  - EMBEDDING_MODEL_NAME=baai/bge-m3 in .env  (note: "bge-m3", not "bge-m")
  - LLM_API_KEY=sk-or-v1-... in .env
"""
from __future__ import annotations

import pytest


def _should_skip() -> tuple[bool, str]:
    """Determine skip condition from settings (reads .env, not os.environ)."""
    try:
        from src.core.config.settings import settings
        url = settings.EMBEDDING_BASE_URL
        if not url:
            return True, "EMBEDDING_BASE_URL is empty in .env"
        if "tei-embedding" in url:
            return True, (
                f"EMBEDDING_BASE_URL='{url}' is a Docker internal hostname. "
                "Change to http://localhost:8081 to run from host."
            )
        if not settings.LLM_API_KEY:
            return True, "LLM_API_KEY is empty in .env — needed for OpenRouter auth."
        return False, ""
    except Exception as exc:
        return True, f"Could not load settings: {exc}"


_SKIP, _REASON = _should_skip()
pytestmark = pytest.mark.skipif(_SKIP, reason=_REASON)


@pytest.fixture
def embedding_client():
    """Build embedding client from .env settings."""
    from src.clients.embedding.embedding_client import OpenAIEmbeddingClient
    from src.core.config.settings import settings
    return OpenAIEmbeddingClient(
        base_url=settings.EMBEDDING_BASE_URL,
        api_key=settings.LLM_API_KEY,
        model_name=settings.EMBEDDING_MODEL_NAME,
        timeout=settings.EMBEDDING_TIMEOUT,
    )


@pytest.mark.asyncio
async def test_embed_single_returns_float_list(embedding_client):
    """embed_single() must return a non-empty list of floats."""
    result = await embedding_client.embed_single("Pasal 1338 KUH Perdata")
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(isinstance(v, (float, int)) for v in result)


@pytest.mark.asyncio
async def test_embed_multiple_returns_correct_count(embedding_client):
    """embed() must return one vector per input text."""
    texts = ["Wanprestasi", "Force majeure", "Perjanjian sewa"]
    result = await embedding_client.embed(texts)
    assert len(result) == len(texts)
    for vec in result:
        assert isinstance(vec, list)
        assert len(vec) > 0


@pytest.mark.asyncio
async def test_embed_empty_list_returns_empty(embedding_client):
    """embed([]) must return [] without raising."""
    result = await embedding_client.embed([])
    assert result == []


@pytest.mark.asyncio
async def test_embed_vectors_have_same_dimension(embedding_client):
    """All vectors from one call must have the same dimension."""
    texts = ["Hukum perdata", "Hukum pidana"]
    result = await embedding_client.embed(texts)
    dims = [len(v) for v in result]
    assert len(set(dims)) == 1, f"Vectors have different dimensions: {dims}"