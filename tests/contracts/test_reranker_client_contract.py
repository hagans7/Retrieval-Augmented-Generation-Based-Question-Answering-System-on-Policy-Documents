"""
Contract tests for BaseRerankerClient → OpenRouter/Cohere backend.

Requires RERANKER_BASE_URL and LLM_API_KEY set in .env.
Skip if keys absent.

What is tested:
- rerank() returns list[dict] with {text, score}
- Results sorted by score DESC
- min_score threshold respected with min-1 guarantee
- top_k limits results

To run:
    pytest tests/contracts/test_reranker_client_contract.py -v
"""
from __future__ import annotations

import pytest


def _should_skip() -> tuple[bool, str]:
    try:
        from src.core.config.settings import settings
        if not settings.LLM_API_KEY:
            return True, "LLM_API_KEY not set in .env"
        if not settings.RERANKER_BASE_URL:
            return True, "RERANKER_BASE_URL not set in .env"
        return False, ""
    except Exception as exc:
        return True, f"Settings load failed: {exc}"


_SKIP, _REASON = _should_skip()
pytestmark = pytest.mark.skipif(_SKIP, reason=_REASON)

DOCUMENTS = [
    "Mahasiswa PENS diwajibkan mendaftar BPJS Kesehatan.",
    "Tarif pajak penghasilan badan usaha adalah 22 persen.",
    "Bunga bank ditetapkan oleh Bank Indonesia setiap bulan.",
    "Ketentuan wisuda diumumkan oleh Wakil Direktur bidang Kemahasiswaan.",
    "Dokumen akta notaris harus dilegalisir oleh pejabat berwenang.",
]


@pytest.fixture(scope="module")
def reranker_client():
    from src.clients.reranker.reranker_client import OpenAIRerankerClient
    from src.core.config.settings import settings
    return OpenAIRerankerClient(
        base_url=settings.RERANKER_BASE_URL,
        api_key=settings.LLM_API_KEY,
        model_name=settings.RERANKER_MODEL_NAME,
        timeout=settings.RERANKER_TIMEOUT,
    )


@pytest.mark.asyncio
async def test_rerank_returns_list_of_dicts(reranker_client):
    results = await reranker_client.rerank(
        query="BPJS Kesehatan mahasiswa",
        documents=DOCUMENTS,
        top_k=3,
    )
    assert isinstance(results, list)
    assert all(isinstance(r, dict) for r in results)


@pytest.mark.asyncio
async def test_result_has_text_and_score_keys(reranker_client):
    results = await reranker_client.rerank(
        query="BPJS Kesehatan mahasiswa",
        documents=DOCUMENTS,
        top_k=3,
    )
    assert len(results) > 0
    for item in results:
        assert "text" in item, f"Missing 'text' in {item}"
        assert "score" in item, f"Missing 'score' in {item}"
        assert isinstance(item["text"], str)
        assert isinstance(item["score"], float)


@pytest.mark.asyncio
async def test_results_sorted_by_score_desc(reranker_client):
    results = await reranker_client.rerank(
        query="BPJS Kesehatan mahasiswa",
        documents=DOCUMENTS,
        top_k=5,
    )
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True), "Results must be sorted by score DESC"


@pytest.mark.asyncio
async def test_top_k_limits_results(reranker_client):
    results = await reranker_client.rerank(
        query="dokumen hukum",
        documents=DOCUMENTS,
        top_k=2,
    )
    assert len(results) <= 2


@pytest.mark.asyncio
async def test_min_score_guarantees_at_least_one_result(reranker_client):
    """Even with extremely high threshold, must return at least 1 result."""
    results = await reranker_client.rerank(
        query="BPJS mahasiswa PENS",
        documents=DOCUMENTS,
        top_k=5,
        min_score=0.9999,
    )
    assert len(results) >= 1, "min_score must guarantee at least 1 result"


@pytest.mark.asyncio
async def test_most_relevant_document_ranks_first(reranker_client):
    """The BPJS document should rank highest for a BPJS query."""
    results = await reranker_client.rerank(
        query="BPJS Kesehatan pendaftaran mahasiswa",
        documents=DOCUMENTS,
        top_k=len(DOCUMENTS),
    )
    assert "BPJS" in results[0]["text"], (
        f"Expected BPJS document first, got: {results[0]['text'][:80]}"
    )