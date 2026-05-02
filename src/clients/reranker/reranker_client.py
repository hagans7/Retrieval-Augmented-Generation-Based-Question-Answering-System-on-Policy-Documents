"""
Reranker client — OpenRouter/Cohere rerank API.

Works with any OpenRouter-compatible rerank endpoint:
  - OpenRouter: RERANKER_BASE_URL=https://openrouter.ai/api/v1/rerank
  - Cohere:     RERANKER_BASE_URL=https://api.cohere.com/v2/rerank

IMPORTANT — base_url is the FULL endpoint URL for this client.
The client POSTs directly to RERANKER_BASE_URL without appending any path.

Request format:
    POST {RERANKER_BASE_URL}
    Authorization: Bearer {api_key}
    {
      "model": "cohere/rerank-4-fast",
      "query": "...",
      "documents": ["doc1", "doc2", ...],
      "top_n": 5
    }

Response format:
    {
      "results": [
        {"index": 0, "relevance_score": 0.95, "document": {"text": "..."}}
      ]
    }
"""

from __future__ import annotations

import httpx

from src.clients.reranker.score_processor import ScoreProcessor
from src.core.exceptions.infrastructure import RerankerError
from src.core.logging.logger import get_logger
from src.interfaces.clients.base_reranker_client import BaseRerankerClient


class OpenAIRerankerClient(BaseRerankerClient):
    """
    Async reranker client for OpenRouter/Cohere rerank API.

    Renamed from TEIRerankerClient — no longer TEI-specific.

    Args:
        base_url: Full rerank endpoint URL.
                  For OpenRouter: "https://openrouter.ai/api/v1/rerank"
        api_key: Bearer token for Authorization header (same as LLM_API_KEY).
        model_name: Model identifier sent in request body
                    (e.g. "cohere/rerank-4-fast").
        timeout: HTTP timeout in seconds.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        timeout: int = 30,
    ) -> None:
        # base_url IS the endpoint — do not append any path
        self._endpoint = base_url.rstrip("/")
        self._api_key = api_key
        self._model_name = model_name
        self._timeout = timeout
        self._logger = get_logger(__name__)

    async def rerank(
        self,
        query: str,
        documents: list[str],
        top_k: int,
        min_score: float = 0.0,
    ) -> list[dict]:
        """
        Rerank documents by relevance to query using OpenRouter/Cohere API.

        Args:
            query: The search query.
            documents: List of document text strings to rank.
            top_k: Maximum number of results to return.

        Returns:
            List of dicts {text: str, score: float} sorted by score DESC.

        Raises:
            RerankerError: If the API call fails.
        """
        if not documents:
            return []

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    self._endpoint,
                    json={
                        "model": self._model_name,
                        "query": query,
                        "documents": documents,
                        "top_n": top_k,
                    },
                    headers={"Authorization": f"Bearer {self._api_key}"},
                )
                response.raise_for_status()
                return ScoreProcessor.process(
                    raw_response=response.json(),
                    documents=documents,
                    top_k=top_k,
                    min_score=min_score,
                )

        except httpx.HTTPStatusError as exc:
            raise RerankerError(
                message=(
                    f"Reranker API returned HTTP {exc.response.status_code}. "
                    f"Check RERANKER_MODEL_NAME and RERANKER_BASE_URL."
                ),
                context={
                    "status_code": exc.response.status_code,
                    "model": self._model_name,
                    "endpoint": self._endpoint,
                },
            ) from exc
        except httpx.RequestError as exc:
            raise RerankerError(
                message=(
                    f"Reranker service unreachable at '{self._endpoint}'. "
                    f"Check RERANKER_BASE_URL."
                ),
                context={"endpoint": self._endpoint, "error": str(exc)},
            ) from exc


# Backward-compatible alias
TEIRerankerClient = OpenAIRerankerClient