"""
Embedding batch processor — OpenAI-compatible format for OpenRouter.

Request format:  POST {base_url}/embeddings
                 {"model": "...", "input": ["text1", "text2"]}
                 Authorization: Bearer {api_key}

Response format: {"data": [{"embedding": [...], "index": 0}, ...]}

Automatically batches large input lists. Results are re-sorted by index
to guarantee output order matches input order, regardless of response order.
"""

from __future__ import annotations

import httpx

from src.core.constants.ingestion import EMBEDDING_BATCH_SIZE
from src.core.exceptions.infrastructure import EmbeddingError
from src.core.logging.logger import get_logger

logger = get_logger(__name__)


class EmbeddingBatchProcessor:
    """
    Processes large text lists by batching OpenAI-compatible embedding requests.

    Args:
        base_url: API base URL (e.g. "https://openrouter.ai/api/v1").
                  The processor appends "/embeddings" to this URL.
        api_key:  Bearer token for Authorization header.
        model_name: Model identifier sent in the request body.
        timeout: HTTP timeout in seconds.
        batch_size: Maximum texts per request.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        timeout: int,
        batch_size: int = EMBEDDING_BATCH_SIZE,
    ) -> None:
        self._endpoint = base_url.rstrip("/") + "/embeddings"
        self._api_key = api_key
        self._model_name = model_name
        self._timeout = timeout
        self._batch_size = batch_size

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of texts, batching automatically if needed.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of float vectors in the same order as input.

        Raises:
            EmbeddingError: If any batch fails.
        """
        if not texts:
            return []

        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            vectors = await self._embed_single_batch(batch)
            all_vectors.extend(vectors)

        return all_vectors

    async def _embed_single_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Send one batch to the embedding API and return ordered vectors.

        Sorts by index to guarantee input-output order consistency.
        """
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    self._endpoint,
                    json={"model": self._model_name, "input": texts},
                    headers={"Authorization": f"Bearer {self._api_key}"},
                )
                response.raise_for_status()
                return self._parse_response(response.json(), expected_count=len(texts))

        except httpx.HTTPStatusError as exc:
            raise EmbeddingError(
                message=(
                    f"Embedding API returned HTTP {exc.response.status_code}. "
                    f"Check EMBEDDING_MODEL_NAME and EMBEDDING_BASE_URL."
                ),
                context={
                    "status_code": exc.response.status_code,
                    "model": self._model_name,
                    "batch_size": len(texts),
                },
            ) from exc
        except httpx.RequestError as exc:
            raise EmbeddingError(
                message=(
                    f"Embedding service unreachable at '{self._endpoint}'. "
                    f"Check EMBEDDING_BASE_URL."
                ),
                context={"endpoint": self._endpoint, "error": str(exc)},
            ) from exc

    @staticmethod
    def _parse_response(response_json: dict, expected_count: int) -> list[list[float]]:
        """
        Parse OpenAI-compatible embedding response into ordered vector list.

        Handles: {"data": [{"embedding": [...], "index": N}, ...]}
        Sorts by index field to guarantee order matches input order.
        """
        data = response_json.get("data", [])
        if not data:
            raise EmbeddingError(
                message="Embedding response contained no data.",
                context={"response_keys": list(response_json.keys())},
            )
        # Sort by index — API may return in any order
        sorted_data = sorted(data, key=lambda x: x.get("index", 0))
        vectors = [item["embedding"] for item in sorted_data]

        if len(vectors) != expected_count:
            raise EmbeddingError(
                message=f"Expected {expected_count} vectors, got {len(vectors)}.",
                context={"expected": expected_count, "received": len(vectors)},
            )
        return vectors