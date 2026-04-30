"""
Embedding client — OpenAI-compatible embedding API (OpenRouter / self-hosted).

Works with any OpenAI-compatible embeddings endpoint:
  - OpenRouter: EMBEDDING_BASE_URL=https://openrouter.ai/api/v1
  - OpenAI:     EMBEDDING_BASE_URL=https://api.openai.com/v1
  - Local vLLM: EMBEDDING_BASE_URL=http://localhost:8000/v1

The API key is passed as Bearer token in the Authorization header.
The LLM_API_KEY from settings is reused (same OpenRouter account).
"""

from __future__ import annotations

from src.clients.embedding.batch_processor import EmbeddingBatchProcessor
from src.clients.embedding.response_validator import EmbeddingResponseValidator
from src.core.logging.logger import get_logger
from src.interfaces.clients.base_embedding_client import BaseEmbeddingClient


class OpenAIEmbeddingClient(BaseEmbeddingClient):
    """
    Async embedding client for OpenAI-compatible APIs.

    Renamed from TEIEmbeddingClient — no longer TEI-specific.

    Args:
        base_url: API base URL. Client appends "/embeddings" to this.
        api_key: Bearer token for Authorization header.
        model_name: Model identifier sent in request body
                    (e.g. "baai/bge-m3" for OpenRouter).
        timeout: HTTP timeout in seconds.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model_name: str,
        timeout: int = 30,
    ) -> None:
        self._model_name = model_name
        self._processor = EmbeddingBatchProcessor(
            base_url=base_url,
            api_key=api_key,
            model_name=model_name,
            timeout=timeout,
        )
        self._logger = get_logger(__name__)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of texts using the OpenAI-compatible embeddings API.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of float vectors, one per input text.

        Raises:
            EmbeddingError: If the API call fails.
        """
        if not texts:
            return []
        vectors = await self._processor.embed_batch(texts)
        return EmbeddingResponseValidator.validate(vectors, len(texts))

    async def embed_single(self, text: str) -> list[float]:
        """
        Embed a single text string.

        Returns:
            Float vector for the input text.

        Raises:
            EmbeddingError: If the API call fails.
        """
        vectors = await self.embed([text])
        return vectors[0]


# Backward-compatible alias — existing imports of TEIEmbeddingClient still work.
TEIEmbeddingClient = OpenAIEmbeddingClient