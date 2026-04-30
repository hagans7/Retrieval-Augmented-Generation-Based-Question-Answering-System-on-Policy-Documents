"""Abstract base for all embedding clients."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseEmbeddingClient(ABC):
    """Contract for all text embedding service clients."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Generate dense vector embeddings for a list of texts.

        Args:
            texts: List of text strings to embed. Must not be empty.

        Returns:
            List of float vectors, one per input text.
            Order is preserved: result[i] corresponds to texts[i].

        Raises:
            EmbeddingError: If the embedding service fails.
        """
        ...

    @abstractmethod
    async def embed_single(self, text: str) -> list[float]:
        """
        Generate a dense vector embedding for a single text.

        Args:
            text: Text string to embed.

        Returns:
            Float vector representing the text.

        Raises:
            EmbeddingError: If the embedding service fails.
        """
        ...
