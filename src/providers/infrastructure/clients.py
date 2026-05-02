"""
Infrastructure client providers.

All singleton clients created here via functools.lru_cache.
Route handlers and services never instantiate clients directly.
Return types are always the interface type (ABC), never concrete class.

Embedding and Reranker:
    Both use OpenRouter as backend, which follows the OpenAI-compatible API.
    The LLM_API_KEY is reused as the Bearer token for all OpenRouter calls.
    Set EMBEDDING_BASE_URL and RERANKER_BASE_URL to the appropriate endpoints.
"""
from __future__ import annotations

from functools import lru_cache

from src.core.config.settings import settings
from src.core.logging.logger import get_logger
from src.interfaces.clients.base_cache_client import BaseCacheClient
from src.interfaces.clients.base_embedding_client import BaseEmbeddingClient
from src.interfaces.clients.base_graph_client import BaseGraphClient
from src.interfaces.clients.base_llm_client import BaseLLMClient
from src.interfaces.clients.base_ocr_client import BaseOCRClient
from src.interfaces.clients.base_reranker_client import BaseRerankerClient
from src.interfaces.clients.base_storage_client import BaseStorageClient
from src.interfaces.clients.base_observability_client import BaseObservabilityClient
from src.interfaces.clients.base_vector_store_client import BaseVectorStoreClient


logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_llm_client() -> BaseLLMClient:
    """Return the singleton LLM client (OpenAI-compatible)."""
    from src.clients.llm.llm_client import OpenAICompatibleLLMClient
    return OpenAICompatibleLLMClient(
        base_url=settings.LLM_BASE_URL,
        api_key=settings.LLM_API_KEY,
        default_model=settings.LLM_DEFAULT_MODEL,
        timeout=settings.LLM_TIMEOUT,
        max_retries=settings.LLM_MAX_RETRIES,
    )


@lru_cache(maxsize=1)
def get_embedding_client() -> BaseEmbeddingClient:
    """
    Return the singleton embedding client (OpenAI-compatible).

    Uses LLM_API_KEY as the Bearer token — same OpenRouter account.
    EMBEDDING_BASE_URL should be the API base (e.g. https://openrouter.ai/api/v1).
    Client appends '/embeddings' to the base URL.
    """
    from src.clients.embedding.embedding_client import OpenAIEmbeddingClient
    return OpenAIEmbeddingClient(
        base_url=settings.EMBEDDING_BASE_URL,
        api_key=settings.LLM_API_KEY,
        model_name=settings.EMBEDDING_MODEL_NAME,
        timeout=settings.EMBEDDING_TIMEOUT,
    )


@lru_cache(maxsize=1)
def get_reranker_client() -> BaseRerankerClient:
    """
    Return the singleton reranker client (OpenRouter/Cohere format).

    Uses LLM_API_KEY as the Bearer token — same OpenRouter account.
    RERANKER_BASE_URL should be the full rerank endpoint URL
    (e.g. https://openrouter.ai/api/v1/rerank).
    Client POSTs directly to this URL without appending any path.
    """
    from src.clients.reranker.reranker_client import OpenAIRerankerClient
    return OpenAIRerankerClient(
        base_url=settings.RERANKER_BASE_URL,
        api_key=settings.LLM_API_KEY,
        model_name=settings.RERANKER_MODEL_NAME,
        timeout=settings.RERANKER_TIMEOUT,
    )


@lru_cache(maxsize=1)
def get_vector_client() -> BaseVectorStoreClient:
    """Return the singleton Weaviate client (connected at startup)."""
    from src.clients.vector_store.vector_store_client import WeaviateVectorStoreClient
    return WeaviateVectorStoreClient(
        weaviate_url=settings.WEAVIATE_URL,
        collection_name=settings.WEAVIATE_COLLECTION_NAME,
    )


@lru_cache(maxsize=1)
def get_graph_client() -> BaseGraphClient:
    """Return the singleton Neo4j client."""
    from src.clients.graph.graph_client import Neo4jGraphClient
    return Neo4jGraphClient(
        uri=settings.NEO4J_URI,
        user=settings.NEO4J_USER,
        password=settings.NEO4J_PASSWORD,
    )


@lru_cache(maxsize=1)
def get_storage_client() -> BaseStorageClient:
    """Return the singleton S3-compatible storage client."""
    from src.clients.storage.storage_client import S3CompatibleStorageClient
    return S3CompatibleStorageClient(
        endpoint=settings.STORAGE_ENDPOINT,
        access_key=settings.STORAGE_ACCESS_KEY,
        secret_key=settings.STORAGE_SECRET_KEY,
        bucket=settings.STORAGE_BUCKET,
        use_ssl=settings.STORAGE_USE_SSL,
    )


@lru_cache(maxsize=1)
def get_cache_client() -> BaseCacheClient:
    """Return the singleton Redis cache client."""
    from src.clients.cache.cache_client import RedisCacheClient
    return RedisCacheClient(url=settings.REDIS_URL)


@lru_cache(maxsize=1)
def get_ocr_client() -> BaseOCRClient:
    """Return the singleton OCR client (lazy init — no startup cost)."""
    from src.clients.ocr.ocr_client import DoclingOCRClient
    return DoclingOCRClient(
        model_repo=settings.OCR_MODEL_REPO,
        model_dir=settings.OCR_MODEL_DIR,
        hf_token=settings.HF_TOKEN,
    )

@lru_cache(maxsize=1)
def get_observability_client() -> BaseObservabilityClient:
    """
    Return the singleton observability client.

    Selects implementation based on settings:
      - LangfuseObservabilityClient when LANGFUSE keys are present
      - NoOpObservabilityClient when keys are absent (dev / test)

    Return type is always BaseObservabilityClient — callers never
    depend on the concrete class.

    NOTE: settings is imported at module level (not inside this function)
    so that tests can patch it via:
        patch("src.providers.infrastructure.clients.settings")
    """
    if settings.LANGFUSE_SECRET_KEY and settings.LANGFUSE_PUBLIC_KEY:
        try:
            from src.core.observability.langfuse_client import LangfuseObservabilityClient
            return LangfuseObservabilityClient(
                public_key=settings.LANGFUSE_PUBLIC_KEY,
                secret_key=settings.LANGFUSE_SECRET_KEY,
                host=settings.LANGFUSE_HOST,
            )
        except Exception as exc:
            logger.warning(
                "Langfuse init failed, falling back to no-op",
                extra={"error": str(exc)},
            )

    logger.info("Observability disabled — using no-op client")
    from src.core.observability.noop_client import NoOpObservabilityClient
    return NoOpObservabilityClient()