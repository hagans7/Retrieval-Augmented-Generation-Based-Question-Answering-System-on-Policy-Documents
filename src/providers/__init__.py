"""
Providers package — single import entry point for all dependency providers.

Route handlers import exclusively from this package.
If a provider moves between internal files, only this __init__ changes.
"""
from src.providers.infrastructure.clients import (
    get_cache_client,
    get_embedding_client,
    get_graph_client,
    get_llm_client,
    get_ocr_client,
    get_reranker_client,
    get_storage_client,
    get_vector_client,
)
from src.providers.infrastructure.database import get_db_session, check_db_connectivity
from src.providers.repositories.repositories import (
    get_available_model_repo,
    get_conversation_repo,
    get_document_repo,
    get_message_repo,
    get_system_prompt_repo,
)
from src.providers.services.services import (
    get_ingest_document_service,
    get_manage_prompt_service,
    get_process_chat_service,
    get_resolve_model_service,
)

__all__ = [
    "get_db_session",
    "check_db_connectivity",
    "get_llm_client",
    "get_embedding_client",
    "get_reranker_client",
    "get_vector_client",
    "get_graph_client",
    "get_storage_client",
    "get_cache_client",
    "get_ocr_client",
    "get_conversation_repo",
    "get_message_repo",
    "get_system_prompt_repo",
    "get_available_model_repo",
    "get_document_repo",
    "get_process_chat_service",
    "get_ingest_document_service",
    "get_manage_prompt_service",
    "get_resolve_model_service",
]
