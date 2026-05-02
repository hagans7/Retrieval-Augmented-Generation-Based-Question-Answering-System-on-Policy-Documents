"""
Service provider functions.

Wires all dependencies into service constructors.
This file is the only place where concrete implementations are combined.
"""
from __future__ import annotations

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.providers.infrastructure.clients import (
    get_cache_client,
    get_embedding_client,
    get_graph_client,
    get_llm_client,
    get_observability_client,
    get_ocr_client,
    get_reranker_client,
    get_storage_client,
    get_vector_client,
)
from src.providers.infrastructure.database import get_db_session
from src.providers.repositories.repositories import (
    get_available_model_repo,
    get_conversation_repo,
    get_document_repo,
    get_message_repo,
    get_system_prompt_repo,
)


def get_process_chat_service(db: AsyncSession = Depends(get_db_session)):
    from src.services.process_chat.process_chat import ProcessChatService
    return ProcessChatService(
        conversation_repo=get_conversation_repo(db),
        message_repo=get_message_repo(db),
        system_prompt_repo=get_system_prompt_repo(db),
        model_repo=get_available_model_repo(db),
        llm_client=get_llm_client(),
        vector_client=get_vector_client(),
        graph_client=get_graph_client(),
        embedding_client=get_embedding_client(),
        reranker_client=get_reranker_client(),
        cache_client=get_cache_client(),
        observability_client=get_observability_client(),
    )


def get_ingest_document_service(db: AsyncSession = Depends(get_db_session)):
    from src.services.ingest_document.ingest_document import IngestDocumentService
    return IngestDocumentService(
        document_repo=get_document_repo(db),
        storage_client=get_storage_client(),
        ocr_client=get_ocr_client(),
        embedding_client=get_embedding_client(),
        vector_client=get_vector_client(),
        graph_client=get_graph_client(),
        llm_client=get_llm_client(),
    )


def get_manage_prompt_service(db: AsyncSession = Depends(get_db_session)):
    from src.services.manage_prompt.manage_prompt import ManagePromptService
    return ManagePromptService(system_prompt_repo=get_system_prompt_repo(db))


def get_resolve_model_service(db: AsyncSession = Depends(get_db_session)):
    from src.services.resolve_model.resolve_model import ResolveModelService
    return ResolveModelService(model_repo=get_available_model_repo(db))