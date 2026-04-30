"""Conversation management routes."""
from __future__ import annotations
from fastapi import APIRouter, Depends, Query, status
from sqlalchemy.ext.asyncio import AsyncSession
from src.api.schemas.common.pagination import PaginatedResponse, PaginationMeta
from src.api.schemas.common.response_envelope import SuccessResponse
from src.api.schemas.conversation.conversation_request import CreateConversationRequest, UpdateConversationRequest
from src.api.schemas.conversation.conversation_response import ConversationResponse, MessageResponse
from src.core.config.settings import settings
from src.core.constants.pagination import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE
from src.entities.conversation.conversation import Conversation
from src.entities.message.message import Message
from src.providers import get_conversation_repo, get_message_repo, get_db_session

router = APIRouter(prefix="/conversations", tags=["conversations"])


def _conv_to_response(c: Conversation) -> ConversationResponse:
    return ConversationResponse(
        conversation_id=c.conversation_id, user_id=c.user_id, title=c.title,
        is_active=c.is_active, created_at=c.created_at, updated_at=c.updated_at,
    )


def _msg_to_response(m: Message) -> MessageResponse:
    return MessageResponse(
        message_id=m.message_id, conversation_id=m.conversation_id,
        role=m.role, content=m.content, system_prompt_id=m.system_prompt_id,
        model_id=m.model_id, token_count=m.token_count, created_at=m.created_at,
    )


@router.post("", response_model=SuccessResponse[ConversationResponse], status_code=status.HTTP_201_CREATED)
async def create_conversation(
    body: CreateConversationRequest,
    db: AsyncSession = Depends(get_db_session),
):
    user_id = settings.DEFAULT_USER_ID or None
    repo = get_conversation_repo(db)
    conv = await repo.create(user_id=user_id, title=body.title)
    return SuccessResponse(data=_conv_to_response(conv))


@router.get("", response_model=PaginatedResponse[ConversationResponse])
async def list_conversations(
    limit: int = Query(default=DEFAULT_PAGE_SIZE, le=MAX_PAGE_SIZE, ge=1),
    offset: int = Query(default=0, ge=0),
    db: AsyncSession = Depends(get_db_session),
):
    user_id = settings.DEFAULT_USER_ID or None
    repo = get_conversation_repo(db)
    convs = await repo.get_all_by_user(user_id=user_id, limit=limit, offset=offset)
    return PaginatedResponse(
        data=[_conv_to_response(c) for c in convs],
        pagination=PaginationMeta(page=offset // limit + 1, page_size=limit, has_next=len(convs) == limit),
    )


@router.get("/{conversation_id}", response_model=SuccessResponse[ConversationResponse])
async def get_conversation(conversation_id: str, db: AsyncSession = Depends(get_db_session)):
    from src.core.exceptions.not_found import ConversationNotFoundError
    repo = get_conversation_repo(db)
    conv = await repo.get_by_id(conversation_id)
    if not conv:
        raise ConversationNotFoundError(conversation_id)
    return SuccessResponse(data=_conv_to_response(conv))


@router.patch("/{conversation_id}", response_model=SuccessResponse[ConversationResponse])
async def update_conversation(
    conversation_id: str,
    body: UpdateConversationRequest,
    db: AsyncSession = Depends(get_db_session),
):
    repo = get_conversation_repo(db)
    conv = await repo.update_title(conversation_id, body.title)
    return SuccessResponse(data=_conv_to_response(conv))


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(conversation_id: str, db: AsyncSession = Depends(get_db_session)):
    repo = get_conversation_repo(db)
    await repo.soft_delete(conversation_id)


@router.get("/{conversation_id}/messages", response_model=PaginatedResponse[MessageResponse])
async def list_messages(
    conversation_id: str,
    limit: int = Query(default=DEFAULT_PAGE_SIZE, le=MAX_PAGE_SIZE, ge=1),
    db: AsyncSession = Depends(get_db_session),
):
    repo = get_message_repo(db)
    messages = await repo.get_history(conversation_id=conversation_id, limit=limit)
    return PaginatedResponse(
        data=[_msg_to_response(m) for m in messages],
        pagination=PaginationMeta(page=1, page_size=limit, has_next=len(messages) == limit),
    )
