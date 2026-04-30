"""
Chat routes — handles POST /chat for both stream and non-stream modes.

conversation_id is optional. When null, the service auto-creates
a new conversation and returns its ID in the response.
"""
from __future__ import annotations
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from src.api.schemas.chat.chat_request import ChatRequest
from src.api.schemas.chat.chat_response import ChatResponse
from src.api.schemas.common.response_envelope import SuccessResponse
from src.api.sse.event_emitter import SSEEventEmitter
from src.entities.chat_result.chat_result import ChatResult
from src.providers import get_process_chat_service

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("")
async def chat(
    body: ChatRequest,
    request: Request,
    service=Depends(get_process_chat_service),
):
    """
    Process a chat message.

    - conversation_id=null → auto-create a new conversation
    - stream=false → JSON response with full answer
    - stream=true  → text/event-stream SSE response
    """
    correlation_id = getattr(request.state, "correlation_id", "")

    result = await service.execute(
        conversation_id=str(body.conversation_id) if body.conversation_id else None,
        user_message=body.message,
        model_id=str(body.model_id) if body.model_id else None,
        system_prompt_id=str(body.system_prompt_id) if body.system_prompt_id else None,
        stream=body.stream,
        correlation_id=correlation_id,
    )

    if body.stream:
        return StreamingResponse(
            SSEEventEmitter.stream_from_service(result),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Correlation-ID": correlation_id,
            },
        )

    assert isinstance(result, ChatResult)
    response = ChatResponse(
        conversation_id=result.conversation_id,
        message_id=result.message_id,
        answer=result.answer,
        sources=result.sources,
        model_name=result.model_name,
        system_prompt_id=result.system_prompt_id,
        created_at=result.created_at,
    )
    return SuccessResponse(data=response)