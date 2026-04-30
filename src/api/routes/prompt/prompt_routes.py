"""System prompt management routes."""
from __future__ import annotations
from fastapi import APIRouter, Depends, status
from src.api.schemas.common.response_envelope import SuccessResponse
from src.api.schemas.prompt.prompt_request import CreatePromptRequest, UpdatePromptRequest
from src.api.schemas.prompt.prompt_response import SystemPromptResponse
from src.core.config.settings import settings
from src.entities.system_prompt.system_prompt import SystemPrompt
from src.providers import get_manage_prompt_service

router = APIRouter(prefix="/prompts", tags=["prompts"])


def _to_response(p: SystemPrompt) -> SystemPromptResponse:
    return SystemPromptResponse(
        system_prompt_id=p.system_prompt_id, user_id=p.user_id, name=p.name,
        content=p.content, is_default=p.is_default, is_active=p.is_active,
        created_at=p.created_at, updated_at=p.updated_at,
    )


@router.post("", response_model=SuccessResponse[SystemPromptResponse], status_code=status.HTTP_201_CREATED)
async def create_prompt(body: CreatePromptRequest, service=Depends(get_manage_prompt_service)):
    user_id = settings.DEFAULT_USER_ID or None
    prompt = await service.execute_create(user_id=user_id, name=body.name, content=body.content, is_default=body.is_default)
    return SuccessResponse(data=_to_response(prompt))


@router.get("", response_model=SuccessResponse[list[SystemPromptResponse]])
async def list_prompts(service=Depends(get_manage_prompt_service)):
    user_id = settings.DEFAULT_USER_ID or None
    prompts = await service.execute_list(user_id)
    return SuccessResponse(data=[_to_response(p) for p in prompts])


@router.get("/{system_prompt_id}", response_model=SuccessResponse[SystemPromptResponse])
async def get_prompt(system_prompt_id: str, service=Depends(get_manage_prompt_service)):
    user_id = settings.DEFAULT_USER_ID or None
    prompt = await service.execute_get(system_prompt_id, user_id)
    return SuccessResponse(data=_to_response(prompt))


@router.patch("/{system_prompt_id}", response_model=SuccessResponse[SystemPromptResponse])
async def update_prompt(system_prompt_id: str, body: UpdatePromptRequest, service=Depends(get_manage_prompt_service)):
    user_id = settings.DEFAULT_USER_ID or None
    prompt = await service.execute_update(system_prompt_id, user_id, body.name, body.content)
    return SuccessResponse(data=_to_response(prompt))


@router.post("/{system_prompt_id}/set-default", response_model=SuccessResponse[dict])
async def set_default_prompt(system_prompt_id: str, service=Depends(get_manage_prompt_service)):
    user_id = settings.DEFAULT_USER_ID or None
    await service.execute_set_default(system_prompt_id, user_id)
    return SuccessResponse(data={"message": "Default prompt updated."})


@router.delete("/{system_prompt_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_prompt(system_prompt_id: str, service=Depends(get_manage_prompt_service)):
    user_id = settings.DEFAULT_USER_ID or None
    await service.execute_delete(system_prompt_id, user_id)
