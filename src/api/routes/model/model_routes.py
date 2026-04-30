"""Model catalog routes — read-only."""
from __future__ import annotations
from fastapi import APIRouter, Depends
from src.api.schemas.common.response_envelope import SuccessResponse
from src.api.schemas.model.model_response import ModelResponse
from src.entities.available_model.available_model import AvailableModel
from src.providers import get_resolve_model_service

router = APIRouter(prefix="/models", tags=["models"])


def _to_response(m: AvailableModel) -> ModelResponse:
    return ModelResponse(
        model_id=m.model_id, model_name=m.model_name, provider=m.provider,
        display_name=m.display_name, description=m.description,
        context_window=m.context_window, is_active=m.is_active, created_at=m.created_at,
    )


@router.get("", response_model=SuccessResponse[list[ModelResponse]])
async def list_models(service=Depends(get_resolve_model_service)):
    models = await service.execute_get_all()
    return SuccessResponse(data=[_to_response(m) for m in models])


@router.get("/{model_id}", response_model=SuccessResponse[ModelResponse])
async def get_model(model_id: str, service=Depends(get_resolve_model_service)):
    model = await service.execute_resolve(model_id)
    return SuccessResponse(data=_to_response(model))
