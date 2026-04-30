"""
ResolveModelService — validates and retrieves models from the catalog.

Provides two operations: list all active models, and resolve one by ID.
Raises typed exceptions on invalid or inactive model selection.
"""
from __future__ import annotations

from src.core.exceptions.domain import ModelNotActiveError
from src.core.exceptions.not_found import ModelNotFoundError
from src.core.logging.logger import get_logger
from src.entities.available_model.available_model import AvailableModel
from src.interfaces.repositories.base_available_model_repository import BaseAvailableModelRepository


class ResolveModelService:
    """
    Service for model catalog operations.

    Args:
        model_repo: Repository for available_models table.
    """

    def __init__(self, model_repo: BaseAvailableModelRepository) -> None:
        self._model_repo = model_repo
        self._logger = get_logger(__name__)

    async def execute_get_all(self) -> list[AvailableModel]:
        """
        Return all active models for the listing endpoint.

        Returns:
            List of active AvailableModel entities ordered by display_name.
        """
        return await self._model_repo.get_all_active()

    async def execute_resolve(self, model_id: str) -> AvailableModel:
        """
        Validate that a model_id exists and is active.

        Args:
            model_id: UUID of the model to resolve.

        Returns:
            AvailableModel entity.

        Raises:
            ModelNotFoundError: If the model_id does not exist.
            ModelNotActiveError: If the model exists but is not active.
        """
        model = await self._model_repo.get_by_id(model_id)
        if model is None:
            raise ModelNotFoundError(model_id)
        if not model.is_available():
            raise ModelNotActiveError(model_id)
        return model
