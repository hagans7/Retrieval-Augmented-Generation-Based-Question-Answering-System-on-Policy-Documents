"""Abstract base for available model repositories."""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.entities.available_model.available_model import AvailableModel


class BaseAvailableModelRepository(ABC):
    """Contract for model catalog read access."""

    @abstractmethod
    async def get_by_id(self, model_id: str) -> AvailableModel | None:
        """Return model by UUID, or None if not found."""
        ...

    @abstractmethod
    async def get_all_active(self) -> list[AvailableModel]:
        """Return all active models ordered by display_name."""
        ...

    @abstractmethod
    async def get_by_model_name(self, model_name: str) -> AvailableModel | None:
        """Return model by technical name (e.g. 'qwen/qwen3-6b-plus:free'), or None."""
        ...
