"""
Available model repository — implements BaseAvailableModelRepository.

Read-only from the application's perspective; models are managed
via Alembic seeding and direct DB operations only.
"""
from __future__ import annotations
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.exceptions.domain import PersistenceError
from src.core.logging.logger import get_logger
from src.db_models.available_model.available_model_orm import AvailableModelORM
from src.entities.available_model.available_model import AvailableModel
from src.interfaces.repositories.base_available_model_repository import BaseAvailableModelRepository
from src.repositories.available_model.available_model_mapper import to_entity


class AvailableModelRepository(BaseAvailableModelRepository):
    def __init__(self, db_session: AsyncSession) -> None:
        self._session = db_session
        self._logger = get_logger(__name__)

    async def get_by_id(self, model_id: str) -> AvailableModel | None:
        try:
            stmt = select(AvailableModelORM).where(AvailableModelORM.model_id == model_id)
            result = await self._session.execute(stmt)
            orm = result.scalar_one_or_none()
            return to_entity(orm) if orm else None
        except SQLAlchemyError as exc:
            raise PersistenceError(
                message=f"Failed to retrieve model '{model_id}'.",
                context={"model_id": model_id, "error": str(exc)},
            ) from exc

    async def get_all_active(self) -> list[AvailableModel]:
        try:
            stmt = (
                select(AvailableModelORM)
                .where(AvailableModelORM.is_active.is_(True))
                .order_by(AvailableModelORM.display_name)
            )
            result = await self._session.execute(stmt)
            return [to_entity(r) for r in result.scalars().all()]
        except SQLAlchemyError as exc:
            raise PersistenceError(
                message="Failed to list active models.",
                context={"error": str(exc)},
            ) from exc

    async def get_by_model_name(self, model_name: str) -> AvailableModel | None:
        try:
            stmt = select(AvailableModelORM).where(AvailableModelORM.model_name == model_name)
            result = await self._session.execute(stmt)
            orm = result.scalar_one_or_none()
            return to_entity(orm) if orm else None
        except SQLAlchemyError as exc:
            raise PersistenceError(
                message=f"Failed to retrieve model by name '{model_name}'.",
                context={"model_name": model_name, "error": str(exc)},
            ) from exc
