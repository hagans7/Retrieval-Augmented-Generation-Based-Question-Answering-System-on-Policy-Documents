"""
System prompt repository.

Transaction management: session lifecycle is owned by get_db_session().
Repositories call commit() + refresh() directly — no nested begin().

Why no begin():
    SQLAlchemy 2.0 autobegin — every session.execute() starts an implicit
    transaction. Calling session.begin() on a session that already has an
    implicit transaction open raises InvalidRequestError.
    Correct pattern: use the implicit transaction, call commit() when done.
"""
from __future__ import annotations
import uuid
from sqlalchemy import select, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.exceptions.domain import PersistenceError
from src.core.exceptions.not_found import SystemPromptNotFoundError
from src.core.logging.logger import get_logger
from src.db_models.system_prompt.system_prompt_orm import SystemPromptORM
from src.entities.system_prompt.system_prompt import SystemPrompt
from src.interfaces.repositories.base_system_prompt_repository import BaseSystemPromptRepository
from src.repositories.system_prompt.system_prompt_mapper import to_entity


class SystemPromptRepository(BaseSystemPromptRepository):
    def __init__(self, db_session: AsyncSession) -> None:
        self._session = db_session
        self._logger = get_logger(__name__)

    async def create(self, user_id, name, content, is_default) -> SystemPrompt:
        try:
            if is_default:
                await self._session.execute(
                    update(SystemPromptORM)
                    .where(SystemPromptORM.user_id == user_id, SystemPromptORM.is_active.is_(True))
                    .values(is_default=False)
                )
            orm = SystemPromptORM(
                system_prompt_id=str(uuid.uuid4()),
                user_id=user_id, name=name, content=content, is_default=is_default,
            )
            self._session.add(orm)
            await self._session.commit()
            await self._session.refresh(orm)
            return to_entity(orm)
        except SQLAlchemyError as exc:
            await self._session.rollback()
            raise PersistenceError(
                message="Failed to create system prompt.",
                context={"error": str(exc)},
            ) from exc

    async def get_by_id(self, system_prompt_id) -> SystemPrompt | None:
        try:
            stmt = select(SystemPromptORM).where(
                SystemPromptORM.system_prompt_id == system_prompt_id,
                SystemPromptORM.is_active.is_(True),
            )
            result = await self._session.execute(stmt)
            orm = result.scalar_one_or_none()
            return to_entity(orm) if orm else None
        except SQLAlchemyError as exc:
            raise PersistenceError(
                message="Failed to get system prompt.",
                context={"error": str(exc)},
            ) from exc

    async def get_all_by_user(self, user_id) -> list[SystemPrompt]:
        try:
            stmt = select(SystemPromptORM).where(
                SystemPromptORM.user_id == user_id,
                SystemPromptORM.is_active.is_(True),
            ).order_by(SystemPromptORM.created_at.desc())
            result = await self._session.execute(stmt)
            return [to_entity(r) for r in result.scalars().all()]
        except SQLAlchemyError as exc:
            raise PersistenceError(
                message="Failed to list system prompts.",
                context={"error": str(exc)},
            ) from exc

    async def get_default_for_user(self, user_id) -> SystemPrompt | None:
        try:
            stmt = select(SystemPromptORM).where(
                SystemPromptORM.user_id == user_id,
                SystemPromptORM.is_default.is_(True),
                SystemPromptORM.is_active.is_(True),
            ).limit(1)
            result = await self._session.execute(stmt)
            orm = result.scalar_one_or_none()
            return to_entity(orm) if orm else None
        except SQLAlchemyError as exc:
            raise PersistenceError(
                message="Failed to get default prompt.",
                context={"error": str(exc)},
            ) from exc

    async def update(self, system_prompt_id, name, content) -> SystemPrompt:
        try:
            values: dict = {}
            if name is not None:
                values["name"] = name
            if content is not None:
                values["content"] = content
            stmt = (
                update(SystemPromptORM)
                .where(
                    SystemPromptORM.system_prompt_id == system_prompt_id,
                    SystemPromptORM.is_active.is_(True),
                )
                .values(**values)
                .returning(SystemPromptORM)
            )
            result = await self._session.execute(stmt)
            orm = result.scalar_one_or_none()
            if not orm:
                raise SystemPromptNotFoundError(system_prompt_id)
            await self._session.commit()
            return to_entity(orm)
        except SystemPromptNotFoundError:
            raise
        except SQLAlchemyError as exc:
            await self._session.rollback()
            raise PersistenceError(
                message="Failed to update system prompt.",
                context={"error": str(exc)},
            ) from exc

    async def set_as_default(self, system_prompt_id, user_id) -> None:
        try:
            await self._session.execute(
                update(SystemPromptORM)
                .where(SystemPromptORM.user_id == user_id)
                .values(is_default=False)
            )
            result = await self._session.execute(
                update(SystemPromptORM)
                .where(SystemPromptORM.system_prompt_id == system_prompt_id)
                .values(is_default=True)
            )
            if result.rowcount == 0:
                raise SystemPromptNotFoundError(system_prompt_id)
            await self._session.commit()
        except SystemPromptNotFoundError:
            raise
        except SQLAlchemyError as exc:
            await self._session.rollback()
            raise PersistenceError(
                message="Failed to set default prompt.",
                context={"error": str(exc)},
            ) from exc

    async def soft_delete(self, system_prompt_id) -> None:
        try:
            result = await self._session.execute(
                update(SystemPromptORM)
                .where(SystemPromptORM.system_prompt_id == system_prompt_id)
                .values(is_active=False)
            )
            if result.rowcount == 0:
                raise SystemPromptNotFoundError(system_prompt_id)
            await self._session.commit()
        except SystemPromptNotFoundError:
            raise
        except SQLAlchemyError as exc:
            await self._session.rollback()
            raise PersistenceError(
                message="Failed to delete system prompt.",
                context={"error": str(exc)},
            ) from exc