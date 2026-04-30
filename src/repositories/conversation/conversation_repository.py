"""
Conversation repository.

No begin() — uses implicit autobegin transaction from SQLAlchemy 2.0.
commit() is called explicitly after writes. Session lifecycle owned by get_db_session().
"""
from __future__ import annotations
import uuid
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.exceptions.domain import DuplicateResourceError, PersistenceError
from src.core.exceptions.not_found import ConversationNotFoundError
from src.core.logging.logger import get_logger
from src.db_models.conversation.conversation_orm import ConversationORM
from src.entities.conversation.conversation import Conversation
from src.interfaces.repositories.base_conversation_repository import BaseConversationRepository
from src.repositories.conversation.conversation_mapper import to_entity


class ConversationRepository(BaseConversationRepository):
    def __init__(self, db_session: AsyncSession) -> None:
        self._session = db_session
        self._logger = get_logger(__name__)

    async def create(self, user_id: str | None, title: str) -> Conversation:
        orm = ConversationORM(
            conversation_id=str(uuid.uuid4()),
            user_id=user_id,
            title=title,
        )
        try:
            self._session.add(orm)
            await self._session.commit()
            await self._session.refresh(orm)
            self._logger.debug("Conversation created", extra={"conversation_id": orm.conversation_id})
            return to_entity(orm)
        except IntegrityError as exc:
            await self._session.rollback()
            raise DuplicateResourceError("conversation", orm.conversation_id) from exc
        except SQLAlchemyError as exc:
            await self._session.rollback()
            raise PersistenceError(
                message="Failed to create conversation.",
                context={"error": str(exc)},
            ) from exc

    async def get_by_id(self, conversation_id: str) -> Conversation | None:
        try:
            stmt = select(ConversationORM).where(
                ConversationORM.conversation_id == conversation_id,
                ConversationORM.is_active.is_(True),
            )
            result = await self._session.execute(stmt)
            orm = result.scalar_one_or_none()
            return to_entity(orm) if orm else None
        except SQLAlchemyError as exc:
            raise PersistenceError(
                message=f"Failed to retrieve conversation '{conversation_id}'.",
                context={"conversation_id": conversation_id, "error": str(exc)},
            ) from exc

    async def get_all_by_user(self, user_id: str | None, limit: int, offset: int) -> list[Conversation]:
        try:
            stmt = (
                select(ConversationORM)
                .where(ConversationORM.user_id == user_id, ConversationORM.is_active.is_(True))
                .order_by(ConversationORM.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            result = await self._session.execute(stmt)
            return [to_entity(row) for row in result.scalars().all()]
        except SQLAlchemyError as exc:
            raise PersistenceError(
                message="Failed to list conversations.",
                context={"user_id": user_id, "error": str(exc)},
            ) from exc

    async def update_title(self, conversation_id: str, title: str) -> Conversation:
        try:
            stmt = (
                update(ConversationORM)
                .where(
                    ConversationORM.conversation_id == conversation_id,
                    ConversationORM.is_active.is_(True),
                )
                .values(title=title)
                .returning(ConversationORM)
            )
            result = await self._session.execute(stmt)
            orm = result.scalar_one_or_none()
            if not orm:
                raise ConversationNotFoundError(conversation_id)
            await self._session.commit()
            return to_entity(orm)
        except ConversationNotFoundError:
            raise
        except SQLAlchemyError as exc:
            await self._session.rollback()
            raise PersistenceError(
                message=f"Failed to update title for conversation '{conversation_id}'.",
                context={"conversation_id": conversation_id, "error": str(exc)},
            ) from exc

    async def soft_delete(self, conversation_id: str) -> None:
        try:
            stmt = (
                update(ConversationORM)
                .where(
                    ConversationORM.conversation_id == conversation_id,
                    ConversationORM.is_active.is_(True),
                )
                .values(is_active=False)
            )
            result = await self._session.execute(stmt)
            if result.rowcount == 0:
                raise ConversationNotFoundError(conversation_id)
            await self._session.commit()
        except ConversationNotFoundError:
            raise
        except SQLAlchemyError as exc:
            await self._session.rollback()
            raise PersistenceError(
                message=f"Failed to soft-delete conversation '{conversation_id}'.",
                context={"conversation_id": conversation_id, "error": str(exc)},
            ) from exc