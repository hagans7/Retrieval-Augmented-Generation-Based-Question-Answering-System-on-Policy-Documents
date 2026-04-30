"""
Message repository.

No begin() — uses implicit autobegin transaction from SQLAlchemy 2.0.
"""
from __future__ import annotations
import uuid
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.exceptions.domain import PersistenceError
from src.core.logging.logger import get_logger
from src.db_models.message.message_orm import MessageORM
from src.entities.message.message import Message
from src.interfaces.repositories.base_message_repository import BaseMessageRepository
from src.repositories.message.message_mapper import to_entity


class MessageRepository(BaseMessageRepository):
    def __init__(self, db_session: AsyncSession) -> None:
        self._session = db_session
        self._logger = get_logger(__name__)

    async def save_user_message(self, conversation_id, content, system_prompt_id, model_id) -> Message:
        return await self._save(conversation_id, "user", content, None, None, None)

    async def save_assistant_message(self, conversation_id, content, system_prompt_id, model_id, token_count) -> Message:
        return await self._save(conversation_id, "assistant", content, system_prompt_id, model_id, token_count)

    async def _save(self, conversation_id, role, content, system_prompt_id, model_id, token_count) -> Message:
        orm = MessageORM(
            message_id=str(uuid.uuid4()),
            conversation_id=conversation_id,
            role=role,
            content=content,
            system_prompt_id=system_prompt_id,
            model_id=model_id,
            token_count=token_count,
        )
        try:
            self._session.add(orm)
            await self._session.commit()
            await self._session.refresh(orm)
            return to_entity(orm)
        except SQLAlchemyError as exc:
            await self._session.rollback()
            raise PersistenceError(
                message="Failed to save message.",
                context={"role": role, "error": str(exc)},
            ) from exc

    async def get_history(self, conversation_id: str, limit: int) -> list[Message]:
        try:
            stmt = (
                select(MessageORM)
                .where(MessageORM.conversation_id == conversation_id)
                .order_by(MessageORM.created_at.asc())
            )
            result = await self._session.execute(stmt)
            all_messages = result.scalars().all()
            recent = all_messages[-limit:] if limit > 0 else all_messages
            return [to_entity(m) for m in recent]
        except SQLAlchemyError as exc:
            raise PersistenceError(
                message="Failed to get message history.",
                context={"conversation_id": conversation_id, "error": str(exc)},
            ) from exc