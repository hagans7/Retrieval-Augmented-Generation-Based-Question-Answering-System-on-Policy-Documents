"""Message mapper — converts MessageORM to Message entity."""
from __future__ import annotations
from src.db_models.message.message_orm import MessageORM
from src.entities.message.message import Message

def to_entity(orm: MessageORM) -> Message:
    return Message(
        message_id=orm.message_id,
        conversation_id=orm.conversation_id,
        role=orm.role,
        content=orm.content,
        system_prompt_id=orm.system_prompt_id,
        model_id=orm.model_id,
        token_count=orm.token_count,
        created_at=orm.created_at,
    )
