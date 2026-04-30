"""
Conversation mapper — converts ConversationORM to Conversation entity.

Only this file knows both the ORM model and the domain entity.
Service layer only ever sees the Conversation entity.
"""

from __future__ import annotations

from src.db_models.conversation.conversation_orm import ConversationORM
from src.entities.conversation.conversation import Conversation


def to_entity(orm: ConversationORM) -> Conversation:
    """
    Map a ConversationORM instance to a Conversation domain entity.

    The messages field is populated as an empty list by default.
    When message history is needed, use message_repository.get_history().

    Args:
        orm: ConversationORM SQLAlchemy model instance.

    Returns:
        Conversation domain entity.
    """
    return Conversation(
        conversation_id=orm.conversation_id,
        user_id=orm.user_id,
        title=orm.title,
        is_active=orm.is_active,
        created_at=orm.created_at,
        updated_at=orm.updated_at,
        messages=[],
    )
