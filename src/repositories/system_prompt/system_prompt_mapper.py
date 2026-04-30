"""SystemPrompt mapper."""
from __future__ import annotations
from src.db_models.system_prompt.system_prompt_orm import SystemPromptORM
from src.entities.system_prompt.system_prompt import SystemPrompt

def to_entity(orm: SystemPromptORM) -> SystemPrompt:
    return SystemPrompt(
        system_prompt_id=orm.system_prompt_id,
        user_id=orm.user_id,
        name=orm.name,
        content=orm.content,
        is_default=orm.is_default,
        is_active=orm.is_active,
        created_at=orm.created_at,
        updated_at=orm.updated_at,
    )
