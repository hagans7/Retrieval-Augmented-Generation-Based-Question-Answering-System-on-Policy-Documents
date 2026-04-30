"""AvailableModel mapper — ORM to entity."""
from __future__ import annotations
from src.db_models.available_model.available_model_orm import AvailableModelORM
from src.entities.available_model.available_model import AvailableModel

def to_entity(orm: AvailableModelORM) -> AvailableModel:
    return AvailableModel(
        model_id=orm.model_id,
        model_name=orm.model_name,
        provider=orm.provider,
        display_name=orm.display_name,
        description=orm.description,
        context_window=orm.context_window,
        is_active=orm.is_active,
        created_at=orm.created_at,
    )
