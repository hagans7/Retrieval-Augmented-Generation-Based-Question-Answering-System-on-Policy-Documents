"""Initial schema with all tables, indexes, and seed data.

Revision ID: 0001
Revises:
Create Date: 2025-01-01 00:00:00
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ── Create enum types ───────────────────────────────────

    # ── users ────────────────────────────────────────────────
    op.create_table(
        "users",
        sa.Column("user_id", UUID(as_uuid=False), primary_key=True),
        sa.Column("role", sa.Enum("user", "admin", "supervisor", name="user_role"),
                  nullable=False, server_default="user"),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
    )

    # ── available_models ─────────────────────────────────────
    op.create_table(
        "available_models",
        sa.Column("model_id", UUID(as_uuid=False), primary_key=True),
        sa.Column("model_name", sa.Text, nullable=False, unique=True),
        sa.Column("provider", sa.Text, nullable=False),
        sa.Column("display_name", sa.Text, nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("context_window", sa.Integer, nullable=True),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_available_models_is_active", "available_models", ["is_active"])
    op.create_index("ix_available_models_provider", "available_models", ["provider"])

    # ── system_prompts ───────────────────────────────────────
    op.create_table(
        "system_prompts",
        sa.Column("system_prompt_id", UUID(as_uuid=False), primary_key=True),
        sa.Column("user_id", UUID(as_uuid=False), sa.ForeignKey("users.user_id", ondelete="SET NULL"), nullable=True),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("is_default", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_system_prompts_user_id", "system_prompts", ["user_id"])
    op.create_index("ix_system_prompts_is_default", "system_prompts", ["is_default"])
    op.create_index("ix_system_prompts_is_active", "system_prompts", ["is_active"])

    # ── conversations ────────────────────────────────────────
    op.create_table(
        "conversations",
        sa.Column("conversation_id", UUID(as_uuid=False), primary_key=True),
        sa.Column("user_id", UUID(as_uuid=False), sa.ForeignKey("users.user_id", ondelete="SET NULL"), nullable=True),
        sa.Column("title", sa.Text, nullable=False, server_default="New Conversation"),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_conversations_user_id", "conversations", ["user_id"])
    op.create_index("ix_conversations_created_at", "conversations", ["created_at"])
    op.create_index("ix_conversations_is_active", "conversations", ["is_active"])

    # ── messages ─────────────────────────────────────────────
    op.create_table(
        "messages",
        sa.Column("message_id", UUID(as_uuid=False), primary_key=True),
        sa.Column("conversation_id", UUID(as_uuid=False),
                  sa.ForeignKey("conversations.conversation_id", ondelete="CASCADE"), nullable=False),
        sa.Column("role", sa.Enum("user", "assistant", name="message_role"), nullable=False),
        sa.Column("content", sa.Text, nullable=False),
        sa.Column("system_prompt_id", UUID(as_uuid=False),
                  sa.ForeignKey("system_prompts.system_prompt_id", ondelete="SET NULL"), nullable=True),
        sa.Column("model_id", UUID(as_uuid=False),
                  sa.ForeignKey("available_models.model_id", ondelete="SET NULL"), nullable=True),
        sa.Column("token_count", sa.Integer, nullable=True),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_messages_conversation_id", "messages", ["conversation_id"])
    op.create_index("ix_messages_created_at", "messages", ["created_at"])

    # ── documents ────────────────────────────────────────────
    op.create_table(
        "documents",
        sa.Column("document_id", UUID(as_uuid=False), primary_key=True),
        sa.Column("user_id", UUID(as_uuid=False), sa.ForeignKey("users.user_id", ondelete="SET NULL"), nullable=True),
        sa.Column("file_name", sa.Text, nullable=False),
        sa.Column("file_type", sa.Text, nullable=False),
        sa.Column("storage_key", sa.Text, nullable=False, unique=True),
        sa.Column("file_size_bytes", sa.BigInteger, nullable=True),
        sa.Column("ingestion_status", sa.Enum("pending", "processing", "completed", "failed",
                  name="ingestion_status"),
                  nullable=False, server_default="pending"),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("chunk_count", sa.Integer, nullable=True),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.TIMESTAMP(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_documents_user_id", "documents", ["user_id"])
    op.create_index("ix_documents_ingestion_status", "documents", ["ingestion_status"])
    op.create_index("ix_documents_created_at", "documents", ["created_at"])

    # ── Seed: default user ────────────────────────────────────
    default_user_id = os.environ.get("DEFAULT_USER_ID", str(uuid.uuid4()))
    op.execute(
        f"INSERT INTO users (user_id, role, is_active) "
        f"VALUES ('{default_user_id}', 'admin', true) "
        f"ON CONFLICT (user_id) DO NOTHING"
    )

    # ── Seed: default available models ────────────────────────
    models = [
        (str(uuid.uuid4()), "qwen/qwen3-6b-plus:free", "openrouter", "Qwen 3 6B (Free)", 32768),
        (str(uuid.uuid4()), "google/gemma-3-12b-it:free", "openrouter", "Gemma 3 12B (Free)", 8192),
        (str(uuid.uuid4()), "meta-llama/llama-3.1-8b-instruct:free", "openrouter", "Llama 3.1 8B (Free)", 131072),
    ]
    for model_id, model_name, provider, display_name, ctx in models:
        op.execute(
            f"INSERT INTO available_models (model_id, model_name, provider, display_name, context_window) "
            f"VALUES ('{model_id}', '{model_name}', '{provider}', '{display_name}', {ctx}) "
            f"ON CONFLICT (model_name) DO NOTHING"
        )

    # ── Seed: default system prompt ───────────────────────────
    prompt_id = str(uuid.uuid4())
    op.execute(
        f"INSERT INTO system_prompts (system_prompt_id, user_id, name, content, is_default) "
        f"VALUES ('{prompt_id}', '{default_user_id}', 'Default Assistant', "
        f"'Kamu adalah asisten AI yang membantu dalam menganalisis dan menjelaskan dokumen hukum Indonesia. "
        f"Berikan jawaban yang akurat, jelas, dan berbasis pada dokumen yang tersedia.', true) "
        f"ON CONFLICT DO NOTHING"
    )


def downgrade() -> None:
    op.drop_table("documents")
    op.drop_table("messages")
    op.drop_table("conversations")
    op.drop_table("system_prompts")
    op.drop_table("available_models")
    op.drop_table("users")
    op.execute("DROP TYPE IF EXISTS ingestion_status")
    op.execute("DROP TYPE IF EXISTS message_role")
    op.execute("DROP TYPE IF EXISTS user_role")
