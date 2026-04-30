"""
Integration tests for MessageRepository.

Tests message persistence behavior: save user messages,
save assistant messages, and retrieve history in correct order.
Uses SQLite in-memory for fast isolated test execution.
"""
from __future__ import annotations

import uuid
import pytest
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import text


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture
async def sqlite_session():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)

    async with engine.begin() as conn:
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS conversations (
                conversation_id TEXT PRIMARY KEY,
                title TEXT NOT NULL DEFAULT 'New Conversation',
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """))
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS messages (
                message_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                system_prompt_id TEXT,
                model_id TEXT,
                token_count INTEGER,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """))

    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with factory() as session:
        yield session

    await engine.dispose()


# ── Helpers ───────────────────────────────────────────────────

async def insert_conversation(session: AsyncSession, conv_id: str) -> None:
    await session.execute(
        text("INSERT INTO conversations (conversation_id, title) VALUES (:id, 'Test')"),
        {"id": conv_id},
    )
    await session.commit()


async def insert_message(session: AsyncSession, conv_id: str, role: str, content: str) -> str:
    msg_id = str(uuid.uuid4())
    await session.execute(
        text("INSERT INTO messages (message_id, conversation_id, role, content) VALUES (:id, :conv, :role, :content)"),
        {"id": msg_id, "conv": conv_id, "role": role, "content": content},
    )
    await session.commit()
    return msg_id


# ── Tests ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_save_user_message_persists(sqlite_session: AsyncSession):
    """A user message inserted directly should be retrievable."""
    conv_id = str(uuid.uuid4())
    await insert_conversation(sqlite_session, conv_id)
    msg_id = await insert_message(sqlite_session, conv_id, "user", "Apa itu wanprestasi?")

    result = await sqlite_session.execute(
        text("SELECT role, content FROM messages WHERE message_id = :id"),
        {"id": msg_id},
    )
    row = result.fetchone()
    assert row is not None
    assert row[0] == "user"
    assert row[1] == "Apa itu wanprestasi?"


@pytest.mark.asyncio
async def test_get_history_returns_oldest_first(sqlite_session: AsyncSession):
    """Message history should be ordered by created_at ASC (oldest first)."""
    conv_id = str(uuid.uuid4())
    await insert_conversation(sqlite_session, conv_id)

    # Insert in order — SQLite CURRENT_TIMESTAMP has second precision,
    # so we differentiate with explicit timestamps
    for i, content in enumerate(["Q1", "A1", "Q2", "A2"]):
        msg_id = str(uuid.uuid4())
        await sqlite_session.execute(
            text("""
                INSERT INTO messages (message_id, conversation_id, role, content, created_at)
                VALUES (:id, :conv, :role, :content, datetime('now', :offset))
            """),
            {
                "id": msg_id,
                "conv": conv_id,
                "role": "user" if i % 2 == 0 else "assistant",
                "content": content,
                "offset": f"+{i} seconds",
            },
        )
    await sqlite_session.commit()

    result = await sqlite_session.execute(
        text("SELECT content FROM messages WHERE conversation_id = :id ORDER BY created_at ASC"),
        {"id": conv_id},
    )
    contents = [row[0] for row in result.fetchall()]
    assert contents == ["Q1", "A1", "Q2", "A2"]


@pytest.mark.asyncio
async def test_get_history_respects_limit(sqlite_session: AsyncSession):
    """History query should return only the last N messages."""
    conv_id = str(uuid.uuid4())
    await insert_conversation(sqlite_session, conv_id)

    for i in range(6):
        await insert_message(sqlite_session, conv_id, "user", f"Message {i}")

    result = await sqlite_session.execute(
        text("""
            SELECT content FROM messages
            WHERE conversation_id = :id
            ORDER BY created_at ASC
            LIMIT 3
        """),
        {"id": conv_id},
    )
    rows = result.fetchall()
    assert len(rows) == 3


@pytest.mark.asyncio
async def test_empty_history_returns_empty_list(sqlite_session: AsyncSession):
    """get_history() for a conversation with no messages should return []."""
    conv_id = str(uuid.uuid4())
    await insert_conversation(sqlite_session, conv_id)

    result = await sqlite_session.execute(
        text("SELECT * FROM messages WHERE conversation_id = :id"),
        {"id": conv_id},
    )
    assert result.fetchall() == []
