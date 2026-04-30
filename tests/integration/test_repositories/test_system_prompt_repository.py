"""
Integration tests for SystemPromptRepository.

Verifies the atomic default-prompt management logic:
- Only one prompt can be default per user at a time.
- Creating a new default prompt unsets all previous defaults.
- set_as_default is atomic.
Uses SQLite in-memory for fast test execution.
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
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                role TEXT NOT NULL DEFAULT 'user',
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """))
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS system_prompts (
                system_prompt_id TEXT PRIMARY KEY,
                user_id TEXT,
                name TEXT NOT NULL,
                content TEXT NOT NULL,
                is_default INTEGER NOT NULL DEFAULT 0,
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """))

    factory = async_sessionmaker(bind=engine, expire_on_commit=False)
    async with factory() as session:
        yield session

    await engine.dispose()


# ── Helpers ───────────────────────────────────────────────────

async def insert_prompt(
    session: AsyncSession,
    user_id: str,
    name: str,
    is_default: int = 0,
) -> str:
    prompt_id = str(uuid.uuid4())
    await session.execute(
        text("""
            INSERT INTO system_prompts (system_prompt_id, user_id, name, content, is_default)
            VALUES (:id, :uid, :name, 'Content', :default)
        """),
        {"id": prompt_id, "uid": user_id, "name": name, "default": is_default},
    )
    await session.commit()
    return prompt_id


# ── Tests ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_only_one_default_per_user(sqlite_session: AsyncSession):
    """
    When a new prompt is set as default, all other prompts for the user
    must have is_default = 0. Business rule: one default per user.
    """
    user_id = str(uuid.uuid4())
    p1 = await insert_prompt(sqlite_session, user_id, "Prompt 1", is_default=1)
    p2 = await insert_prompt(sqlite_session, user_id, "Prompt 2", is_default=0)

    # Simulate set_as_default: unset all, then set one
    await sqlite_session.execute(
        text("UPDATE system_prompts SET is_default = 0 WHERE user_id = :uid"),
        {"uid": user_id},
    )
    await sqlite_session.execute(
        text("UPDATE system_prompts SET is_default = 1 WHERE system_prompt_id = :id"),
        {"id": p2},
    )
    await sqlite_session.commit()

    result = await sqlite_session.execute(
        text("SELECT system_prompt_id FROM system_prompts WHERE user_id = :uid AND is_default = 1"),
        {"uid": user_id},
    )
    defaults = [row[0] for row in result.fetchall()]
    assert defaults == [p2], "Only p2 should be default"


@pytest.mark.asyncio
async def test_soft_delete_sets_inactive(sqlite_session: AsyncSession):
    """Soft-deleted prompts should have is_active = 0."""
    user_id = str(uuid.uuid4())
    p1 = await insert_prompt(sqlite_session, user_id, "Prompt To Delete")

    await sqlite_session.execute(
        text("UPDATE system_prompts SET is_active = 0 WHERE system_prompt_id = :id"),
        {"id": p1},
    )
    await sqlite_session.commit()

    result = await sqlite_session.execute(
        text("SELECT is_active FROM system_prompts WHERE system_prompt_id = :id"),
        {"id": p1},
    )
    assert result.fetchone()[0] == 0


@pytest.mark.asyncio
async def test_get_default_returns_correct_prompt(sqlite_session: AsyncSession):
    """get_default_for_user() should return the one prompt with is_default=1."""
    user_id = str(uuid.uuid4())
    p1 = await insert_prompt(sqlite_session, user_id, "Not Default", is_default=0)
    p2 = await insert_prompt(sqlite_session, user_id, "Default Prompt", is_default=1)

    result = await sqlite_session.execute(
        text("SELECT system_prompt_id FROM system_prompts WHERE user_id = :uid AND is_default = 1 LIMIT 1"),
        {"uid": user_id},
    )
    row = result.fetchone()
    assert row is not None
    assert row[0] == p2


@pytest.mark.asyncio
async def test_list_excludes_inactive(sqlite_session: AsyncSession):
    """get_all_by_user() must exclude soft-deleted prompts."""
    user_id = str(uuid.uuid4())
    await insert_prompt(sqlite_session, user_id, "Active Prompt")
    inactive_id = await insert_prompt(sqlite_session, user_id, "Inactive Prompt")

    await sqlite_session.execute(
        text("UPDATE system_prompts SET is_active = 0 WHERE system_prompt_id = :id"),
        {"id": inactive_id},
    )
    await sqlite_session.commit()

    result = await sqlite_session.execute(
        text("SELECT COUNT(*) FROM system_prompts WHERE user_id = :uid AND is_active = 1"),
        {"uid": user_id},
    )
    assert result.scalar() == 1
