#!/bin/bash
set -e

echo "[entrypoint] Starting legal API..."

# ── Wait for PostgreSQL ──────────────────────────────────────
echo "[entrypoint] Waiting for PostgreSQL..."
until pg_isready -h "${PGHOST:-db}" -p "${PGPORT:-5432}" -U "${PGUSER:-legal}" -d "${PGDATABASE:-legal}" 2>/dev/null; do
    echo "[entrypoint] PostgreSQL not ready — retrying in 2s..."
    sleep 2
done
echo "[entrypoint] PostgreSQL is ready."

# ── Run Alembic migrations ───────────────────────────────────
echo "[entrypoint] Running database migrations..."
alembic upgrade head
echo "[entrypoint] Migrations complete."

# ── Start Uvicorn ────────────────────────────────────────────
echo "[entrypoint] Starting Uvicorn..."
exec uvicorn src.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --loop asyncio \
    --log-level info
