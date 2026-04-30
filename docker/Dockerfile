# ============================================================
# Stage 1: Builder
# Install dependencies via uv into a virtual environment.
# This layer is cached as long as pyproject.toml does not change.
# ============================================================
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1

# Install uv
RUN pip install uv==0.5.11 --no-cache-dir

WORKDIR /app

# Copy dependency manifest only — cache this layer
COPY pyproject.toml ./

# Install all runtime dependencies
RUN uv pip install --system -e ".[dev]" --no-cache

# ============================================================
# Stage 2: Runtime
# Lean image — only what is needed to run the application.
# ============================================================
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install curl for health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source
COPY src/ ./src/
COPY alembic/ ./alembic/
COPY alembic.ini ./alembic.ini
COPY docker/entrypoint.sh ./entrypoint.sh

RUN chmod +x ./entrypoint.sh

# Create non-root user
RUN groupadd -r legal && useradd -r -g legal legal
RUN mkdir -p /app/models && chown -R legal:legal /app

USER legal

EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]
