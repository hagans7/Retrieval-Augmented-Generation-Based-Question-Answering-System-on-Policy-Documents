# ⚖️ legal

> **GraphRAG-powered backend for Indonesian legal document analysis** — combining vector search, knowledge graphs, and agentic AI orchestration.

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-agentic-FF6B6B?style=flat-square)](https://github.com/langchain-ai/langgraph)
[![Neo4j](https://img.shields.io/badge/Neo4j-graph_db-008CC1?style=flat-square&logo=neo4j&logoColor=white)](https://neo4j.com)
[![Weaviate](https://img.shields.io/badge/Weaviate-vector_db-green?style=flat-square)](https://weaviate.io)
[![License](https://img.shields.io/badge/license-MIT-blue?style=flat-square)](LICENSE)

---

## What is this?

`legal` is an AI backend that lets you **upload Indonesian legal documents and have intelligent conversations about them**. It's not just a RAG chatbot — it uses a knowledge graph to understand relationships between legal entities (laws, articles, parties, obligations) so answers are contextual, structured, and citable.

### 🎯 Use Cases

| Domain | Example Questions |
|--------|------------------|
| **Contract Review** | *"Does this contract violate any KUHPerdata provisions?"* |
| **Legal Research** | *"How has Article 1338 been interpreted across case law?"* |
| **Compliance** | *"Which regulations apply to this business activity?"* |
| **Due Diligence** | *"Summarize the obligations and liabilities in this agreement."* |
| **Legal Education** | *"Explain wanprestasi and its legal consequences."* |
| **Document Comparison** | *"How does this draft differ from the standard template?"* |

Built for **Indonesian legal context** (KUHPerdata, KUHPidana, peraturan, putusan, kontrak), but the architecture generalizes to any structured legal corpus.

---

## 🧠 How It Works

### Document Storage Pipeline

When you upload a document, the system doesn't just chunk it blindly:

```
PDF / DOCX / TXT
      │
      ▼
  OCR + Text Extraction  (LightOnOCR-2-1B + Docling)
      │
      ▼
  Smart Chunking
  ├── Detects Indonesian legal structure (BAB / Pasal / Ayat)
  ├── Hierarchical chunking if structure is found
  └── Sliding window fallback (512 tokens, 128 overlap)
      │
      ┌────────────┴────────────┐
      ▼                         ▼
  Embedding                Entity Extraction
  BGE-M3 via TEI           LLM-based NER
      │                         │
      ▼                         ▼
  Weaviate                 Neo4j
  (vector search)          (knowledge graph)
```

### Chat Pipeline

Every question goes through a multi-step agentic loop:

```
User Question
      │
      ▼
  ROUTER — classify query type
  ├── simple   → direct LLM answer (no retrieval)
  └── complex  → planning pipeline
      │
      ▼
  PLANNER — break question into retrieval steps
      │
      ▼
  EXECUTOR — run tools
  ├── vector_search  → Weaviate hybrid search + BGE reranker
  ├── graph_query    → Neo4j multi-hop traversal
  └── hybrid_search  → both, merged and ranked
      │
      ▼
  AUDITOR — is evidence sufficient?
  ├── yes → generate answer
  └── no  → retry up to 2x, then proceed with low confidence flag
      │
      ▼
  GENERATOR — final answer with citations
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **API** | FastAPI + Uvicorn |
| **Agent** | LangGraph (StateGraph) |
| **LLM** | OpenRouter (Qwen3, configurable) |
| **Embeddings** | BGE-M3 via HuggingFace TEI |
| **Reranker** | BGE-Reranker-v2-M3 via TEI |
| **Vector DB** | Weaviate |
| **Knowledge Graph** | Neo4j |
| **OCR** | LightOnOCR-2-1B + Docling |
| **Async Tasks** | Celery + Redis |
| **Object Storage** | MinIO (S3-compatible) |
| **Database** | PostgreSQL + SQLAlchemy async |
| **Observability** | Langfuse |
| **Package Manager** | uv |

---

## ⚡ Quick Start

### Prerequisites

| Tool | Min Version | Check |
|------|------------|-------|
| Python | 3.12 | `python3 --version` |
| uv | 0.5+ | `uv --version` |
| Docker | 24+ | `docker --version` |
| Docker Compose | 2.20+ | `docker compose version` |

### 1 — Clone & Install

```bash
git clone <repo-url> legal && cd legal

# Create virtualenv and install all deps
uv venv --python 3.12
source .venv/bin/activate           # Windows: .venv\Scripts\Activate.ps1
uv pip install -e ".[dev]"
uv pip install aiosqlite            # needed for integration tests
```

> **Heads up:** `torch` + `transformers` are ~2 GB. First install takes 5–15 min depending on your connection.

Verify the install:

```bash
python -c "import fastapi, sqlalchemy, langfuse, langgraph, weaviate; print('✓ core deps OK')"
python -c "import celery, redis, httpx, tenacity; print('✓ runtime deps OK')"
```

### 2 — Configure Environment

```bash
cp .env.example .env
```

Open `.env` and fill in the required values:

```env
# Required
LLM_API_KEY=sk-or-v1-xxxxx          # get from openrouter.ai
NEO4J_PASSWORD=your_password_here
DEFAULT_USER_ID=$(python3 -c "import uuid; print(uuid.uuid4())")

# Optional
HF_TOKEN=                            # only for private HuggingFace repos
LANGFUSE_SECRET_KEY=                 # leave blank to skip observability
LANGFUSE_PUBLIC_KEY=
```

<details>
<summary>📋 Full environment variable reference</summary>

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_NAME` | `legal` | Application name |
| `DEPLOYMENT_MODE` | `development` | `development` or `production` |
| `DEBUG` | `false` | Enable SQL query logging |
| `DEFAULT_USER_ID` | *(required)* | UUID for the default user |
| `DATABASE_URL` | `postgresql+asyncpg://legal:legal@db:5432/legal` | PostgreSQL async URL |
| `REDIS_URL` | `redis://redis:6379/0` | Redis URL |
| `LLM_BASE_URL` | `https://openrouter.ai/api/v1` | LLM provider base URL |
| `LLM_API_KEY` | *(required)* | LLM API key |
| `LLM_DEFAULT_MODEL` | `qwen/qwen3-6b-plus:free` | Default LLM model |
| `LLM_TIMEOUT` | `60` | LLM call timeout in seconds |
| `LLM_MAX_RETRIES` | `3` | Max LLM retry attempts |
| `EMBEDDING_BASE_URL` | `https://openrouter.ai/api/v1` | embedding service |
| `EMBEDDING_MODEL_NAME` | `BAAI/bge-m3` | Embedding model |
| `RERANKER_BASE_URL` | `https://openrouter.ai/api/v1/rerank` | reranker service |
| `RERANKER_MODEL_NAME` | `BAAI/bge-reranker-v2-m3` | Reranker model |
| `OCR_MODEL_REPO` | `lightonai/LightOnOCR-2-1B` | HuggingFace repo ID |
| `OCR_MODEL_DIR` | `/app/models/ocr` | Model cache directory |
| `HF_TOKEN` | *(optional)* | HuggingFace access token |
| `STORAGE_ENDPOINT` | `http://minio:9000` | S3-compatible endpoint |
| `STORAGE_ACCESS_KEY` | `minioadmin` | Storage access key |
| `STORAGE_SECRET_KEY` | `minioadmin` | Storage secret key |
| `STORAGE_BUCKET` | `legal-docs` | Bucket name |
| `WEAVIATE_URL` | `http://weaviate:8080` | Weaviate endpoint |
| `WEAVIATE_COLLECTION_NAME` | `DocumentChunk` | Weaviate collection |
| `NEO4J_URI` | `bolt://neo4j:7687` | Neo4j Bolt URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | *(required)* | Neo4j password |
| `CHAT_HISTORY_MAX_TURNS` | `10` | Chat turns passed to LLM |
| `LANGFUSE_HOST` | `http://langfuse:3000` | Langfuse URL |

</details>

### 3 — Start Infrastructure

```bash
docker compose up -d
```

Monitor startup:

```bash
docker compose ps                    # all should show "healthy"

# OCR model download can take 5–15 min on first run
docker compose logs -f api | grep -i ocr
```

Once ready:

```bash
curl http://localhost:8000/health/readiness
# {"status": "ready", "checks": {"database": true, "redis": true, "ocr_model": true}}
```

### Create MinIO bucket
```bash
docker compose exec minio mc alias set local http://localhost:9000 minioadmin minioadmin
docker compose exec minio mc mb local/legal-docs
```

### 4 — Run Database Migrations

```bash
# If running from host (not inside Docker), update DATABASE_URL in .env:
# DATABASE_URL=postgresql+asyncpg://legal:legal@localhost:5432/legal

alembic upgrade head
```

Expected output:

```
INFO  [alembic.runtime.migration] Running upgrade -> 0001, Initial schema with all tables, indexes, and seed data
```

This creates all tables, indexes, and seeds: 1 default user, 3 LLM models, 1 default system prompt.

---

✅ **API is live at `http://localhost:8000` — explore it at [`/docs`](http://localhost:8000/docs)**

---

## 💻 Running Locally (Faster Iteration)

Prefer hot-reload without rebuilding Docker images? Run infrastructure in Docker, app code on your machine:

```bash
# Start only infrastructure services
docker compose up -d db redis weaviate neo4j minio tei-embedding tei-reranker

# Update .env to point to localhost
DATABASE_URL=postgresql+asyncpg://legal:legal@localhost:5432/legal
REDIS_URL=redis://localhost:6379/0
WEAVIATE_URL=http://localhost:8080
NEO4J_URI=bolt://localhost:7687
STORAGE_ENDPOINT=http://localhost:9000
EMBEDDING_BASE_URL=http://localhost:8081
RERANKER_BASE_URL=http://localhost:8082

# Run migrations
alembic upgrade head

# Start API with hot-reload
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload --log-level debug
```

---

## ⚙️ Celery Worker

Document ingestion runs asynchronously. The `worker` service starts automatically with `docker compose up`, but you can also run it locally:

```bash
# Ensure Redis is up
docker compose up -d redis

# Start worker
celery -A src.celery_app worker --loglevel=info --concurrency=2

# Single-process mode (easier to debug)
celery -A src.celery_app worker --pool=solo --loglevel=debug
```

Expected startup output:

```
[tasks]
  . legal.ingest_document
[INFO/MainProcess] celery@hostname ready.
```

**Optional: Flower dashboard** for monitoring tasks

```bash
uv pip install flower
celery -A src.celery_app flower --port=5555
# → http://localhost:5555
```

---

## 🧪 Tests

Three layers — run whichever fits your context.

### Unit Tests — no services needed, everything mocked

```bash
pytest tests/unit/ -v

# With coverage
pytest tests/unit/ --cov=src --cov-report=term-missing

# Specific module
pytest tests/unit/test_services/ -v
pytest tests/unit/test_entities/ -v
pytest tests/unit/test_clients/ -v
```

Expected:

```
tests/unit/test_entities/test_message.py::test_is_from_user PASSED
...
========== 30 passed in 0.47s ==========
```

### Integration Tests — SQLite in-memory, no Docker needed

```bash
pytest tests/integration/ -v

# Repository layer
pytest tests/integration/test_repositories/ -v

# API endpoints
pytest tests/integration/test_api/ -v
```

### Contract Tests — requires live services, auto-skipped if env vars missing

```bash
LLM_API_KEY=sk-...              pytest tests/contracts/test_llm_client_contract.py -v
EMBEDDING_BASE_URL=http://localhost:8081  pytest tests/contracts/test_embedding_client_contract.py -v
WEAVIATE_URL=http://localhost:8080       pytest tests/contracts/test_vector_store_client_contract.py -v

# All at once
LLM_API_KEY=sk-... \
EMBEDDING_BASE_URL=http://localhost:8081 \
WEAVIATE_URL=http://localhost:8080 \
pytest tests/contracts/ -v
```

### Recommended CI command

```bash
# Unit + integration only — fast, no external deps
pytest tests/unit/ tests/integration/ -v --tb=short

# With HTML report
pytest tests/unit/ tests/integration/ \
  --cov=src --cov-report=html --cov-report=term-missing
open htmlcov/index.html
```

### Useful flags

```bash
pytest tests/unit/ -x        # stop on first failure
pytest tests/unit/ --lf      # rerun only previously failed tests
pytest tests/unit/ -k "chat" # filter by name
pytest tests/unit/ -sv       # verbose + stdout
pytest tests/unit/ -n 4      # parallel (requires pytest-xdist)
```

---

## 🔧 Alembic Migrations

```bash
# Apply all pending migrations
alembic upgrade head

# Check current state
alembic current

# Create a new migration after changing ORM models
alembic revision --autogenerate -m "add_column_xyz"
alembic upgrade head

# Roll back one step
alembic downgrade -1

# Roll back to a specific version
alembic downgrade 0001

# Reset everything (dev only — destroys all data)
alembic downgrade base
```

---

## 🌐 Service Ports

| Service | URL | Credentials |
|---------|-----|-------------|
| API | http://localhost:8000 | — |
| Swagger UI | http://localhost:8000/docs | — |
| PostgreSQL | localhost:5432 | `legal` / `legal` |
| Redis | localhost:6379 | — |
| Weaviate | http://localhost:8080 | — |
| Neo4j Browser | http://localhost:7474 | `neo4j` / `NEO4J_PASSWORD` |
| MinIO Console | http://localhost:9001 | `minioadmin` / `minioadmin` |
| Langfuse | http://localhost:3000 | — |

---

## 🗂️ Project Structure

```
legal/
├── src/
│   ├── core/           # Config, logging, exceptions, constants
│   ├── entities/       # Pure Python domain objects
│   ├── db_models/      # SQLAlchemy ORM models
│   ├── interfaces/     # ABC contracts for clients & repos
│   ├── clients/        # LLM, Weaviate, Neo4j, TEI, MinIO
│   ├── repositories/   # Persistence layer
│   ├── prompts/        # Agent prompt resources
│   ├── services/       # Business logic & orchestration
│   ├── api/            # HTTP routes, schemas, SSE
│   ├── providers/      # Dependency injection
│   ├── main.py         # FastAPI entry point
│   └── celery_app.py   # Celery task definitions
├── tests/
│   ├── unit/           # Pure unit tests (fully mocked)
│   ├── integration/    # SQLite in-memory API/repo tests
│   └── contracts/      # Live service contract tests
├── alembic/            # DB migration scripts
├── docker/             # Dockerfile + entrypoints
├── docker-compose.yml
├── pyproject.toml
└── .env.example
```

---

## 🚨 Troubleshooting

**OCR model not loading** — `readiness` shows `"ocr_model": false`

```bash
docker compose logs api | grep -i ocr
# If stalled, download manually from host:
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('lightonai/LightOnOCR-2-1B', local_dir='./models/ocr')
"
# Then add to docker-compose.yml: volumes: - ./models:/app/models
```

**Alembic migration fails**

```bash
docker compose ps db          # confirm db is healthy
psql postgresql://legal:legal@localhost:5432/legal -c "SELECT 1"

# Dev reset
docker compose down db -v && docker compose up -d db && alembic upgrade head
```

**LLM calls returning 401 / 402**

```bash
curl https://openrouter.ai/api/v1/models \
  -H "Authorization: Bearer $LLM_API_KEY" | python3 -m json.tool
# Check credits: https://openrouter.ai/settings/credits
```

**Port already in use**

```bash
lsof -i :8000   # API
lsof -i :5432   # PostgreSQL
lsof -i :8080   # Weaviate
lsof -i :6379   # Redis
```

**Celery tasks stuck in queue**

```bash
redis-cli -u redis://localhost:6379 ping   # should return PONG
docker compose logs worker
docker compose restart worker
```

**Import errors in tests**

```bash
which python      # should point to .venv/bin/python
uv pip install -e ".[dev]" && uv pip install aiosqlite
pytest tests/unit/ -v --tb=short
```
