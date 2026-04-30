"""
Celery application — background document ingestion task.

Initialization order (CRITICAL):
  1. register_all_models() — loads all ORM classes, calls configure_mappers()
     This MUST happen before any SQLAlchemy session or model import.
     Without this, string-based relationship() calls hang silently in async.
  2. Create NullPool engine — avoids QueuePool deadlock across event loops
  3. Run _pipeline() in fresh SelectorEventLoop (Windows IocpProactor fix)

Why Celery needs explicit model registration:
  FastAPI startup imports main.py → routers → services → all ORM classes,
  so mapper is fully configured before any request arrives.
  Celery workers only load celery_app.py — ORM classes are never imported
  unless explicitly triggered. SQLAlchemy's lazy mapper resolution then
  silently deadlocks inside async coroutines.
"""
from __future__ import annotations
import asyncio
import base64
import logging
import sys

from celery import Celery
from src.core.config.settings import settings

# Register all ORM models at worker startup — before any task runs.
# This is a module-level call so it runs once when celery_app.py is imported.
from src.db_models import register_all_models
register_all_models()

logger = logging.getLogger(__name__)

logger.info("=== STARTING GLOBAL IMPORTS ===")

logger.info("1. Importing DB Models...")
from src.db_models import register_all_models
register_all_models()

logger.info("2. Importing SQLAlchemy components...")
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.pool import NullPool

logger.info("3. Importing Repositories...")
from src.repositories.document.document_repository import DocumentRepository

logger.info("4. Importing Storage Client...")
from src.clients.storage.storage_client import S3CompatibleStorageClient

logger.info("5. Importing Vector & Graph Clients...")
from src.clients.vector_store.vector_store_client import WeaviateVectorStoreClient
from src.clients.graph.graph_client import Neo4jGraphClient

logger.info("6. Importing LLM & Embedding...")
from src.clients.embedding.embedding_client import OpenAIEmbeddingClient
from src.clients.llm.llm_client import OpenAICompatibleLLMClient
from src.services.ingest_document.entity_extractor import EntityExtractor
from src.services.ingest_document.chunker import DocumentChunker

logger.info("7. Importing OCR (Docling)...")
from src.clients.ocr.ocr_client import DoclingOCRClient

logger.info("=== ALL GLOBAL IMPORTS SUCCESSFUL ===")

celery = Celery(
    "legal_api_celery",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
)

celery.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_max_retries=3,
    broker_connection_retry_on_startup=True,
)


def _run_async(coro):
    """
    Run a coroutine from Celery worker thread safely on all platforms.

    Forces SelectorEventLoop on Windows (IocpProactor deadlocks with
    asyncio called from Celery threads).
    Creates a fresh event loop per call — never reuses an existing one.
    """
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        try:
            pending = asyncio.all_tasks(loop)
            if pending:
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
        finally:
            loop.close()
            asyncio.set_event_loop(None)


@celery.task(
    bind=True,
    name="legal.ingest_document",
    max_retries=3,
    default_retry_delay=60,
)
def ingest_document_task(
    self,
    file_b64: str,
    file_name: str,
    file_type: str,
    user_id: str | None,
    model_name: str,
    document_id: str,
    storage_key: str,
) -> str:
    """
    Process a pre-created document through the full ingestion pipeline.

    register_all_models() is called at module level above — mappers are
    guaranteed to be configured before this task function is ever called.
    All client classes are imported inside the coroutine to keep them
    isolated from the FastAPI provider cache (lru_cache).
    """
    file_bytes = base64.b64decode(file_b64)
    logger.info(
        f"[CELERY] STARTED: {file_name} ({len(file_bytes)} bytes) "
        f"doc_id={document_id} platform={sys.platform}"
    )

    async def _pipeline() -> str:
        # ── Step 1: DB engine (NullPool) ──────────────────────
        # NullPool: no connection pooling, no event-loop-bound state.
        # Required because each Celery task creates a new event loop.
        # QueuePool would try to reuse connections from a closed loop → deadlock.
        logger.info("[CELERY][1a] Creating NullPool DB engine...")

        engine = create_async_engine(
            settings.DATABASE_URL,
            poolclass=NullPool,
            echo=False,
        )
        factory = async_sessionmaker(bind=engine, expire_on_commit=False)
        logger.info("[CELERY][1b] NullPool engine ready.")

        logger.info("[CELERY][1c] Opening DB session...")
        async with factory() as session:
            logger.info("[CELERY][1d] DB session open OK.")

            logger.info("[DBG] before import document_repository module")
            from src.repositories.document import document_repository
            logger.info("[DBG] after import module")
            DocumentRepository = document_repository.DocumentRepository
            logger.info("[DBG] before repo init")
            doc_repo = DocumentRepository(session)
            logger.info("[DBG] after repo init")

            logger.info("[CELERY][1e] DocumentRepository ready.")

            # ── Step 2: Mark processing ───────────────────────
            logger.info("[CELERY][2a] BEFORE update_status(processing)...")
            await doc_repo.update_status(document_id, "processing")
            logger.info("[CELERY][2b] AFTER update_status(processing) OK.")

            try:
                # ── Step 3: MinIO upload ──────────────────────
                logger.info(f"[CELERY][3a] BEFORE MinIO upload. endpoint={settings.STORAGE_ENDPOINT}")
                storage = S3CompatibleStorageClient(
                    endpoint=settings.STORAGE_ENDPOINT,
                    access_key=settings.STORAGE_ACCESS_KEY,
                    secret_key=settings.STORAGE_SECRET_KEY,
                    bucket=settings.STORAGE_BUCKET,
                    use_ssl=settings.STORAGE_USE_SSL,
                )
                content_type_map = {
                    "pdf": "application/pdf",
                    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    "txt": "text/plain",
                }
                await storage.upload_file(
                    key=storage_key,
                    data=file_bytes,
                    content_type=content_type_map.get(file_type, "application/octet-stream"),
                )
                logger.info(f"[CELERY][3b] AFTER MinIO upload OK.")

                # ── Step 4: OCR ───────────────────────────────
                logger.info("[CELERY][4a] BEFORE OCR (Docling)...")
                ocr = DoclingOCRClient()
                text = await ocr.extract_text(file_bytes, file_type)
                logger.info(f"[CELERY][4b] AFTER OCR done. chars={len(text)}")

                if not text.strip():
                    raise ValueError("OCR produced empty text — document may be unreadable.")

                # ── Step 5: Chunk ─────────────────────────────
                logger.info("[CELERY][5a] BEFORE chunking...")
                chunks = DocumentChunker(
                    document_id=document_id, file_name=file_name
                ).chunk(text)
                chunk_dicts = [c.to_dict() for c in chunks]
                logger.info(f"[CELERY][5b] AFTER chunking. chunks={len(chunks)}")

                if not chunks:
                    raise ValueError("Chunking produced 0 chunks.")

                # ── Step 6: Embed ─────────────────────────────
                logger.info(f"[CELERY][6a] BEFORE embedding {len(chunks)} chunks. model={settings.EMBEDDING_MODEL_NAME}")
                embedding_client = OpenAIEmbeddingClient(
                    base_url=settings.EMBEDDING_BASE_URL,
                    api_key=settings.LLM_API_KEY,
                    model_name=settings.EMBEDDING_MODEL_NAME,
                    timeout=settings.EMBEDDING_TIMEOUT,
                )
                texts = [c["content"] for c in chunk_dicts]
                vectors = await embedding_client.embed(texts)
                for i, cd in enumerate(chunk_dicts):
                    cd["vector"] = vectors[i]
                logger.info(f"[CELERY][6b] AFTER embedding. vectors={len(vectors)} dim={len(vectors[0]) if vectors else 0}")

                # ── Step 7: Weaviate ──────────────────────────
                logger.info(f"[CELERY][7a] BEFORE Weaviate connect. url={settings.WEAVIATE_URL}")
                vector_client = WeaviateVectorStoreClient(
                    weaviate_url=settings.WEAVIATE_URL,
                    collection_name=settings.WEAVIATE_COLLECTION_NAME,
                )
                vector_client.connect()
                logger.info("[CELERY][7b] Weaviate connected.")

                logger.info(f"[CELERY][7c] BEFORE upsert_chunks ({len(chunk_dicts)})...")
                await vector_client.upsert_chunks(chunk_dicts)
                vector_client.close()
                logger.info("[CELERY][7d] AFTER Weaviate upsert + close OK.")

                # ── Step 8: Entity extraction ─────────────────
                logger.info(f"[CELERY][8a] BEFORE entity extraction. model={model_name}")
                llm_client = OpenAICompatibleLLMClient(
                    base_url=settings.LLM_BASE_URL,
                    api_key=settings.LLM_API_KEY,
                    default_model=model_name,
                    timeout=settings.LLM_TIMEOUT,
                    max_retries=settings.LLM_MAX_RETRIES,
                )
                entities, relations = await EntityExtractor(
                    llm_client=llm_client, model_name=model_name
                ).extract(chunk_dicts)
                logger.info(f"[CELERY][8b] AFTER entity extraction. entities={len(entities)} relations={len(relations)}")

                # ── Step 9: Neo4j ─────────────────────────────
                logger.info(f"[CELERY][9a] BEFORE Neo4j connect. uri={settings.NEO4J_URI}")
                graph_client = Neo4jGraphClient(
                    uri=settings.NEO4J_URI,
                    user=settings.NEO4J_USER,
                    password=settings.NEO4J_PASSWORD,
                )
                await graph_client.connect()
                logger.info("[CELERY][9b] Neo4j connected.")

                if entities:
                    logger.info(f"[CELERY][9c] BEFORE upsert_entities ({len(entities)})...")
                    await graph_client.upsert_entities(entities)
                    logger.info("[CELERY][9d] upsert_entities OK.")
                if relations:
                    logger.info(f"[CELERY][9e] BEFORE upsert_relations ({len(relations)})...")
                    await graph_client.upsert_relations(relations)
                    logger.info("[CELERY][9f] upsert_relations OK.")

                await graph_client.close()
                logger.info("[CELERY][9g] Neo4j closed.")

                # ── Step 10: Mark completed ───────────────────
                logger.info("[CELERY][10a] BEFORE update_status(completed)...")
                await doc_repo.update_status(
                    document_id, "completed", chunk_count=len(chunks)
                )
                logger.info(f"[CELERY][10b] COMPLETE. doc_id={document_id} chunks={len(chunks)}")

            except Exception as exc:
                logger.error(
                    f"[CELERY][ERR] Pipeline failed: {type(exc).__name__}: {exc}",
                    exc_info=True,
                )
                await doc_repo.update_status(
                    document_id, "failed", error_message=str(exc)
                )
                logger.info("[CELERY][ERR] status=failed committed.")
                raise

        await engine.dispose()
        return document_id

    try:
        result = _run_async(_pipeline())
        logger.info(f"[CELERY] Task SUCCESS: {document_id}")
        return result
    except Exception as exc:
        logger.error(f"[CELERY] Task FAILED: {type(exc).__name__}: {exc}", exc_info=True)
        raise self.retry(exc=exc)