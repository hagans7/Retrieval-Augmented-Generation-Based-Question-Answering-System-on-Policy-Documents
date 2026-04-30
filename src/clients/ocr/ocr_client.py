"""
OCR client — lazy document text extraction using Docling.

Design: Lazy initialization.

The DoclingPipeline is NOT created at application startup.
It is instantiated on the first extract_text() call and reused
for all subsequent calls (lazy singleton within this client instance).

Consequence for startup:
    Application starts instantly.
    No model download. No blocking. No OCR-related startup delay.
    The first document ingestion request bears the Docling initialization
    cost (Docling warm-up, a few seconds) — all subsequent calls are fast.

Consequence for is_ready():
    Returns False until the first successful extract_text() call.
    The /health/readiness endpoint reflects this accurately.
    A False OCR readiness does NOT prevent the API from serving chat
    requests — it only means no document has been processed yet.

Thread pool:
    Docling's DocumentConverter.convert() is synchronous and CPU-bound.
    We offload it to a dedicated single-worker ThreadPoolExecutor so the
    async event loop is never blocked during document conversion.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from src.clients.ocr.docling_pipeline import DoclingPipeline
from src.clients.ocr.text_cleaner import TextCleaner
from src.core.exceptions.infrastructure import OCRError
from src.core.logging.logger import get_logger
from src.interfaces.clients.base_ocr_client import BaseOCRClient

# Dedicated single-worker thread pool for Docling's blocking CPU/IO work.
# One worker is intentional: Docling is memory-intensive; running two
# concurrent conversions would risk OOM on typical machines.
_DOCLING_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="docling")


class DoclingOCRClient(BaseOCRClient):
    """
    OCR client using Docling's native document conversion pipeline.

    Lazy: pipeline is created on first extract_text() call, not at startup.
    Concurrent-safe: uses asyncio.Lock to prevent double initialization.

    Args:
        model_repo: Reserved for future custom OCR backend. Not used now.
        model_dir: Reserved for future custom OCR backend. Not used now.
        hf_token: Reserved for future custom OCR backend. Not used now.
    """

    def __init__(
        self,
        model_repo: str = "",
        model_dir: str = "",
        hf_token: str | None = None,
    ) -> None:
        # Parameters retained for interface compatibility and future use.
        # Docling currently manages its own internal models.
        self._model_repo = model_repo
        self._model_dir = model_dir

        self._pipeline: DoclingPipeline | None = None
        self._is_ready_flag: bool = False
        self._init_lock = asyncio.Lock()
        self._logger = get_logger(__name__)

    def is_ready(self) -> bool:
        """
        Return True if the Docling pipeline has been used at least once.

        Returns False until the first successful extract_text() call.
        """
        return self._is_ready_flag

    async def _ensure_pipeline(self) -> DoclingPipeline:
        """
        Return the DoclingPipeline, lazily creating it on first call.

        Uses asyncio.Lock with double-checked locking to prevent
        concurrent initialization races if multiple ingestion tasks
        arrive simultaneously before the first one completes.

        Returns:
            Ready DoclingPipeline instance.
        """
        if self._pipeline is not None:
            return self._pipeline

        async with self._init_lock:
            # Re-check inside the lock: another coroutine may have
            # completed initialization while we waited for the lock.
            if self._pipeline is not None:
                return self._pipeline

            self._logger.info(
                "Lazily initializing Docling pipeline on first document request. "
                "This will take a few seconds on the first call."
            )
            self._pipeline = DoclingPipeline()
            self._logger.info("Docling pipeline initialized and ready.")

        return self._pipeline

    async def extract_text(self, file_bytes: bytes, file_type: str) -> str:
        """
        Extract and clean text from document bytes.

        Lazily initializes the Docling pipeline on first call.
        Docling's blocking conversion runs in a thread pool executor
        so the async event loop is never blocked.

        Args:
            file_bytes: Raw document bytes.
            file_type: Document type: "pdf", "docx", or "txt".

        Returns:
            Cleaned extracted text string.

        Raises:
            OCRError: If Docling fails to process the document.
        """
        pipeline = await self._ensure_pipeline()

        loop = asyncio.get_event_loop()

        # DoclingPipeline._run_docling() is synchronous and blocking.
        # We call it directly in the executor (not pipeline.extract()
        # which is async) to avoid nested event loop issues.
        import tempfile
        import os

        # Write bytes to temp file for Docling (requires a file path)
        suffix = f".{file_type}"
        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=suffix, delete=False, mode="wb"
            ) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name

            self._logger.debug(
                "Running Docling extraction in thread pool",
                extra={"file_type": file_type, "bytes": len(file_bytes)},
            )

            # Run the synchronous Docling conversion off the event loop
            raw_text = await loop.run_in_executor(
                _DOCLING_EXECUTOR,
                partial(pipeline._run_docling, tmp_path, file_type),
            )

        except OCRError:
            raise
        except Exception as exc:
            raise OCRError(
                message=f"OCR extraction failed for file_type='{file_type}'.",
                context={"file_type": file_type, "error": str(exc)},
            ) from exc
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass  # Best-effort cleanup

        cleaned = TextCleaner.clean(raw_text)

        if not self._is_ready_flag:
            self._is_ready_flag = True
            self._logger.info(
                "OCR client marked ready after first successful extraction.",
                extra={"file_type": file_type, "output_chars": len(cleaned)},
            )

        return cleaned