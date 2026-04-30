"""
Docling pipeline — lazy document extraction using Docling's native pipeline.

Design decision:
    Docling has its own configurable OCR backend system.
    We do NOT load an external OCR model manually and inject it.
    Instead, we configure Docling to use its own OCR capabilities
    which it activates automatically when scanned content is detected.

    For LightOnOCR specifically: Docling supports it via RapidOcrOptions
    or as a custom pipeline backend. For our use case, Docling's default
    EasyOcrOptions is the correct integration point — it handles
    layout detection + OCR in one unified pipeline.

    The pipeline is instantiated lazily on first use and reused thereafter.
    No model is loaded at application startup.

Why NOT load via transformers.pipeline("image-to-text"):
    LightOnOCR-2-1B uses the mistral3 architecture which requires
    a transformers version >= 4.50.0. Current env uses 4.47.x.
    More importantly, Docling's own pipeline already orchestrates OCR
    region detection + model inference — we should not duplicate that.

Thread safety:
    DocumentConverter is not thread-safe. We create one per extraction call.
    This is safe because extract() is called from Celery workers, not
    concurrently within a single process.
"""

from __future__ import annotations

import os
import tempfile

from src.core.exceptions.infrastructure import OCRError
from src.core.logging.logger import get_logger

logger = get_logger(__name__)


class DoclingPipeline:
    """
    Thin wrapper around Docling's DocumentConverter.

    One instance per DoclingOCRClient. DocumentConverter is constructed
    fresh per extract() call because Docling's converter is not designed
    to be shared across concurrent invocations.
    """

    async def extract(self, file_bytes: bytes, file_type: str) -> str:
        """
        Extract text from document bytes using Docling.

        Writes bytes to a temp file (Docling requires a file path),
        runs the converter, exports to Markdown, then cleans up.

        Args:
            file_bytes: Raw bytes of the document.
            file_type: Extension without dot: "pdf", "docx", or "txt".

        Returns:
            Extracted text string (Markdown-structured for legal docs).

        Raises:
            OCRError: If Docling conversion fails.
        """
        suffix = f".{file_type}"
        tmp_path: str | None = None

        try:
            # Docling requires a real file path on disk.
            with tempfile.NamedTemporaryFile(
                suffix=suffix, delete=False, mode="wb"
            ) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name

            logger.debug(
                "Starting Docling extraction",
                extra={"file_type": file_type, "bytes": len(file_bytes)},
            )

            extracted_text = self._run_docling(tmp_path, file_type)

            logger.debug(
                "Docling extraction complete",
                extra={"file_type": file_type, "chars": len(extracted_text)},
            )
            return extracted_text

        except OCRError:
            raise
        except Exception as exc:
            raise OCRError(
                message=f"Docling extraction failed for file_type='{file_type}'.",
                context={"file_type": file_type, "error": str(exc)},
            ) from exc
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass  # Best effort cleanup

    def _run_docling(self, file_path: str, file_type: str) -> str:
        """
        Run Docling DocumentConverter synchronously.

        Separated from extract() so it can be called in a thread pool
        executor if needed in the future (Docling is CPU-bound/blocking).

        Args:
            file_path: Absolute path to the temp file.
            file_type: File type string for logging.

        Returns:
            Extracted text in Markdown format.

        Raises:
            OCRError: If the converter fails.
        """
        try:
            from docling.document_converter import DocumentConverter

            converter = DocumentConverter()
            result = converter.convert(file_path)
            return result.document.export_to_markdown()

        except Exception as exc:
            raise OCRError(
                message=f"DocumentConverter failed for '{file_type}' file.",
                context={"file_path": file_path, "error": str(exc)},
            ) from exc