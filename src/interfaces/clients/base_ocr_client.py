"""
Abstract base for OCR clients.

Design: Lazy initialization.
The OCR pipeline is NOT loaded at application startup.
It is activated on the first call to extract_text() and kept alive
for subsequent calls (lazy singleton within the client instance).

This means:
- Application starts immediately without waiting for model download.
- OCR overhead is paid only by the first document ingestion request.
- is_ready() reflects whether the pipeline has been lazily initialized.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseOCRClient(ABC):
    """Contract for all OCR client implementations."""

    @abstractmethod
    def is_ready(self) -> bool:
        """
        Return True if the OCR pipeline has been initialized and is usable.

        Before the first extract_text() call this returns False.
        After the first successful extraction it returns True.

        Returns:
            True when the pipeline is loaded and operational.
        """
        ...

    @abstractmethod
    async def extract_text(self, file_bytes: bytes, file_type: str) -> str:
        """
        Extract text from a document file.

        Lazily initializes the Docling pipeline on first call.
        Subsequent calls reuse the already-initialized pipeline.

        Args:
            file_bytes: Raw bytes of the document file.
            file_type: File type string: "pdf", "docx", or "txt".

        Returns:
            Extracted and cleaned text as a single string.

        Raises:
            OCRError: If the pipeline cannot be initialized or extraction fails.
        """
        ...