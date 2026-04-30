"""
OCR text cleaner — post-processes raw extracted text.

Normalizes whitespace, removes OCR artefacts common in Indonesian legal documents,
and ensures consistent newline formatting before the text reaches the chunker.
"""

from __future__ import annotations

import re


class TextCleaner:
    """
    Cleans raw text extracted by Docling + LightOnOCR.

    Targets artefacts common in scanned Indonesian government PDFs:
    - Excessive whitespace and irregular newlines
    - Page number noise (e.g. "- 5 -", "Halaman 5 dari 20")
    - Header/footer repetitions
    - Hyphenated word-breaks from column layouts
    """

    # Patterns for common artefacts
    _PAGE_NUM_PATTERN = re.compile(r"-\s*\d+\s*-")
    _HALAMAN_PATTERN = re.compile(r"[Hh]alaman\s+\d+\s+dari\s+\d+")
    _EXCESSIVE_NEWLINES = re.compile(r"\n{3,}")
    _TRAILING_SPACES = re.compile(r"[ \t]+\n")
    _HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")

    @classmethod
    def clean(cls, raw_text: str) -> str:
        """
        Apply all cleaning steps to raw extracted text.

        Args:
            raw_text: Unprocessed text from Docling/OCR pipeline.

        Returns:
            Cleaned text string ready for chunking.
        """
        text = raw_text

        # Rejoin hyphenated word-breaks
        text = cls._HYPHEN_BREAK.sub(r"\1\2", text)

        # Remove page number patterns
        text = cls._PAGE_NUM_PATTERN.sub("", text)
        text = cls._HALAMAN_PATTERN.sub("", text)

        # Normalize whitespace
        text = cls._TRAILING_SPACES.sub("\n", text)
        text = cls._EXCESSIVE_NEWLINES.sub("\n\n", text)

        return text.strip()
