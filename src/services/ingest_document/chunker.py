"""
Document chunker — hierarchical and sliding-window chunking strategies.

Primary: hierarchical chunking for Indonesian legal documents (BAB/Pasal/Ayat).
Fallback: sliding-window tokenisation for unstructured text.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field

from src.core.constants.ingestion import (
    CHUNK_MIN_TOKENS,
    CHUNK_OVERLAP_TOKENS,
    CHUNK_TARGET_TOKENS,
)
from src.core.logging.logger import get_logger

logger = get_logger(__name__)

# Regex markers for Indonesian legal document hierarchy
_BAB_RE = re.compile(r"^(BAB\s+[IVXLCDM\d]+)", re.MULTILINE)
_PASAL_RE = re.compile(r"^(Pasal\s+\d+[A-Z]?)", re.MULTILINE)
_AYAT_RE = re.compile(r"^(\(\d+\))", re.MULTILINE)

# Minimum fraction of paragraphs that must have structural markers
# for hierarchical strategy to be selected.
_HIERARCHICAL_THRESHOLD = 0.10


@dataclass
class Chunk:
    """Represents a single text chunk ready for embedding and storage."""

    chunk_id: str
    document_id: str
    content: str
    chunk_index: int
    token_count: int
    section_type: str = ""
    parent_section: str = ""
    file_name: str = ""

    def to_dict(self) -> dict:
        """Return a flat dict for Weaviate upsert."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "content": self.content,
            "chunk_index": self.chunk_index,
            "token_count": self.token_count,
            "section_type": self.section_type,
            "parent_section": self.parent_section,
            "file_name": self.file_name,
        }


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: 1 token ≈ 4 characters for Indonesian text."""
    return max(1, len(text) // 4)


class DocumentChunker:
    """
    Chunks extracted document text into overlapping segments.

    Automatically selects between hierarchical (legal structure-aware)
    and sliding-window strategies based on the document's content.

    Args:
        document_id: UUID of the parent document.
        file_name: Original filename for metadata.
    """

    def __init__(self, document_id: str, file_name: str = "") -> None:
        self._document_id = document_id
        self._file_name = file_name

    def chunk(self, text: str) -> list[Chunk]:
        """
        Chunk the given text using the most appropriate strategy.

        Args:
            text: Cleaned extracted text from the document.

        Returns:
            Ordered list of Chunk objects with metadata.
        """
        if self._detect_legal_structure(text):
            logger.debug("Using hierarchical chunking strategy", extra={"document_id": self._document_id})
            return self._hierarchical_chunk(text)
        logger.debug("Using sliding-window chunking strategy", extra={"document_id": self._document_id})
        return self._sliding_window_chunk(text)

    def _detect_legal_structure(self, text: str) -> bool:
        """Return True if the text has sufficient legal structural markers."""
        paragraphs = [p for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            return False
        marker_count = sum(
            1 for p in paragraphs
            if _PASAL_RE.search(p) or _BAB_RE.search(p) or _AYAT_RE.search(p)
        )
        return (marker_count / len(paragraphs)) >= _HIERARCHICAL_THRESHOLD

    def _hierarchical_chunk(self, text: str) -> list[Chunk]:
        """
        Split by Pasal boundaries. Each Pasal becomes one chunk.
        If a Pasal exceeds CHUNK_TARGET_TOKENS, further split with sliding window.
        """
        chunks: list[Chunk] = []
        # Split on Pasal markers, keeping the marker with its content
        parts = _PASAL_RE.split(text)
        current_bab = ""
        chunk_index = 0

        i = 0
        while i < len(parts):
            part = parts[i].strip()
            if not part:
                i += 1
                continue

            # Detect BAB changes in preamble or between pasals
            bab_match = _BAB_RE.search(part)
            if bab_match:
                current_bab = bab_match.group(1)

            # Check if this part is a Pasal header
            if _PASAL_RE.fullmatch(part.strip()):
                pasal_header = part.strip()
                # Consume the content that follows this header
                pasal_content = parts[i + 1].strip() if i + 1 < len(parts) else ""
                i += 2
                full_text = f"{pasal_header}\n\n{pasal_content}".strip()
            else:
                full_text = part
                i += 1

            if not full_text or _estimate_tokens(full_text) < CHUNK_MIN_TOKENS:
                continue

            # If Pasal content is too long, apply sliding window on it
            if _estimate_tokens(full_text) > CHUNK_TARGET_TOKENS * 1.5:
                sub_chunks = self._split_with_sliding_window(
                    full_text, section_type="pasal", parent_section=current_bab,
                    start_index=chunk_index
                )
                chunks.extend(sub_chunks)
                chunk_index += len(sub_chunks)
            else:
                chunks.append(Chunk(
                    chunk_id=str(uuid.uuid4()),
                    document_id=self._document_id,
                    content=full_text,
                    chunk_index=chunk_index,
                    token_count=_estimate_tokens(full_text),
                    section_type="pasal",
                    parent_section=current_bab,
                    file_name=self._file_name,
                ))
                chunk_index += 1

        return chunks

    def _sliding_window_chunk(self, text: str) -> list[Chunk]:
        """Split text into overlapping windows by sentence boundaries."""
        return self._split_with_sliding_window(text, section_type="paragraph", parent_section="", start_index=0)

    def _split_with_sliding_window(
        self,
        text: str,
        section_type: str,
        parent_section: str,
        start_index: int,
    ) -> list[Chunk]:
        """Core sliding-window implementation over sentence-split text."""
        # Split on sentence boundaries (period + space or newline)
        sentences = re.split(r"(?<=[.!?])\s+|\n", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks: list[Chunk] = []
        window: list[str] = []
        window_tokens = 0
        chunk_index = start_index

        for sentence in sentences:
            s_tokens = _estimate_tokens(sentence)

            if window_tokens + s_tokens > CHUNK_TARGET_TOKENS and window:
                # Emit current window
                content = " ".join(window)
                if _estimate_tokens(content) >= CHUNK_MIN_TOKENS:
                    chunks.append(Chunk(
                        chunk_id=str(uuid.uuid4()),
                        document_id=self._document_id,
                        content=content,
                        chunk_index=chunk_index,
                        token_count=_estimate_tokens(content),
                        section_type=section_type,
                        parent_section=parent_section,
                        file_name=self._file_name,
                    ))
                    chunk_index += 1

                # Retain overlap: keep last N tokens worth of sentences
                overlap_tokens = 0
                overlap_sentences: list[str] = []
                for s in reversed(window):
                    if overlap_tokens + _estimate_tokens(s) > CHUNK_OVERLAP_TOKENS:
                        break
                    overlap_sentences.insert(0, s)
                    overlap_tokens += _estimate_tokens(s)

                window = overlap_sentences
                window_tokens = overlap_tokens

            window.append(sentence)
            window_tokens += s_tokens

        # Emit the last window
        if window:
            content = " ".join(window)
            if _estimate_tokens(content) >= CHUNK_MIN_TOKENS:
                chunks.append(Chunk(
                    chunk_id=str(uuid.uuid4()),
                    document_id=self._document_id,
                    content=content,
                    chunk_index=chunk_index,
                    token_count=_estimate_tokens(content),
                    section_type=section_type,
                    parent_section=parent_section,
                    file_name=self._file_name,
                ))

        return chunks
