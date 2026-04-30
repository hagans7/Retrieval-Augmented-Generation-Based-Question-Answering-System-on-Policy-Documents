"""
Constants for the document ingestion pipeline.

Controls file size limits, supported types, and chunking parameters.
Chunking constants are calibrated for Indonesian legal documents.
"""

# File upload limits
MAX_FILE_SIZE_BYTES: int = 52_428_800  # 50 MB

# Supported MIME file types
SUPPORTED_FILE_TYPES: list[str] = ["pdf", "docx", "txt"]

# Chunking parameters
CHUNK_TARGET_TOKENS: int = 512
CHUNK_OVERLAP_TOKENS: int = 128
CHUNK_MIN_TOKENS: int = 50  # Fragments shorter than this are merged with previous chunk

# Embedding batching
EMBEDDING_BATCH_SIZE: int = 32  # TEI batch size limit

# Entity extraction triage threshold
ENTITY_EXTRACTION_MIN_TOKENS: int = 50  # Skip LLM extraction for chunks below this
