"""
Storage key builder — consistent key generation for object storage.

All storage keys follow a predictable hierarchical pattern.
This makes listing, grouping, and deletion by document reliable.
"""

from __future__ import annotations


class StorageKeyBuilder:
    """
    Builds storage keys for S3-compatible object storage.

    Key format: {file_type}/{document_id}/{filename}
    Example: "pdf/3f7a2b4c-uuid/peraturan-pemerintah.pdf"
    """

    @staticmethod
    def document_key(file_type: str, document_id: str, file_name: str) -> str:
        """
        Build the storage key for a document file.

        Args:
            file_type: File extension without dot (e.g. "pdf", "docx").
            document_id: UUID of the document record.
            file_name: Original filename including extension.

        Returns:
            Storage key string.
        """
        safe_name = file_name.replace(" ", "_")
        return f"{file_type}/{document_id}/{safe_name}"
