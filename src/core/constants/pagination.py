"""
Pagination constants for all listing endpoints.

Applied consistently across all GET /resources endpoints to enforce
maximum page size and provide sensible defaults.
"""

DEFAULT_PAGE_SIZE: int = 20
MAX_PAGE_SIZE: int = 100
