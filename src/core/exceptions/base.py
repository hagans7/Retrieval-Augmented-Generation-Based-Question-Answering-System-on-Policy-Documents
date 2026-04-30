"""
Base exception for the entire legal application.

All application-level exceptions inherit from AppBaseError.
Never raise bare Exception — always raise from this hierarchy.
"""

from __future__ import annotations


class AppBaseError(Exception):
    """
    Root exception for all legal application errors.

    Provides a consistent interface for error message and optional
    developer context that is safe to log but must never be returned
    in HTTP responses.

    Args:
        message: Human-readable description of what went wrong.
        context: Optional dict with additional debug information.
                 Never included in HTTP responses — log-only.
    """

    def __init__(self, message: str, context: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.context: dict = context or {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message={self.message!r}, context={self.context!r})"
