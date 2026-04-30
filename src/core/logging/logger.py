"""
Logger factory — single source of truth for all logging configuration.

All application code calls get_logger(__name__) to obtain a configured logger.
Never configure logging outside of this file.
Never create module-level loggers outside of this file.

Usage:
    from src.core.logging.logger import get_logger

    class MyService:
        def __init__(self) -> None:
            self._logger = get_logger(__name__)
"""

from __future__ import annotations

import logging
import sys

from src.core.logging.formatters import DevelopmentFormatter, ProductionFormatter

_CONFIGURED = False


def _configure_root_logger(deployment_mode: str, debug: bool) -> None:
    """
    Configure the root logger once at application startup.

    Subsequent calls are no-ops — configuration is idempotent.
    This is called lazily on the first get_logger() call.

    Args:
        deployment_mode: "development" or "production"
        debug: If True, set level to DEBUG regardless of mode.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    root_logger = logging.getLogger()

    # Remove any existing handlers to avoid duplicate output
    root_logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)

    if deployment_mode.lower() == "production":
        handler.setFormatter(ProductionFormatter())
        root_logger.setLevel(logging.INFO)
    else:
        handler.setFormatter(DevelopmentFormatter())
        root_logger.setLevel(logging.DEBUG if debug else logging.INFO)

    root_logger.addHandler(handler)

    # Suppress noisy third-party loggers
    # logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(
        logging.INFO if debug else logging.WARNING
    )
    logging.getLogger("sqlalchemy.engine").setLevel(
        logging.INFO if debug else logging.WARNING
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("hpack").setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """
    Return a configured logger for the given module name.

    Configures the root logger on first call using settings from the
    environment. All subsequent calls return pre-configured loggers.

    Args:
        name: Module name — always pass __name__.

    Returns:
        A configured logging.Logger instance.

    Example:
        self._logger = get_logger(__name__)
        self._logger.info("Workflow started", extra={"conversation_id": conv_id})
    """
    # Lazy import to avoid circular dependency at module load time
    from src.core.config.settings import settings

    _configure_root_logger(settings.DEPLOYMENT_MODE, settings.DEBUG)
    return logging.getLogger(name)
