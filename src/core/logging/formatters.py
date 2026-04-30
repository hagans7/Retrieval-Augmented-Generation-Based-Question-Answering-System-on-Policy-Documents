"""
Log formatters for development and production environments.

Development: colored, human-readable output with aligned fields.
Production: single-line JSON objects for machine parsing and log aggregation.

Selected by logger.py based on DEPLOYMENT_MODE setting.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone, timedelta

# WIB offset: UTC+7
_WIB = timezone(timedelta(hours=7))

# ANSI color codes
_COLORS = {
    "DEBUG": "\033[36m",     # Cyan
    "INFO": "\033[32m",      # Green
    "WARNING": "\033[33m",   # Yellow
    "ERROR": "\033[31m",     # Red
    "CRITICAL": "\033[1;31m", # Bold Red
    "RESET": "\033[0m",
}


def _wib_now(record: logging.LogRecord) -> str:
    """Convert log record timestamp to WIB ISO 8601 string."""
    utc_dt = datetime.fromtimestamp(record.created, tz=timezone.utc)
    wib_dt = utc_dt.astimezone(_WIB)
    return wib_dt.strftime("%Y-%m-%dT%H:%M:%S+07:00")


class DevelopmentFormatter(logging.Formatter):
    """
    Human-readable, colored formatter for development mode.

    Format: [TIMESTAMP_WIB] [LEVEL   ] [module.name] [cid=...] MESSAGE | key=val
    Colors are applied per log level for visual scanning.
    """

    def format(self, record: logging.LogRecord) -> str:
        level_name = record.levelname
        color = _COLORS.get(level_name, _COLORS["RESET"])
        reset = _COLORS["RESET"]

        timestamp = _wib_now(record)
        level_padded = f"{level_name:<8}"
        module = record.name

        # Extract correlation_id from extra if present
        correlation_id = getattr(record, "correlation_id", "-")

        # Build extra fields string (exclude standard LogRecord attrs)
        _standard_attrs = {
            "name", "msg", "args", "levelname", "levelno", "pathname",
            "filename", "module", "exc_info", "exc_text", "stack_info",
            "lineno", "funcName", "created", "msecs", "relativeCreated",
            "thread", "threadName", "processName", "process", "message",
            "correlation_id",
        }
        extras = {
            k: v for k, v in record.__dict__.items()
            if k not in _standard_attrs and not k.startswith("_")
        }
        extras_str = " | " + " ".join(f"{k}={v}" for k, v in extras.items()) if extras else ""

        message = record.getMessage()

        line = (
            f"{color}[{timestamp}] [{level_padded}] [{module}] "
            f"[cid={correlation_id}] {message}{extras_str}{reset}"
        )

        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)

        return line


class ProductionFormatter(logging.Formatter):
    """
    JSON structured formatter for production mode.

    Outputs one JSON object per line. All extra={} fields from log calls
    are included as top-level keys in the JSON object.
    Machine-parseable by log aggregation systems (ELK, Loki, CloudWatch).
    """

    def format(self, record: logging.LogRecord) -> str:
        _standard_attrs = {
            "name", "msg", "args", "levelname", "levelno", "pathname",
            "filename", "module", "exc_info", "exc_text", "stack_info",
            "lineno", "funcName", "created", "msecs", "relativeCreated",
            "thread", "threadName", "processName", "process", "message",
        }

        log_object: dict = {
            "timestamp": _wib_now(record),
            "level": record.levelname,
            "module": record.name,
            "correlation_id": getattr(record, "correlation_id", None),
            "message": record.getMessage(),
        }

        # Merge all extra fields at top level
        for k, v in record.__dict__.items():
            if k not in _standard_attrs and not k.startswith("_") and k != "correlation_id":
                log_object[k] = v

        if record.exc_info:
            log_object["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_object, ensure_ascii=False, default=str)
