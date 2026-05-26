"""Structured JSON logging configuration."""
from __future__ import annotations

import logging
import json
import sys
import traceback
from datetime import datetime
from typing import Any, Optional
from contextvars import ContextVar

from app.config.settings import settings

# Context vars for trace propagation
_trace_id: ContextVar[Optional[str]] = ContextVar("trace_id", default=None)
_span_id: ContextVar[Optional[str]] = ContextVar("span_id", default=None)
_session_id: ContextVar[Optional[str]] = ContextVar("session_id", default=None)


def set_trace_context(trace_id: str, span_id: str = None, session_id: str = None) -> None:
    """Set trace context for current async context."""
    _trace_id.set(trace_id)
    if span_id:
        _span_id.set(span_id)
    if session_id:
        _session_id.set(session_id)


class StructuredFormatter(logging.Formatter):
    """JSON structured log formatter for production observability."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "service": settings.otel_service_name,
            "environment": settings.environment,
            "version": settings.app_version,
        }

        # Inject trace context
        trace_id = _trace_id.get()
        if trace_id:
            log_entry["trace_id"] = trace_id
        span_id = _span_id.get()
        if span_id:
            log_entry["span_id"] = span_id
        session_id = _session_id.get()
        if session_id:
            log_entry["session_id"] = session_id

        # Include extra fields
        if hasattr(record, "__dict__"):
            reserved = {
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message",
                "taskName",
            }
            for key, value in record.__dict__.items():
                if key not in reserved and not key.startswith("_"):
                    log_entry[key] = value

        # Include exception info
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info),
            }

        return json.dumps(log_entry, default=str)


def configure_logging() -> None:
    """Configure application-wide structured logging."""
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.log_level, logging.INFO))

    # Remove existing handlers
    root_logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredFormatter())
    root_logger.addHandler(handler)

    # Suppress noisy third-party loggers
    for noisy in ["uvicorn.access", "chromadb", "httpx", "httpcore"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Get a named logger."""
    return logging.getLogger(name)
