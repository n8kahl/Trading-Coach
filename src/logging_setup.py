"""Logging helpers providing structured JSON output with request ID context."""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Dict


_STANDARD_ATTRS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
}

REQUEST_ID_CONTEXT: ContextVar[str | None] = ContextVar("request_id", default=None)


class RequestIdFilter(logging.Filter):
    """Attach the current request ID (if any) to the log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = REQUEST_ID_CONTEXT.get()
        return True


class JsonFormatter(logging.Formatter):
    """Render log records as structured JSON lines."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401 - inherited docstring
        payload: Dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        request_id = getattr(record, "request_id", None)
        if request_id:
            payload["request_id"] = request_id

        if record.exc_info:
            payload["exception"] = super().formatException(record.exc_info)
        if record.stack_info:
            payload["stack"] = record.stack_info

        extra = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _STANDARD_ATTRS
            and not key.startswith("_")
            and key not in {"request_id"}
        }
        if extra:
            payload["extra"] = extra

        return json.dumps(payload, default=str, separators=(",", ":"))


_CONFIGURED = False


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging once with JSON formatter + request ID filter."""

    global _CONFIGURED
    if _CONFIGURED:
        return

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(JsonFormatter())
    handler.addFilter(RequestIdFilter())

    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()
    root_logger.addHandler(handler)

    logging.captureWarnings(True)
    _CONFIGURED = True


__all__ = ["REQUEST_ID_CONTEXT", "RequestIdFilter", "JsonFormatter", "setup_logging"]
