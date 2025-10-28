"""
Helper entrypoint that resolves the deployment port before booting uvicorn.

Some hosting providers invoke the start command without shell expansion, so
expressions like ``--port $PORT`` end up passing the literal string ``"$PORT"``
to uvicorn.  This module reads the environment directly, providing a sensible
default and logging when the value is malformed.
"""

from __future__ import annotations

import logging
import os

import uvicorn


logger = logging.getLogger(__name__)


def _resolve_port(default: int = 8000) -> int:
    """Return the port uvicorn should bind to, guarding against bad inputs."""
    raw = os.environ.get("PORT")
    if raw is None or raw == "":
        return default
    try:
        port = int(raw)
        if port <= 0:
            raise ValueError("Port must be positive")
        return port
    except (TypeError, ValueError):
        logger.warning("Invalid PORT=%r; falling back to %d", raw, default)
        return default


def main() -> None:
    port = _resolve_port()
    host = os.environ.get("HOST", "0.0.0.0")
    enrich = os.environ.get("ENRICH_SERVICE_URL", "")
    print(f"[start_server] PORT={port} ENRICH_SERVICE_URL={enrich!r}")
    uvicorn.run("src.agent_server:app", host=host, port=port)


if __name__ == "__main__":
    main()
