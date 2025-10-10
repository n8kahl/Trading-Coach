"""Entrypoint for the enrichment sidecar with safe PORT resolution."""

from __future__ import annotations

import os
import uvicorn


def _resolve_port(default: int = 8080) -> int:
    raw = os.environ.get("PORT")
    try:
        return int(raw) if raw else default
    except (TypeError, ValueError):
        return default


def main() -> None:
    port = _resolve_port()
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run("src.enrich_service:app", host=host, port=port)


if __name__ == "__main__":
    main()

