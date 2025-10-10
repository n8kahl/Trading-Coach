"""Thin wrapper so you can run `uvicorn src.enrich_service:app`.

This re-exports the `app` defined at the repository root in `enrich_service.py`.
Using the `src.*` module path tends to be more reliable on managed hosts.
"""

from enrich_service import app  # noqa: F401  (re-export)

__all__ = ["app"]

