"""Wrapper to run `uvicorn src.finnhub_sidecar:app` on managed hosts.

This re-exports the app defined at the repository root in `finnhub_sidecar.py`.
"""

from finnhub_sidecar import app  # noqa: F401

__all__ = ["app"]

