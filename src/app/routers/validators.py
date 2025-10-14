"""Validation helpers for router responses."""

from __future__ import annotations

import os

_DEFAULT_CHART_BASE = "https://app.fancytrader.io/chart"
_CANONICAL_BASE = os.getenv("CHART_BASE", _DEFAULT_CHART_BASE).split("?", 1)[0].rstrip("?")
_CANONICAL_PREFIX = f"{_CANONICAL_BASE}?"


def canonical_chart_url(url: str | None) -> bool:
    """Return True if the provided URL points at the canonical Fancy Trader chart."""
    return isinstance(url, str) and url.startswith(_CANONICAL_PREFIX)


__all__ = ["canonical_chart_url"]
