from __future__ import annotations

from typing import Any, Dict

from .geometry import GeometryBundle


def backfill_key_levels_used(symbol: str, geometry: GeometryBundle | None) -> Dict[str, Any]:
    """Return a baseline key-level structure when upstream data is missing."""
    if geometry:
        detail = geometry.get(symbol)
        if detail and detail.key_levels_used:
            return detail.key_levels_used
        fallback = geometry.key_levels
        if isinstance(fallback, dict) and fallback:
            return fallback
    return {"session": {}, "structural": {}}
