from __future__ import annotations

from typing import Any, Dict

from .geometry import GeometryBundle


def backfill_key_levels_used(symbol: str, geometry: GeometryBundle | None) -> Dict[str, Any]:
    """Return a baseline key-level structure when upstream data is missing."""
    if geometry and geometry.key_levels:
        return geometry.key_levels
    return {"session": [], "structural": []}
