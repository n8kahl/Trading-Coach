from __future__ import annotations

from typing import Any, Dict

from ..lib.data_source import DataRoute
from .geometry import GeometryBundle
from .series import SeriesBundle


async def plan(
    symbol: str,
    *,
    series: SeriesBundle,
    geometry: GeometryBundle,
    route: DataRoute,
) -> Dict[str, Any]:
    """Produce a synthetic plan document from geometry inputs."""
    return {
        "plan_id": f"{symbol}-{route.mode}",
        "version": 1,
        "trade_detail": f"Synthetic plan for {symbol} via {route.mode}",
        "symbol": symbol,
        "entry_candidates": [],
        "warnings": [],
        "key_levels_used": geometry.key_levels,
        "snap_trace": ["planner", route.mode],
        "data_quality": {
            "expected_move": geometry.expected_move,
            "remaining_atr": geometry.remaining_atr,
            "em_used": geometry.em_used,
        },
    }
