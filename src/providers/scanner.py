from __future__ import annotations

from typing import Any, Dict, List

from ..lib.data_source import DataRoute
from .geometry import GeometryBundle
from .series import SeriesBundle


async def scan(
    symbols: List[str],
    *,
    style: str,
    limit: int,
    series: SeriesBundle,
    geometry: GeometryBundle,
    route: DataRoute,
) -> Dict[str, Any]:
    """Return a deterministic scan page derived from geometry metrics."""
    count = min(len(symbols), limit)
    candidates = []
    for index, symbol in enumerate(symbols[:count], start=1):
        candidates.append(
            {
                "symbol": symbol,
                "rank": index,
                "score": 0.5,
                "reasons": ["synthetic"],
                "plan_id": f"{symbol}-{route.mode}",
                "expected_move": geometry.expected_move,
            }
        )
    return {
        "phase": "scan",
        "count_candidates": len(candidates),
        "next_cursor": None,
        "candidates": candidates,
        "snap_trace": ["scanner", route.mode],
        "data_quality": {
            "expected_move": geometry.expected_move,
            "remaining_atr": geometry.remaining_atr,
            "em_used": geometry.em_used,
        },
        "meta": {
            "style": style,
            "limit": limit,
            "route": route.mode,
        },
    }
