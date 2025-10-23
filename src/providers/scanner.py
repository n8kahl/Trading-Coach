from __future__ import annotations

from typing import Any, Dict, List

from ..lib.data_route import DataRoute
from .geometry import GeometryBundle
from .series import SeriesBundle


def _score_symbol(symbol: str, bundle: GeometryBundle) -> float:
    detail = bundle.get(symbol)
    if detail is None:
        return 0.0
    expected_move = detail.expected_move or 0.0
    price = detail.last_close or 0.0
    if price <= 0.0:
        return 0.0
    base = expected_move / price
    bias_bonus = 0.02 if detail.bias == "long" else 0.01
    return (base + bias_bonus) * 100.0


async def scan(
    symbols: List[str],
    *,
    style: str,
    limit: int,
    series: SeriesBundle,
    geometry: GeometryBundle,
    route: DataRoute,
) -> Dict[str, Any]:
    """Rank symbols using geometry metrics and return a scan page."""

    scored: List[tuple[float, str]] = []
    for symbol in symbols:
        score = _score_symbol(symbol, geometry)
        scored.append((score, symbol))

    scored.sort(key=lambda item: (-item[0], item[1]))
    selected = scored[: max(1, min(limit, len(scored)))]

    candidates: List[Dict[str, Any]] = []
    for index, (_, symbol) in enumerate(selected, start=1):
        detail = geometry.get(symbol)
        reasons: List[str] = []
        if detail and detail.expected_move:
            reasons.append(f"ATR {detail.expected_move:.2f}")
        if detail and detail.bias:
            reasons.append(f"Bias {detail.bias}")
        candidates.append(
            {
                "symbol": symbol,
                "rank": index,
                "score": _score_symbol(symbol, geometry),
                "reasons": reasons,
                "snap_trace": detail.snap_trace if detail else [],
                "key_levels_used": detail.key_levels_used if detail else {},
                "entry": None,
                "stop": None,
                "tps": [],
            }
        )

    page = {
        "phase": "scan",
        "count_candidates": len(candidates),
        "next_cursor": None,
        "candidates": candidates,
        "snap_trace": ["scanner:atr_rank"],
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
    return page


__all__ = ["scan"]
