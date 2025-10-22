from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .series import SeriesBundle


@dataclass(slots=True)
class GeometryBundle:
    symbols: List[str]
    series: SeriesBundle
    expected_move: float | None = None
    remaining_atr: float | None = None
    em_used: bool | None = None
    snap_trace: List[str] | None = field(default_factory=lambda: ["geometry:synthetic"])
    key_levels: Dict[str, Dict[str, float]] | None = field(
        default_factory=lambda: {"session": {}, "structural": {}}
    )


async def build_geometry(symbols: List[str], series: SeriesBundle) -> GeometryBundle:
    """Derive a simple geometry bundle from the provided series."""
    expected_move = 1.5 if series.mode == "live" else 1.2
    remaining_atr = 2.5 if series.mode == "live" else 2.0
    em_used = series.mode == "live"
    return GeometryBundle(
        symbols=list(symbols),
        series=series,
        expected_move=expected_move,
        remaining_atr=remaining_atr,
        em_used=em_used,
        snap_trace=["geometry", series.mode],
        key_levels={"session": {"vwap": 100.0}, "structural": {"s1": 90.0, "r1": 110.0}},
    )
