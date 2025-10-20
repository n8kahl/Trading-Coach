"""Snapping utilities for aligning prices with contextual levels."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Level:
    tag: str
    price: float
    priority: int


@dataclass(frozen=True)
class SnapContext:
    side: str
    style: str
    strategy: Optional[str]
    window_atr: float
    window_pct: float
    rr_min: float
    entry: float


def collect_levels(level_map: Mapping[str, float]) -> List[Level]:
    priority_order = [
        ("gap_fill", 10),
        ("pdh", 9),
        ("pdl", 9),
        ("pdc", 8),
        ("orh", 8),
        ("orl", 8),
        ("vah", 7),
        ("val", 7),
        ("poc", 7),
        ("vwap", 6),
        ("avwap", 6),
        ("swing_high", 5),
        ("swing_low", 5),
        ("session_high", 4),
        ("session_low", 4),
    ]
    levels: List[Level] = []
    for tag, priority in priority_order:
        price = level_map.get(tag)
        if isinstance(price, (int, float)) and math.isfinite(price):
            levels.append(Level(tag=tag, price=float(price), priority=priority))
    return levels


def _within_window(price: float, candidate: Level, ctx: SnapContext) -> bool:
    window_abs = ctx.window_atr
    window_pct = ctx.window_pct * ctx.entry
    window = max(window_abs, window_pct)
    return abs(candidate.price - price) <= window


def _monotonic(
    snapped: Sequence[Tuple[float, Optional[str]]],
    side: str,
) -> bool:
    if not snapped:
        return True
    prices = [value for value, _ in snapped]
    if side == "long":
        return all(prices[i] < prices[i + 1] for i in range(len(prices) - 1))
    return all(prices[i] > prices[i + 1] for i in range(len(prices) - 1))


def snap_price(
    price: float,
    *,
    levels: Iterable[Level],
    ctx: SnapContext,
) -> Tuple[float, Optional[str]]:
    candidates = [level for level in levels if _within_window(price, level, ctx)]
    if not candidates:
        return price, None
    candidates.sort(key=lambda lvl: (-lvl.priority, abs(lvl.price - price)))
    chosen = candidates[0]
    return round(chosen.price, 2), chosen.tag


def snap_prices(
    entry: float,
    stop: float,
    targets: Sequence[float],
    *,
    levels: Iterable[Level],
    ctx: SnapContext,
) -> Tuple[float, float, List[Tuple[float, Optional[str]]]]:
    level_list = list(levels)
    snapped_stop, stop_tag = snap_price(stop, levels=level_list, ctx=ctx)
    snapped_targets: List[Tuple[float, Optional[str]]] = []
    for target in targets:
        snapped_price, tag = snap_price(target, levels=level_list, ctx=ctx)
        snapped_targets.append((snapped_price, tag))
    if not _monotonic([(stop, stop_tag)] + snapped_targets, ctx.side):
        return stop, stop_tag, [(target, tag) for target, tag in zip(targets, [None] * len(targets))]
    return snapped_stop, stop_tag, snapped_targets
