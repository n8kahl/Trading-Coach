"""Shared helpers for selecting structural trade entries."""

from __future__ import annotations

import math
from typing import Mapping, Optional

from ..levels.snapper import collect_levels
from ..strategy_library import normalize_style_input

_STYLE_MULTIPLIERS: Mapping[str, float] = {
    "scalp": 1.2,
    "intraday": 1.8,
    "swing": 3.5,
    "leaps": 4.5,
}


def _normalize_style(style: Optional[str]) -> str:
    token = normalize_style_input(style) or "intraday"
    if token == "leap":
        return "leaps"
    return token


def _entry_window(style: Optional[str], atr_value: float, expected_move: Optional[float], reference: float) -> float:
    """Compute the max distance we are willing to anchor the entry away from close."""

    style_token = _normalize_style(style)
    atr_val = float(atr_value or 0.0)
    reference_abs = abs(float(reference) if isinstance(reference, (int, float)) else 0.0)
    atr_cap = atr_val * _STYLE_MULTIPLIERS.get(style_token, 2.0)
    base_cap = max(reference_abs * 0.01, 0.25)
    max_distance = max(base_cap, atr_cap)

    if isinstance(expected_move, (int, float)) and math.isfinite(expected_move) and expected_move > 0:
        em_val = float(expected_move)
        if style_token in {"scalp", "intraday"}:
            max_distance = max(max_distance, em_val * 0.75)
        else:
            max_distance = max(max_distance, em_val)

    return max(max_distance, 0.25)


def select_structural_entry(
    *,
    direction: str,
    style: Optional[str],
    close_price: float,
    levels: Mapping[str, float],
    atr: float,
    expected_move: Optional[float],
) -> float:
    """Choose an entry anchor aligned with the trade bias and nearby structure."""

    try:
        close_value = float(close_price)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(close_value):
        return 0.0

    bias = (direction or "").strip().lower()
    if bias not in {"long", "short"}:
        return round(close_value, 2)

    window = _entry_window(style, atr, expected_move, close_value)
    level_objs = collect_levels(levels)

    def _within_window(level_price: float) -> bool:
        if not isinstance(level_price, (int, float)) or not math.isfinite(level_price):
            return False
        return abs(float(level_price) - close_value) <= window + 1e-6

    if bias == "long":
        candidates = [level for level in level_objs if level.price <= close_value + 1e-9 and _within_window(level.price)]
        sort_key = lambda lvl: (-lvl.priority, abs(close_value - lvl.price))
    else:
        candidates = [level for level in level_objs if level.price >= close_value - 1e-9 and _within_window(level.price)]
        sort_key = lambda lvl: (-lvl.priority, abs(close_value - lvl.price))

    if candidates:
        candidates.sort(key=sort_key)
        return round(float(candidates[0].price), 2)
    return round(close_value, 2)


__all__ = ["select_structural_entry"]
