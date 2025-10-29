"""Structural level helpers for snapping stops and targets."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Sequence, Tuple

STRUCTURAL_ORDER = [
    "orh",
    "opening_range_high",
    "orl",
    "opening_range_low",
    "vah",
    "val",
    "poc",
    "hvn",
    "lvn",
    "vwap",
    "vwap_upper_1",
    "vwap_lower_1",
    "vwap_upper_2",
    "vwap_lower_2",
    "vwap_band_high_1",
    "vwap_band_low_1",
    "vwap_band_high_2",
    "vwap_band_low_2",
    "vwap_1sd_high",
    "vwap_1sd_low",
    "vwap_2sd_high",
    "vwap_2sd_low",
    "avwap",
    "avwap_session_open",
    "avwap_prev_high",
    "avwap_prev_low",
    "avwap_prev_close",
    "pdc",
    "prev_close",
    "previous_close",
    "prev_high",
    "prev_low",
    "micro_swing_high_5m",
    "micro_swing_low_5m",
    "micro_swing_high_1m",
    "micro_swing_low_1m",
    "swing_high_5m",
    "swing_low_5m",
    "swing_high_1m",
    "swing_low_1m",
    "swing_high",
    "swing_low",
    "pwh",
    "pwl",
    "pwc",
    "pdh",
    "pdl",
    "session_high",
    "session_low",
    "day_high",
    "day_low",
    "gap_top",
    "gap_bottom",
]


def _normalise_key(key: str) -> str:
    return key.lower().strip()


def directional_nodes(levels: Dict[str, float], direction: str, entry: float) -> List[Tuple[float, str]]:
    """
    Return sorted [(price,label)] moving away from entry in the trade direction, nearest first.
    """

    direction = (direction or "").lower()
    nodes: List[Tuple[float, str]] = []
    entry_val = float(entry)
    seen: set[str] = set()

    def _extract_price(label: str) -> float | None:
        for candidate in {label, label.lower(), label.upper()}:
            value = levels.get(candidate)
            if value is None:
                continue
            try:
                price = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(price):
                continue
            return price
        return None

    def _append(label: str, price: float) -> None:
        norm = _normalise_key(label)
        if norm in seen:
            return
        if direction == "short":
            if price >= entry_val - 1e-9:
                return
        elif direction == "long":
            if price <= entry_val + 1e-9:
                return
        else:
            return
        nodes.append((price, label))
        seen.add(norm)

    for label in STRUCTURAL_ORDER:
        price = _extract_price(label)
        if price is None:
            continue
        _append(label, price)

    dynamic_tokens = ("fractal", "pivot", "micro", "swing_", "local_high", "local_low", "hvn", "lvn")
    for raw_label, raw_value in levels.items():
        norm = _normalise_key(str(raw_label))
        if norm in seen:
            continue
        if not any(token in norm for token in dynamic_tokens):
            continue
        try:
            price = float(raw_value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(price):
            continue
        _append(str(raw_label), price)

    nodes.sort(key=lambda pair: abs(pair[0] - entry_val))
    return nodes


def last_lower_high(levels: Dict[str, float], fallback: float | None = None) -> float | None:
    for key in ("swing_high", "session_high", "prev_high", "orh"):
        value = levels.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return fallback


def last_higher_low(levels: Dict[str, float], fallback: float | None = None) -> float | None:
    for key in ("swing_low", "session_low", "prev_low", "orl"):
        value = levels.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return fallback


def profile_nodes(levels: Dict[str, float]) -> List[str]:
    nodes = []
    for key in ("vah", "val", "poc", "hvn", "lvn"):
        if levels.get(key) is not None:
            nodes.append(key)
    return nodes


def populate_recent_extrema(
    levels: Dict[str, float],
    highs: Sequence[float],
    lows: Sequence[float],
    *,
    window: int = 5,
) -> None:
    """
    Ensure swing_high/swing_low exist using recent highs/lows when source data is available.
    """

    def _valid(value: float | None) -> bool:
        return isinstance(value, (int, float)) and math.isfinite(value)

    def _recent(series: Sequence[float], mode: str) -> float | None:
        values = [float(val) for val in series if _valid(val)]
        if not values:
            return None
        windowed = values[-window:] if len(values) >= window else values
        if mode == "max":
            return max(windowed)
        return min(windowed)

    if not _valid(levels.get("swing_high")):
        candidate = _recent(highs, "max")
        if _valid(candidate):
            levels["swing_high"] = float(candidate)

    if not _valid(levels.get("swing_low")):
        candidate = _recent(lows, "min")
        if _valid(candidate):
            levels["swing_low"] = float(candidate)


__all__ = [
    "STRUCTURAL_ORDER",
    "directional_nodes",
    "last_lower_high",
    "last_higher_low",
    "profile_nodes",
    "populate_recent_extrema",
]
