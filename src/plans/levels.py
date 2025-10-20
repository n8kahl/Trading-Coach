"""Structural level helpers for snapping stops and targets."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

STRUCTURAL_ORDER = [
    "orh",
    "orl",
    "session_high",
    "session_low",
    "prev_high",
    "prev_low",
    "prev_close",
    "vah",
    "val",
    "poc",
    "hvn",
    "lvn",
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

    for label in STRUCTURAL_ORDER:
        value = levels.get(label) or levels.get(label.upper())
        if value is None:
            continue
        try:
            price = float(value)
        except (TypeError, ValueError):
            continue
        if direction == "short" and price < entry_val:
            nodes.append((price, label))
        elif direction == "long" and price > entry_val:
            nodes.append((price, label))
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


__all__ = ["STRUCTURAL_ORDER", "directional_nodes", "last_lower_high", "last_higher_low", "profile_nodes"]
