"""Structural level utilities used by the stop/target refactor."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, MutableMapping, Optional, Sequence, Tuple


_ROUND_INCREMENTS: Tuple[float, ...] = (10.0, 5.0, 1.0, 0.5, 0.25)


@dataclass(frozen=True)
class Level:
    """Normalised structural level."""

    name: str
    price: float


def _coerce_price(value: object) -> Optional[float]:
    if not isinstance(value, (int, float)):
        return None
    candidate = float(value)
    if not math.isfinite(candidate):
        return None
    return candidate


def _apply_precision(value: float, precision: Optional[int]) -> float:
    if precision is None:
        return float(value)
    return round(float(value), precision)


def normalise_levels(raw: Mapping[str, object], *, precision: Optional[int] = None) -> Dict[str, float]:
    """Return mapping of level name to float price."""

    clean: Dict[str, float] = {}
    for key, value in raw.items():
        price = _coerce_price(value)
        if price is None:
            continue
        clean[str(key)] = _apply_precision(price, precision)
    return clean


def _inject_round_numbers(
    levels: MutableMapping[str, float],
    *,
    anchor: float,
    atr: float,
    precision: Optional[int],
) -> None:
    if not math.isfinite(anchor):
        return
    atr_abs = abs(float(atr or 0.0))
    window = max(atr_abs * 3.0, 5.0)
    lower = anchor - window
    upper = anchor + window
    for increment in _ROUND_INCREMENTS:
        if precision is not None and increment < 10 ** -precision:
            continue
        base = math.floor(anchor / increment) * increment
        for offset in range(-6, 7):
            candidate = base + offset * increment
            if candidate <= 0:
                continue
            if candidate < lower or candidate > upper:
                continue
            name = f"round_{increment:g}_{offset:+d}"
            levels[name] = _apply_precision(candidate, precision)


def collect_levels(
    base_levels: Mapping[str, object],
    *,
    anchor: float,
    atr: float,
    precision: Optional[int] = None,
) -> Dict[str, float]:
    """Return structural levels enriched with round numbers near the anchor price."""

    clean = normalise_levels(base_levels, precision=precision)
    _inject_round_numbers(clean, anchor=anchor, atr=atr, precision=precision)
    return clean


def nearest_beyond(levels: Mapping[str, float], price: float, direction: str) -> Optional[Level]:
    """Return nearest level beyond the reference price in trade direction."""

    reference = _coerce_price(price)
    if reference is None:
        return None
    direction_norm = (direction or "long").lower()
    best_name: Optional[str] = None
    best_price: Optional[float] = None
    best_distance = float("inf")
    for name, value in levels.items():
        candidate = _coerce_price(value)
        if candidate is None:
            continue
        if direction_norm == "short":
            if candidate >= reference:
                continue
            distance = reference - candidate
        else:
            if candidate <= reference:
                continue
            distance = candidate - reference
        if distance < best_distance - 1e-9:
            best_name = name
            best_price = candidate
            best_distance = distance
    if best_name is None or best_price is None:
        return None
    return Level(best_name, best_price)


def structural_sequence(
    levels: Mapping[str, float],
    price: float,
    direction: str,
    *,
    include_equal: bool = False,
) -> Sequence[Level]:
    """Return ordered sequence of structural levels beyond reference price."""

    reference = _coerce_price(price)
    if reference is None:
        return ()
    direction_norm = (direction or "long").lower()
    seq: list[Level] = []
    for name, value in levels.items():
        candidate = _coerce_price(value)
        if candidate is None:
            continue
        if direction_norm == "short":
            if candidate > reference or (candidate == reference and not include_equal):
                continue
        else:
            if candidate < reference or (candidate == reference and not include_equal):
                continue
        seq.append(Level(name, candidate))
    seq.sort(key=lambda item: item.price, reverse=(direction_norm == "short"))
    return seq


def snap_price(
    level_price: float,
    *,
    direction: str,
    atr: float,
    pad_mult: float = 0.1,
    precision: Optional[int] = None,
) -> float:
    """Pad a level slightly beyond structure to respect invalidation."""

    atr_abs = abs(float(atr or 0.0))
    pad = atr_abs * pad_mult
    if pad <= 0:
        pad = 0.0
    if (direction or "long").lower() == "short":
        value = level_price + pad
    else:
        value = level_price - pad
    return _apply_precision(value, precision)


__all__ = [
    "Level",
    "collect_levels",
    "nearest_beyond",
    "normalise_levels",
    "snap_price",
    "structural_sequence",
]
