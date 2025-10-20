"""Entry candidate scoring based on structure and liquidity metrics."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Tuple

STRUCTURE_WEIGHTS = {
    "ORH": 1.0,
    "ORL": 1.0,
    "SESSION_HIGH": 0.95,
    "SESSION_LOW": 0.95,
    "PREV_HIGH": 0.9,
    "PREV_LOW": 0.9,
    "PREV_CLOSE": 0.85,
    "VWAP": 0.8,
    "VAH": 0.85,
    "VAL": 0.85,
    "POC": 0.8,
}

LEVEL_SEQUENCE = [
    ("ORL", "opening_range_low"),
    ("ORH", "opening_range_high"),
    ("SESSION_LOW", "session_low"),
    ("SESSION_HIGH", "session_high"),
    ("PREV_LOW", "prev_low"),
    ("PREV_HIGH", "prev_high"),
    ("PREV_CLOSE", "prev_close"),
    ("VAH", "vah"),
    ("VAL", "val"),
    ("POC", "poc"),
    ("VWAP", "vwap"),
]


def actionability_score(
    entry_distance_pct: float,
    actionable_soon: bool,
    rvol: float,
    liquidity_rank_norm: float,
    structure_quality: float,
) -> float:
    """
    Weighted actionability score per spec.
    """

    entry_term = max(0.0, min(1.0, 1.0 - entry_distance_pct))
    actionable_term = 1.0 if actionable_soon else 0.0
    rvol_term = min(1.0, max(0.0, rvol / 2.0))
    liquidity_term = max(0.0, min(1.0, liquidity_rank_norm))
    structure_term = max(0.0, min(1.0, structure_quality))
    score = (
        0.40 * entry_term
        + 0.20 * actionable_term
        + 0.15 * rvol_term
        + 0.15 * liquidity_term
        + 0.10 * structure_term
    )
    return round(score, 4)


def _liquidity_rank_norm(value: float | None) -> float:
    if value is None or not math.isfinite(value):
        return 0.5
    # Market ranks 1..100 -> convert to 0..1 (higher better)
    try:
        norm = 1.0 - min(max(float(value) - 1.0, 0.0), 99.0) / 99.0
    except (TypeError, ValueError):
        norm = 0.5
    return max(0.0, min(1.0, norm))


def compute_entry_candidates(
    symbol: str,
    style: str,
    levels: Dict[str, float],
    indicators: Dict[str, float],
    prices: Dict[str, float],
) -> List[Dict[str, float | str]]:
    """
    Produce deterministic list of entry candidates scored by actionability.
    """

    close_price = prices.get("close")
    if not isinstance(close_price, (int, float)) or close_price <= 0:
        return []
    close_price = float(close_price)

    rvol = indicators.get("rvol") if isinstance(indicators.get("rvol"), (int, float)) else 1.0
    liquidity_rank = indicators.get("liquidity_rank") if isinstance(indicators.get("liquidity_rank"), (int, float)) else None
    liquidity_norm = _liquidity_rank_norm(liquidity_rank)

    candidates: List[Dict[str, float | str]] = []
    seen: set[str] = set()

    for label, key in LEVEL_SEQUENCE:
        label_upper = label.upper()
        if label_upper in seen:
            continue
        level_price = levels.get(key)
        if level_price is None and key == "vwap":
            level_price = prices.get("vwap")
        if level_price is None:
            continue
        try:
            level_price = float(level_price)
        except (TypeError, ValueError):
            continue
        if level_price <= 0:
            continue

        distance_pct = abs(level_price - close_price) / close_price
        actionable_soon = distance_pct <= 0.003  # ~0.3%
        bars_to_trigger = max(int(round(distance_pct * 400, 0)), 0)
        structure_quality = STRUCTURE_WEIGHTS.get(label_upper, 0.7)

        score = actionability_score(
            entry_distance_pct=distance_pct,
            actionable_soon=actionable_soon,
            rvol=float(rvol or 1.0),
            liquidity_rank_norm=liquidity_norm,
            structure_quality=structure_quality,
        )

        candidates.append(
            {
                "level": round(level_price, 2),
                "label": label_upper,
                "type": label_upper,
                "bars_to_trigger": bars_to_trigger,
                "entry_distance_pct": round(distance_pct, 4),
                "score": score,
                "structure_quality": round(structure_quality, 3),
            }
        )
        seen.add(label_upper)

    candidates.sort(key=lambda item: (-item["score"], item["entry_distance_pct"]))  # type: ignore[index]
    return candidates


__all__ = ["actionability_score", "compute_entry_candidates"]
