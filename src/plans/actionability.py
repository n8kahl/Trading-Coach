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
    "SWING_LOW": 0.9,
    "SWING_HIGH": 0.9,
    "EMA9": 0.7,
    "EMA21": 0.65,
    "EMA50": 0.6,
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
    ("SWING_LOW", "swing_low"),
    ("SWING_HIGH", "swing_high"),
    ("EMA9", "ema9"),
    ("EMA21", "ema21"),
    ("EMA50", "ema50"),
]


def _infer_tick_size(price: float) -> float:
    if price >= 500:
        return 0.1
    if price >= 200:
        return 0.05
    if price >= 50:
        return 0.02
    if price >= 10:
        return 0.01
    if price >= 1:
        return 0.005
    return 0.001


def is_actionable_soon(level_price: float, last_price: float, atr: float, tick: float, style: str) -> bool:
    style = (style or "").lower()
    pct_cap = {"scalp": 0.0015, "intraday": 0.0025, "swing": 0.004, "leaps": 0.006}.get(style, 0.0025)
    atr_cap = {"scalp": 0.25, "intraday": 0.40, "swing": 0.70, "leaps": 1.00}.get(style, 0.40)

    close = float(last_price)
    atr = max(float(atr or 0.0), float(tick or 0.01))
    distance_pct = abs(level_price - close) / max(close, 1e-9)
    distance_atr = abs(level_price - close) / atr
    return (distance_pct <= pct_cap) or (distance_atr <= atr_cap)


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
    atr_value = indicators.get("atr") or indicators.get("atr14") or indicators.get("atr_14") or indicators.get("atr_tf")
    if not isinstance(atr_value, (int, float)) or not math.isfinite(atr_value) or atr_value <= 0:
        atr_value = close_price * 0.006
    atr_value = max(float(atr_value), close_price * 0.0005)
    tick_hint = prices.get("tick") or prices.get("tick_size")
    if not isinstance(tick_hint, (int, float)) or tick_hint <= 0:
        tick_hint = _infer_tick_size(close_price)
    tick_size = float(tick_hint)

    candidates: List[Dict[str, float | str]] = []
    seen: set[str] = set()

    for label, key in LEVEL_SEQUENCE:
        label_upper = label.upper()
        if label_upper in seen:
            continue
        level_price = levels.get(key)
        if level_price is None:
            level_price = prices.get(key)
        if level_price is None:
            continue
        try:
            level_price = float(level_price)
        except (TypeError, ValueError):
            continue
        if level_price <= 0:
            continue

        distance_pct = abs(level_price - close_price) / close_price
        distance_atr = abs(level_price - close_price) / atr_value if atr_value > 0 else float("inf")
        actionable_soon = is_actionable_soon(level_price, close_price, atr_value, tick_size, style)
        bars_to_trigger = max(int(round(distance_atr * 2.0)), 0)
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
                "entry_distance_atr": round(distance_atr, 3),
                "actionable_soon": actionable_soon,
                "score": score,
                "structure_quality": round(structure_quality, 3),
                "evaluation": {
                    "actionability": score,
                    "distance_pct": round(distance_pct, 4),
                    "distance_atr": round(distance_atr, 3),
                },
            }
        )
        seen.add(label_upper)

    candidates.sort(
        key=lambda item: (
            -float(item.get("score", 0.0)),  # type: ignore[arg-type]
            float(item.get("bars_to_trigger", 0)),
            float(item.get("entry_distance_atr", 0.0)),
            float(item.get("entry_distance_pct", 0.0)),
        )
    )
    return candidates


__all__ = ["actionability_score", "compute_entry_candidates", "is_actionable_soon"]
