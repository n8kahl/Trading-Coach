"""Risk modelling utilities."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping, Optional, Sequence


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _distance(entry: float, target: float, direction: str) -> float:
    if direction == "short":
        return entry - target
    return target - entry


def _probability(probabilities: Mapping[str, float], key: str) -> float:
    try:
        value = float(probabilities.get(key, 0.0))
    except (TypeError, ValueError):
        value = 0.0
    return _clamp(value)


def compute_expected_value_r(
    entry: float,
    stop: float,
    targets: Sequence[float],
    probabilities: Mapping[str, float],
    direction: str,
) -> float:
    if not targets:
        return 0.0
    rr_stop = _distance(entry, stop, direction)
    if rr_stop <= 0:
        return 0.0

    cumulative = 0.0
    prev_prob = 0.0
    for idx, target in enumerate(targets[:3], start=1):
        key = f"tp{idx}"
        prob = _probability(probabilities, key)
        rr = _distance(entry, target, direction) / abs(rr_stop)
        incremental = max(prob - prev_prob, 0.0)
        cumulative += incremental * rr
        prev_prob = prob
    loss_prob = max(0.0, 1.0 - prev_prob)
    return round(cumulative - loss_prob, 3)


def compute_kelly_fraction(expected_value: float, probability_win: float, average_rr: float) -> float:
    if average_rr <= 0:
        return 0.0
    edge = probability_win * (average_rr + 1.0) - 1.0
    raw = edge / average_rr if average_rr else 0.0
    return round(_clamp(raw * 0.5), 3)


def project_mfe(style: str, atr_used: Optional[float]) -> str:
    multiplier = {
        "scalp": 1.2,
        "intraday": 1.6,
        "swing": 2.4,
        "leap": 3.0,
    }.get(style, 1.6)
    if atr_used is None:
        return f"≈{multiplier:.1f}× ATR (heuristic)"
    return f"≈{multiplier:.1f}× ATR before reversal"


def risk_model_payload(
    *,
    entry: float,
    stop: float,
    targets: Sequence[float],
    probabilities: Mapping[str, float],
    direction: str,
    atr_used: Optional[float],
    style: str,
) -> Dict[str, float | str]:
    expected_value = compute_expected_value_r(entry, stop, targets, probabilities, direction)
    avg_rr = 0.0
    if targets:
        rr_stop = abs(_distance(entry, stop, direction))
        if rr_stop > 0:
            rr_values = [abs(_distance(entry, target, direction) / rr_stop) for target in targets]
            avg_rr = sum(rr_values) / len(rr_values)
    kelly = compute_kelly_fraction(expected_value + 1.0, _probability(probabilities, "tp1"), max(avg_rr, 1e-6))
    mfe_label = project_mfe(style, atr_used)
    return {
        "expected_value_r": round(expected_value, 3),
        "kelly_fraction": round(kelly, 3),
        "mfe_projection": mfe_label,
    }


__all__ = ["risk_model_payload"]
