"""Guardrail checks applied before returning plans to clients."""

from __future__ import annotations

import math
from typing import Mapping, Sequence, Tuple


def _style_key(style: str) -> str:
    token = (style or "intraday").strip().lower()
    if token == "leap":
        token = "leaps"
    return token


def _risk_reward(entry: float, stop: float, target: float, direction: str) -> float:
    direction_norm = (direction or "long").lower()
    risk = entry - stop if direction_norm == "long" else stop - entry
    reward = target - entry if direction_norm == "long" else entry - target
    if risk <= 0:
        return -math.inf
    return reward / risk


def validate_plan(
    *,
    entry: float,
    stop: float,
    targets: Sequence[float],
    direction: str,
    atr: float,
    style: str,
    expected_move: float | None,
    remaining_atr: float | None,
    probabilities: Sequence[float],
    min_stop_atr: Mapping[str, float],
    max_stop_atr: Mapping[str, float],
    tp1_rr_min: Mapping[str, float],
    spacing_min_price: float,
) -> Tuple[bool, str]:
    """Validate key invariants for the upgraded plan logic."""

    if atr is None or atr <= 0:
        return False, "atr_missing"
    style_token = _style_key(style)
    risk_multiple = abs(entry - stop) / atr
    lower = min_stop_atr.get(style_token, 0.6)
    upper = max_stop_atr.get(style_token, 2.5)
    if risk_multiple < lower - 1e-6:
        return False, "stop_too_tight"
    if risk_multiple > upper + 1e-6:
        return False, "stop_too_wide"
    if not targets:
        return False, "no_targets"
    rr_floor = tp1_rr_min.get(style_token, 1.0)
    rr_tp1 = _risk_reward(entry, stop, targets[0], direction)
    if rr_tp1 < rr_floor - 1e-6:
        return False, "rr_below_min"
    spacing = float(spacing_min_price)
    for prev, current in zip(targets, targets[1:]):
        if abs(current - prev) < spacing - 1e-6:
            return False, "tp_spacing"
    cap_candidates = [value for value in (expected_move, remaining_atr) if isinstance(value, (int, float)) and value > 0]
    if cap_candidates:
        cap = min(cap_candidates)
        for target in targets:
            if abs(target - entry) > cap + 1e-6:
                return False, "tp_cap_exceeded"
    last_prob = 1.0
    for value in probabilities:
        try:
            prob = float(value)
        except (TypeError, ValueError):
            return False, "prob_invalid"
        if prob > last_prob + 1e-6:
            return False, "prob_not_monotone"
        last_prob = min(last_prob, prob)
    return True, "ok"


__all__ = ["validate_plan"]
