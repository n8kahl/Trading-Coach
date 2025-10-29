from __future__ import annotations

import math
from typing import Any, Dict, Optional

from ..app.engine.scoring import overall_confidence
from ..providers.geometry import GeometryDetail


def _trend_component(detail: GeometryDetail, direction: str) -> Optional[float]:
    fast = detail.ema_fast
    slow = detail.ema_slow
    if fast is None or slow is None:
        return None
    aligned = (fast >= slow) if direction == "long" else (fast <= slow)
    base = 0.55
    if aligned:
        base = 0.70
        try:
            sep = abs(float(fast) - float(slow)) / max(
                1e-6, float(detail.last_close or slow)
            )
        except Exception:
            sep = 0.0
        if sep >= 0.002:
            base += 0.05
        if sep >= 0.005:
            base += 0.05
    return max(0.0, min(1.0, base))


def _liquidity_component(plan_obj: Dict[str, Any]) -> Optional[float]:
    contracts = plan_obj.get("options_contracts")
    note = plan_obj.get("options_note")
    rejected = plan_obj.get("rejected_contracts")
    if isinstance(contracts, list) and contracts:
        top = contracts[0] or {}
        spread_pct = 0.0
        oi = 0.0
        vol = 0.0
        eff = 0.0
        if isinstance(top, dict):
            try:
                spread_pct = float(top.get("spread_pct") or 0.0)
            except (TypeError, ValueError):
                spread_pct = 0.0
            try:
                oi = float(top.get("oi") or top.get("open_interest") or 0.0)
            except (TypeError, ValueError):
                oi = 0.0
            try:
                vol = float(top.get("volume") or 0.0)
            except (TypeError, ValueError):
                vol = 0.0
            try:
                eff = float(top.get("option_efficiency") or 0.0)
            except (TypeError, ValueError):
                eff = 0.0
        score_spread = (
            0.75
            if spread_pct and spread_pct <= 0.02
            else 0.68
            if spread_pct and spread_pct <= 0.05
            else 0.60
        )
        score_depth = (
            0.75
            if (oi >= 1000 or vol >= 500)
            else 0.68
            if (oi >= 200 or vol >= 100)
            else 0.60
        )
        score_eff = eff if 0.55 <= eff <= 0.85 else (0.65 if eff > 0 else 0.65)
        mixed = 0.4 * score_spread + 0.3 * score_depth + 0.3 * score_eff
        return max(0.56, min(0.80, mixed))
    if isinstance(rejected, list) and rejected:
        return 0.56
    if isinstance(note, str) and note.strip():
        return 0.58
    return 0.6


def _volatility_component(detail: GeometryDetail) -> Optional[float]:
    expected = detail.expected_move
    price = detail.last_close
    if expected is None or price in (None, 0):
        return None
    try:
        ratio = float(expected) / float(price)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    if not math.isfinite(ratio) or ratio <= 0:
        return None
    style = (getattr(detail, "style", None) or "").lower()
    b1, b2, b3 = (
        (0.008, 0.020, 0.040)
        if style in {"scalp", "intraday"}
        else (0.006, 0.015, 0.030)
    )
    if ratio < b1:
        return 0.56
    if ratio < b2:
        return 0.64
    if ratio < b3:
        return 0.68
    return 0.60


def _momentum_component(detail: GeometryDetail) -> Optional[float]:
    """Lightweight momentum proxy from EMA separation."""
    fast = detail.ema_fast
    slow = detail.ema_slow
    price = detail.last_close
    if fast is None or slow is None or price in (None, 0):
        return None
    try:
        dist = abs(float(fast) - float(slow)) / float(price)
    except Exception:
        return None
    if dist < 0.001:
        return 0.52
    if dist < 0.003:
        return 0.58
    if dist < 0.006:
        return 0.65
    return 0.72


def compute_scan_confidence(
    *,
    detail: GeometryDetail,
    plan_obj: Dict[str, Any],
    data_quality: Dict[str, Any],
    planning_context: str,
    banner: Optional[str],
) -> Optional[float]:
    if banner:
        return None
    dq_flags = {
        "series_present": data_quality.get("series_present"),
        "indices_present": data_quality.get("indices_present"),
    }
    dq_penalty = 0.0
    for value in dq_flags.values():
        if value is False:
            dq_penalty += 0.06
    direction = str(plan_obj.get("bias") or detail.bias or "").lower()
    if direction not in {"long", "short"}:
        return None
    if planning_context not in {"live", "frozen"}:
        return None
    trend = _trend_component(detail, direction)
    volatility = _volatility_component(detail)
    momentum = _momentum_component(detail)
    if trend is None:
        trend = 0.58
        dq_penalty += 0.04
    if volatility is None:
        volatility = 0.60
        dq_penalty += 0.04
    liquidity = _liquidity_component(plan_obj) or 0.6
    components = {
        "trend_alignment": trend,
        "liquidity_structure": liquidity,
        "volatility_regime": volatility,
        "momentum_signal": momentum if momentum is not None else 0.58,
    }
    score = overall_confidence(components)
    score = max(0.0, min(1.0, score - dq_penalty))
    return score


__all__ = ["compute_scan_confidence"]
