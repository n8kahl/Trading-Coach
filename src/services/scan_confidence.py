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
    if direction == "long":
        return 0.85 if fast >= slow else 0.55
    if direction == "short":
        return 0.85 if fast <= slow else 0.55
    return None


def _liquidity_component(plan_obj: Dict[str, Any]) -> Optional[float]:
    contracts = plan_obj.get("options_contracts")
    note = plan_obj.get("options_note")
    rejected = plan_obj.get("rejected_contracts")
    if isinstance(contracts, list) and contracts:
        return 0.72
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
    if ratio < 0.006:
        return 0.56
    if ratio < 0.018:
        return 0.64
    if ratio < 0.035:
        return 0.68
    return 0.6


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
    for value in dq_flags.values():
        if value is False:
            return None
    direction = str(plan_obj.get("bias") or detail.bias or "").lower()
    if direction not in {"long", "short"}:
        return None
    if planning_context not in {"live", "frozen"}:
        return None
    trend = _trend_component(detail, direction)
    volatility = _volatility_component(detail)
    if trend is None or volatility is None:
        return None
    liquidity = _liquidity_component(plan_obj) or 0.6
    components = {
        "trend_alignment": trend,
        "liquidity_structure": liquidity,
        "volatility_regime": volatility,
    }
    score = overall_confidence(components)
    return max(0.0, min(1.0, score))


__all__ = ["compute_scan_confidence"]
