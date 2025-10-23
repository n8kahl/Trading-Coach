"""Runner trail policy helpers."""

from __future__ import annotations

import math
from typing import Dict, Iterable, Mapping, Optional


DEFAULT_FRACTIONS: Dict[str, float] = {
    "scalp": 0.18,
    "intraday": 0.22,
    "swing": 0.20,
    "leaps": 0.18,
}


def _style_key(style: str) -> str:
    token = (style or "intraday").strip().lower()
    if token == "leap":
        token = "leaps"
    return token


def build_runner_policy(
    *,
    entry: float,
    targets: Iterable[float],
    direction: str,
    style: str,
    atr: float,
    ctx: Mapping[str, object],
    run_constants: Mapping[str, float],
    giveback_constants: Mapping[str, float],
) -> Dict[str, object]:
    """Construct a hybrid runner policy ready to serialise."""

    direction_norm = (direction or "long").lower()
    style_token = _style_key(style)
    atr_abs = abs(float(atr or 0.0)) or 1.0
    trail_mult = float(run_constants.get(style_token, 1.0))
    giveback = float(giveback_constants.get(style_token, 0.5))
    fraction = DEFAULT_FRACTIONS.get(style_token, 0.2)

    targets_list = [float(tp) for tp in targets if isinstance(tp, (int, float))]
    tp1 = targets_list[0] if targets_list else (entry + atr_abs if direction_norm == "long" else entry - atr_abs)

    closes = ctx.get("closes")
    if isinstance(closes, Iterable):
        cleaned = [float(val) for val in closes if isinstance(val, (int, float)) and math.isfinite(val)]
    else:
        cleaned = []

    if direction_norm == "short":
        ref = min(cleaned + [tp1]) if cleaned else tp1
        chandelier = ref + trail_mult * atr_abs
    else:
        ref = max(cleaned + [tp1]) if cleaned else tp1
        chandelier = ref - trail_mult * atr_abs

    structure_levels = ctx.get("structure_levels") if isinstance(ctx.get("structure_levels"), Mapping) else {}
    structure_price = None
    if isinstance(structure_levels, Mapping):
        preferred_keys = ["reclaim_orh", "reject_orl", "swing"]
        for key in preferred_keys:
            value = structure_levels.get(key)
            if isinstance(value, (int, float)) and math.isfinite(value):
                structure_price = float(value)
                break
    if structure_price is None:
        structure_price = entry

    payload: Dict[str, object] = {
        "fraction": round(fraction, 2),
        "atr_trail_mult": round(trail_mult, 2),
        "atr_trail_step": 0.4,
        "type": "hybrid",
        "start_after": round(tp1, 4),
        "chandelier": {
            "basis": round(ref, 4),
            "mult": round(trail_mult, 2),
            "update_threshold_atr": 0.4,
        },
        "structure_ratchet": {
            "level": round(structure_price, 4),
            "pad_atr": 0.1,
        },
        "giveback": {
            "fraction": round(giveback, 2),
            "metric": "mfe",
        },
        "initial_trail": round(chandelier, 4),
        "notes": [
            "Trail using ATR chandelier while momentum persists.",
            "On momentum fade switch to structure ratchet with 0.1×ATR padding.",
            f"Clamp runner if drawdown exceeds {giveback:.0%} of MFE.",
        ],
    }

    momentum_note = ctx.get("momentum_debug")
    if isinstance(momentum_note, str) and momentum_note:
        payload["notes"].append(momentum_note)

    return payload


__all__ = ["build_runner_policy"]
