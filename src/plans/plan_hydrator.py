"""Helpers for enriching plan dictionaries with entry state metadata."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Sequence


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _normalize_style(style: str | None) -> str:
    token = (style or "").strip().lower()
    if token == "leap":
        return "leaps"
    return token


def compute_entry_status(
    plan: Mapping[str, Any],
    last_price: Any,
    atr: Any,
    style: str | None,
) -> Dict[str, Any]:
    """Classify the plan as waiting/triggered/late relative to the last price."""
    entry_val = _as_float(plan.get("entry"))
    if entry_val is None or entry_val == 0:
        return {"state": "unknown", "dist_pct": None, "dist_atr": None}

    last_val = _as_float(last_price)
    if last_val is None:
        return {"state": "unknown", "dist_pct": None, "dist_atr": None}

    atr_val = _as_float(atr) or 0.0
    side = str(plan.get("bias") or plan.get("direction") or "long").lower()
    style_norm = _normalize_style(style)

    dist_abs = (last_val - entry_val) if side == "long" else (entry_val - last_val)
    dist_pct = dist_abs / entry_val if entry_val else 0.0
    dist_atr = dist_abs / atr_val if atr_val else float("inf")

    pct_thr = 0.003 if style_norm in {"scalp", "intraday"} else 0.008
    atr_thr = 0.35 if style_norm in {"scalp", "intraday"} else 0.60

    if (side == "long" and last_val < entry_val) or (side == "short" and last_val > entry_val):
        return {
            "state": "waiting",
            "dist_pct": dist_pct,
            "dist_atr": dist_atr if math.isfinite(dist_atr) else None,
        }

    state = "late" if (dist_pct >= pct_thr or (math.isfinite(dist_atr) and dist_atr >= atr_thr)) else "triggered"
    return {
        "state": state,
        "dist_pct": dist_pct,
        "dist_atr": dist_atr if math.isfinite(dist_atr) else None,
    }


def _resolve_level(context: Mapping[str, Any], keys: Sequence[str]) -> float | None:
    for key in keys:
        candidate = context.get(key)
        val = _as_float(candidate)
        if val is not None:
            return val
    return None


def build_reentry_cues(plan: Mapping[str, Any], context: Mapping[str, Any] | None) -> list[Dict[str, Any]]:
    """Suggest structured re-entry levels for plans that are already through the trigger."""
    cues: list[Dict[str, Any]] = []
    entry_val = _as_float(plan.get("entry"))
    if entry_val is None:
        return cues

    context_map: Mapping[str, Any] = context or {}
    key_levels = context_map.get("key_levels") or context_map.get("key") or {}
    if isinstance(key_levels, Mapping):
        context_map = dict(context_map)
        context_map.setdefault("prev_high", key_levels.get("prev_high"))

    ema20_val = _as_float(context_map.get("ema20"))
    if ema20_val is not None:
        cues.append({"label": "EMA20 pullback", "level": round(ema20_val, 2), "type": "dynamic"})

    pdh_val = _as_float(context_map.get("prev_high"))
    if pdh_val is not None:
        cues.append({"label": "PDH retest", "level": round(pdh_val, 2), "type": "struct"})

    cues.append({"label": "Entry retest", "level": round(entry_val, 2), "type": "struct"})
    return cues


__all__ = ["compute_entry_status", "build_reentry_cues"]
