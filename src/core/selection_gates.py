"""Lenient, style-aware gating utilities for scan candidates."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from ..config import (
    LENIENCY_PENALTY_SCALE,
    LENIENT_GATES_DEFAULT,
    STYLE_GATES,
    VOLATILITY_RELAX_VIX,
)

CandidateDict = Dict[str, Any]
GateConfig = Dict[str, Dict[str, float]]


def style_cfg(style: str | None, cfg: GateConfig) -> Dict[str, float]:
    """Return the gate configuration for a given style."""
    style_key = (style or "intraday").lower()
    return cfg.get(style_key, cfg["intraday"])


def within_hard_caps(candidate: CandidateDict, s_cfg: Dict[str, float]) -> bool:
    """Check if the candidate respects hard ATR/bars caps."""
    d_atr = candidate.get("entry_distance_atr")
    bars = candidate.get("bars_to_trigger")
    if d_atr is None or bars is None:
        return True
    try:
        d_atr_val = float(d_atr)
        bars_val = float(bars)
    except (TypeError, ValueError):
        return False
    return d_atr_val <= s_cfg["hard_atr_cap"] and bars_val <= s_cfg["hard_bars_cap"]


def _soft_ratio(value: float | None, pref: float, hard: float) -> float:
    if value is None:
        return 0.0
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return 0.0
    if value_f <= pref:
        return 0.0
    if hard <= pref:
        return 1.0
    ratio = (value_f - pref) / (hard - pref)
    return max(0.0, min(1.0, ratio))


def soft_penalty(candidate: CandidateDict, s_cfg: Dict[str, float], *, scale: float) -> float:
    """Compute lenient penalty between preferred and hard caps."""
    d_atr_ratio = _soft_ratio(candidate.get("entry_distance_atr"), s_cfg["pref_atr"], s_cfg["hard_atr_cap"])
    bars_ratio = _soft_ratio(candidate.get("bars_to_trigger"), s_cfg["pref_bars"], s_cfg["hard_bars_cap"])
    return scale * 0.5 * (d_atr_ratio + bars_ratio)


def apply_lenient_gate(
    candidate: CandidateDict,
    style: str | None,
    cfg: GateConfig = STYLE_GATES,
    *,
    lenient: bool = LENIENT_GATES_DEFAULT,
    penalty_scale: float = LENIENCY_PENALTY_SCALE,
) -> Tuple[bool, float]:
    """Apply lenient gating and return (is_allowed, penalty)."""
    s_cfg = style_cfg(style, cfg)
    vol_proxy = candidate.get("vol_proxy") or 0.0
    try:
        vol_value = float(vol_proxy)
    except (TypeError, ValueError):
        vol_value = 0.0
    if vol_value > VOLATILITY_RELAX_VIX:
        s_cfg = {
            **s_cfg,
            "hard_atr_cap": s_cfg["hard_atr_cap"] * 1.15,
            "hard_bars_cap": s_cfg["hard_bars_cap"] + 1,
        }

    if not within_hard_caps(candidate, s_cfg):
        candidate["_gate_reject_reason"] = "beyond_hard_caps"
        return False, 0.0

    penalty = soft_penalty(candidate, s_cfg, scale=penalty_scale) if lenient else 0.0
    candidate["_gate_penalty"] = round(penalty, 4)
    return True, penalty


__all__ = [
    "CandidateDict",
    "GateConfig",
    "apply_lenient_gate",
    "soft_penalty",
    "style_cfg",
    "within_hard_caps",
]
