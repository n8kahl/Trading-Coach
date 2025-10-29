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
    pct = candidate.get("entry_distance_pct")
    hard_atr = s_cfg.get("hard_atr_cap")
    hard_bars = s_cfg.get("hard_bars_cap")
    hard_pct = s_cfg.get("hard_pct_cap")
    checks = []
    if hard_atr is not None and d_atr is not None:
        try:
            d_atr_val = float(d_atr)
        except (TypeError, ValueError):
            return False
        checks.append(d_atr_val <= float(hard_atr))
    if hard_bars is not None and bars is not None:
        try:
            bars_val = float(bars)
        except (TypeError, ValueError):
            return False
        checks.append(bars_val <= float(hard_bars))
    if hard_pct is not None and pct is not None:
        try:
            pct_val = float(pct)
        except (TypeError, ValueError):
            return False
        checks.append(pct_val <= float(hard_pct))
    if not checks:
        return True
    return all(checks)


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
    ratios = []
    pref_atr = s_cfg.get("pref_atr")
    hard_atr = s_cfg.get("hard_atr_cap")
    if pref_atr is not None and hard_atr is not None:
        ratios.append(_soft_ratio(candidate.get("entry_distance_atr"), float(pref_atr), float(hard_atr)))
    pref_bars = s_cfg.get("pref_bars")
    hard_bars = s_cfg.get("hard_bars_cap")
    if pref_bars is not None and hard_bars is not None:
        ratios.append(_soft_ratio(candidate.get("bars_to_trigger"), float(pref_bars), float(hard_bars)))
    pref_pct = s_cfg.get("pref_pct")
    hard_pct = s_cfg.get("hard_pct_cap")
    if pref_pct is not None and hard_pct is not None:
        ratios.append(_soft_ratio(candidate.get("entry_distance_pct"), float(pref_pct), float(hard_pct)))
    if not ratios:
        return 0.0
    return scale * (sum(ratios) / len(ratios))


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
