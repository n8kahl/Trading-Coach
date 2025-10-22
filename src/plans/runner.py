"""Runner policy helpers."""

from __future__ import annotations

from typing import Iterable, List, Set


def compute_runner(
    entry: float,
    tps: List[float],
    style: str,
    em_points: float,
    atr: float,
    profile_nodes: Iterable[str],
) -> dict:
    """
    Runner starts only after TP1 fill and respects EM/ATR guardrails.
    """

    style_token = (style or "").strip().lower()
    if style_token == "leap":
        style_token = "leaps"
    atr = abs(float(atr or 0.0))
    em_points = max(float(em_points or 0.0), 0.0)
    profile_nodes = {str(node).upper() for node in profile_nodes if node}

    if style_token in {"swing", "leaps"}:
        fraction = 0.15
        trail_mult = 0.8
    else:
        fraction = 0.20
        trail_mult = 1.0

    start_after = tps[0] if tps else entry
    notes = [f"Runner starts after TP1 @ {round(start_after, 2)}"]
    if profile_nodes:
        structural_summary = ", ".join(sorted(profile_nodes))
        notes.append(f"Tighten trail after TP1 on acceptance above {structural_summary}")
    else:
        notes.append("Tighten trail after TP1 once structure confirms")
    notes.append("Exit runner on VWAP flip against trade with ADX roll-over")
    notes.append("If remaining ATR < 0.5×ATR near close, time-stop the runner")

    return {
        "fraction": round(fraction, 2),
        "atr_trail_mult": round(trail_mult, 2),
        "atr_trail_step": 0.4,
        "em_fraction_cap": 0.60 if style_token in {"intraday", "scalp"} else 0.75,
        "notes": notes,
        "trail": f"Trail with ATR x{trail_mult:.2f}",
    }


__all__ = ["compute_runner"]
