"""Probability decomposition and trade quality helpers."""

from __future__ import annotations

from typing import Any, Dict, Mapping


def _extract_score(inputs: Mapping[str, Any], key: str, default: float = 0.5) -> float:
    value = inputs.get(key, default)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = default
    return max(0.0, min(1.0, numeric))


def score_components(inputs: Mapping[str, Any]) -> Dict[str, float]:
    """Return deterministic component scores clamped to [0, 1]."""

    trend = _extract_score(inputs, "trend_alignment")
    liquidity = _extract_score(inputs, "liquidity_structure")
    momentum = _extract_score(inputs, "momentum_signal")
    volatility = _extract_score(inputs, "volatility_regime")
    return {
        "trend_alignment": trend,
        "liquidity_structure": liquidity,
        "momentum_signal": momentum,
        "volatility_regime": volatility,
    }


def overall_confidence(components: Mapping[str, Any]) -> float:
    """Blend component scores into a confidence value."""

    trend = _extract_score(components, "trend_alignment")
    liquidity = _extract_score(components, "liquidity_structure")
    volatility = _extract_score(components, "volatility_regime")
    base = 0.6 * trend + 0.2 * liquidity + 0.2 * volatility
    return max(0.0, min(1.0, base))


def quality_grade(confidence: float) -> str:
    """Deterministic confidence-to-quality mapping."""

    score = max(0.0, min(1.0, float(confidence)))
    if score >= 0.85:
        return "A+"
    if score >= 0.78:
        return "A"
    if score >= 0.73:
        return "A-"
    if score >= 0.68:
        return "B+"
    if score >= 0.62:
        return "B"
    if score >= 0.58:
        return "B-"
    if score >= 0.52:
        return "C"
    return "D"


__all__ = ["score_components", "overall_confidence", "quality_grade"]
