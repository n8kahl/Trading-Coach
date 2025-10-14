"""Option scoring helpers for Fancy Trader 2.0."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


SEVERITY_WEIGHTS = {
    "spread": 0.30,
    "spread_stability": 0.10,
    "delta": 0.25,
    "iv": 0.20,
    "volume_oi": 0.15,
}


@dataclass(slots=True)
class OptionScore:
    """Represents a scored option contract."""

    score: float
    components: Dict[str, float]


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _normalize_inverse(value: float, max_value: float) -> float:
    if value is None or not isinstance(value, (int, float)):
        return 0.0
    if max_value <= 0:
        return 0.0
    return _clamp(1.0 - (value / max_value), 0.0, 1.0)


def _delta_component(delta: float | None, prefer: float) -> float:
    if delta is None:
        return 0.0
    try:
        distance = abs(abs(float(delta)) - prefer)
    except (TypeError, ValueError):
        return 0.0
    return _clamp(1.0 - distance / 0.6, 0.0, 1.0)


def _iv_component(iv_percentile: float | None, target_band: Tuple[float, float]) -> float:
    if iv_percentile is None:
        return 0.5
    low, high = target_band
    value = float(iv_percentile)
    if value < low:
        return _clamp(1.0 - ((low - value) / low), 0.0, 1.0)
    if value > high:
        span = max(1.0 - high, 1e-6)
        return _clamp(1.0 - ((value - high) / span), 0.0, 1.0)
    return 1.0


def _volume_oi_component(volume: float | None, open_interest: float | None) -> float:
    try:
        volume_val = float(volume or 0.0)
        oi_val = float(open_interest or 0.0)
    except (TypeError, ValueError):
        return 0.0
    if oi_val <= 0:
        return 0.0
    ratio = volume_val / oi_val
    return _clamp(min(ratio, 1.5) / 1.5, 0.0, 1.0)


def _spread_stability_component(bid: float | None, ask: float | None, spread_pct: float | None) -> float:
    try:
        bid_val = float(bid)
        ask_val = float(ask)
    except (TypeError, ValueError):
        bid_val = None
        ask_val = None
    if bid_val is None or ask_val is None or bid_val <= 0 or ask_val <= 0:
        return 0.5 if spread_pct is None else _normalize_inverse(spread_pct, 1.0)
    midpoint = (bid_val + ask_val) / 2.0
    if midpoint <= 0:
        return 0.0
    stability = abs(ask_val - bid_val) / midpoint
    base = _normalize_inverse(stability, 0.4)
    if spread_pct is not None:
        base = (base + _normalize_inverse(spread_pct, 0.4)) / 2.0
    return _clamp(base, 0.0, 1.0)


def score_contract(
    contract: Dict[str, float | int | str],
    *,
    prefer_delta: float = 0.5,
    iv_percentile: float | None = None,
) -> OptionScore:
    """Return a composite score and component breakdown for an option contract."""

    spread_pct = contract.get("spread_pct")
    bid = contract.get("bid")
    ask = contract.get("ask")
    delta = contract.get("delta")
    volume = contract.get("volume")
    open_interest = contract.get("open_interest")
    iv_pct = iv_percentile if iv_percentile is not None else contract.get("iv_percentile")

    components = {
        "spread": _normalize_inverse(float(spread_pct or 0.0), 0.40),
        "spread_stability": _spread_stability_component(bid, ask, spread_pct),
        "delta": _delta_component(delta, prefer_delta),
        "iv": _iv_component(iv_pct, (0.25, 0.75)),
        "volume_oi": _volume_oi_component(volume, open_interest),
    }
    score = 0.0
    for key, weight in SEVERITY_WEIGHTS.items():
        score += components.get(key, 0.0) * weight

    contract["liquidity_score"] = round(score, 4)
    contract["liquidity_components"] = {k: round(v, 4) for k, v in components.items()}
    return OptionScore(score=score, components=components)


def build_example_leg(contract: Dict[str, float | int | str]) -> Dict[str, float | int | str]:
    """Return a compact example leg description for response payloads."""

    opt_type = contract.get("option_type") or contract.get("type")
    if isinstance(opt_type, str):
        opt_type = opt_type.lower()
    return {
        "symbol": contract.get("symbol"),
        "type": opt_type,
        "strike": contract.get("strike"),
        "expiry": contract.get("expiration_date"),
        "delta": contract.get("delta"),
        "bid": contract.get("bid"),
        "ask": contract.get("ask"),
        "spread_pct": contract.get("spread_pct"),
        "spread_stability": contract.get("liquidity_components", {}).get("spread_stability")
        if isinstance(contract.get("liquidity_components"), dict)
        else None,
        "oi": contract.get("open_interest") or contract.get("oi"),
        "volume": contract.get("volume"),
        "iv_percentile": contract.get("iv_percentile"),
        "composite_score": contract.get("liquidity_score"),
        "tradeability": contract.get("tradeability"),
    }
