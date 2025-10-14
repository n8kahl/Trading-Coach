"""Option scoring helpers for Fancy Trader 2.0."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from src.tradier import TradierNotConfiguredError, fetch_option_chain_cached


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


async def best_contract_example(symbol: str, style: Optional[str], as_of: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Return the top-scoring contract example for a symbol/style."""

    try:
        chain = await fetch_option_chain_cached(symbol)
    except TradierNotConfiguredError:
        return None
    except Exception:
        return None
    if chain is None or chain.empty:
        return None

    df = chain.copy()
    if "spread_pct" not in df.columns:
        bid = df.get("bid")
        ask = df.get("ask")
        if bid is not None and ask is not None:
            mid = (df["bid"] + df["ask"]) / 2.0
            df["spread_pct"] = (df["ask"] - df["bid"]) / mid.replace(0, pd.NA)
        else:
            df["spread_pct"] = pd.NA

    prefer_delta_map = {
        "scalp": 0.55,
        "intraday": 0.50,
        "swing": 0.45,
        "leap": 0.35,
    }
    style_key = (style or "intraday").lower()
    prefer_delta = prefer_delta_map.get(style_key, 0.5)

    candidates: list[Dict[str, Any]] = []
    for _, row in df.iterrows():
        try:
            contract = {
                "symbol": row.get("symbol"),
                "option_type": row.get("option_type"),
                "strike": row.get("strike"),
                "expiration_date": row.get("expiration_date"),
                "bid": row.get("bid"),
                "ask": row.get("ask"),
                "spread_pct": row.get("spread_pct"),
                "volume": row.get("volume"),
                "open_interest": row.get("open_interest"),
                "delta": row.get("delta"),
            }
        except Exception:
            continue
        score = score_contract(contract, prefer_delta=prefer_delta)
        contract["liquidity_score"] = score.score
        contract["liquidity_components"] = score.components
        contract["open_interest"] = row.get("open_interest")
        contract["volume"] = row.get("volume")
        contract["tradeability"] = int(round(score.score * 100))
        candidates.append(contract)

    if not candidates:
        return None
    candidates.sort(key=lambda item: item.get("liquidity_score", 0.0), reverse=True)
    top = candidates[0]
    return build_example_leg(top)
