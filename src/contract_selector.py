"""Option contract selection utilities."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Sequence

import pandas as pd


_DEFAULT_SPREAD_CAP = {
    "scalp": 8.0,
    "intraday": 8.0,
    "swing": 10.0,
    "leaps": 12.0,
}

_DEFAULT_TARGETS = {
    "intraday": (0.50, 0.40, 0.30),
    "scalp": (0.55, 0.45, 0.35),
    "swing": (0.35, 0.25),
    "leaps": (0.25, 0.15),
}

_REVERSION_KEYWORDS = ("reversion", "fader", "trim", "limited", "mean", "fade")

_STYLE_FILTER_RULES = {
    "intraday": {"delta_low": 0.30, "delta_high": 0.60, "dte_low": 0.0, "dte_high": 14.0, "max_spread_pct": 8.0, "min_volume": 10},
    "scalp": {"delta_low": 0.40, "delta_high": 0.70, "dte_low": 0.0, "dte_high": 7.0, "max_spread_pct": 8.0, "min_volume": 10},
    "swing": {"delta_low": 0.20, "delta_high": 0.45, "dte_low": 5.0, "dte_high": 45.0, "max_spread_pct": 10.0, "min_volume": 5},
    "leaps": {"delta_low": 0.15, "delta_high": 0.30, "dte_low": 40.0, "dte_high": 240.0, "max_spread_pct": 12.0, "min_volume": 1},
}


@dataclass(frozen=True)
class SelectionResult:
    rows: List[pd.Series]
    targets: List[float | None]


def filter_chain(chain: pd.DataFrame, rules: Dict[str, object]) -> pd.DataFrame:
    """Filter an option chain based on strategy rules."""
    df = chain.copy()
    numeric_cols = ["delta", "dte", "spread_pct", "open_interest", "volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    dte_low, dte_high = rules.get("dte_range", (0, 365))
    delta_low, delta_high = rules.get("delta_range", (0.0, 1.0))
    max_spread = rules.get("max_spread_pct", 1.0)
    min_oi = rules.get("min_open_interest", 0)
    min_vol = rules.get("min_volume", 0)
    df = df[
        (df["dte"] >= dte_low)
        & (df["dte"] <= dte_high)
        & (df["delta"].abs() >= delta_low)
        & (df["delta"].abs() <= delta_high)
        & (df["spread_pct"] <= max_spread)
        & (df["open_interest"] >= min_oi)
        & (df["volume"] >= min_vol)
    ]
    return df


def pick_best_contract(filtered_chain: pd.DataFrame, prefer_delta: float | None = None) -> pd.Series | None:
    """Select the best option contract from a filtered chain."""
    if filtered_chain.empty:
        return None
    df = filtered_chain.copy().reset_index(drop=True)
    if "delta" in df.columns:
        df["delta"] = pd.to_numeric(df["delta"], errors="coerce")
    if "spread_pct" in df.columns:
        df["spread_pct"] = pd.to_numeric(df["spread_pct"], errors="coerce")
    if prefer_delta is not None and "delta" in df.columns:
        df["delta_score"] = (df["delta"].abs() - abs(prefer_delta)).abs().fillna(999.0)
        df = df.sort_values(by=["delta_score", "spread_pct", "dte"])
    else:
        df = df.sort_values(by=["spread_pct", "dte", "volume"], ascending=[True, True, False])
    return df.iloc[0]


def _normalize_tradeability(value: object, *, fallback: float) -> float:
    """Return a 0–100 tradeability score from known inputs."""
    if value is None:
        return fallback
    try:
        num = float(value)
    except (TypeError, ValueError):
        return fallback
    if math.isnan(num) or math.isinf(num):
        return fallback
    if num <= 1.0:
        return max(0.0, min(100.0, num * 100.0))
    return max(0.0, min(100.0, num))


def _fallback_tradeability(spread_pct: float | None, oi: float | None, volume: float | None) -> float:
    """Blend spread tightness and depth into a coarse 0–100 score."""
    spread_component = 70.0
    if spread_pct is not None:
        spread_component = max(0.0, 95.0 - float(spread_pct) * 4.0)
    depth_component = 30.0
    oi_val = max(0.0, float(oi)) if oi is not None else 0.0
    vol_val = max(0.0, float(volume)) if volume is not None else 0.0
    depth_component = min(30.0, math.log1p(max(oi_val, vol_val)) * 5.0)
    return max(0.0, min(100.0, spread_component + depth_component))


def target_delta_by_style(style: str | None, strategy_id: str | None = None) -> Sequence[float]:
    """Return ordered delta targets (base/momentum/cheaper) for the style."""
    style_token = (style or "").strip().lower()
    key = "intraday"
    if "scalp" in style_token:
        key = "scalp"
    elif "swing" in style_token:
        key = "swing"
    elif "leap" in style_token:
        key = "leaps"
    targets = _DEFAULT_TARGETS.get(key, _DEFAULT_TARGETS["intraday"])
    if strategy_id:
        sid = strategy_id.lower()
        if any(token in sid for token in _REVERSION_KEYWORDS):
            return targets[:2]
    return targets


def style_guardrail_rules(style: str | None) -> Dict[str, float]:
    """Return default guardrail bounds for the given style."""
    style_token = (style or "").strip().lower()
    if "scalp" in style_token:
        key = "scalp"
    elif "swing" in style_token:
        key = "swing"
    elif "leap" in style_token:
        key = "leaps"
    else:
        key = "intraday"
    rules = _STYLE_FILTER_RULES.get(key, _STYLE_FILTER_RULES["intraday"])
    return {
        "delta_low": float(rules["delta_low"]),
        "delta_high": float(rules["delta_high"]),
        "dte_low": float(rules["dte_low"]),
        "dte_high": float(rules["dte_high"]),
        "max_spread_pct": float(rules["max_spread_pct"]),
        "min_volume": float(rules["min_volume"]),
        "style_key": key,
    }


def select_top_n(filtered_chain: pd.DataFrame, target_deltas: Sequence[float], count: int) -> SelectionResult:
    """Return the top `count` contracts ordered deterministically."""
    if filtered_chain is None or filtered_chain.empty or count <= 0:
        return SelectionResult(rows=[], targets=[])

    df = filtered_chain.copy().reset_index(drop=True)
    numeric_cols = ("delta", "dte", "spread_pct", "open_interest", "volume", "bid", "ask", "mid")
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "tradeability" in df.columns:
        tradeability_source = df["tradeability"]
    elif "liquidity_score" in df.columns:
        tradeability_source = df["liquidity_score"]
    else:
        tradeability_source = None

    normalized_tradeability: List[float] = []
    for idx, row in df.iterrows():
        base = tradeability_source.iloc[idx] if tradeability_source is not None else None
        spread_pct = row.get("spread_pct")
        if spread_pct is not None and spread_pct <= 1.0:
            spread_pct = float(spread_pct) * 100.0
            df.at[idx, "spread_pct"] = spread_pct
        oi_val = row.get("open_interest")
        volume = row.get("volume")
        fallback_score = _fallback_tradeability(spread_pct, oi_val, volume)
        normalized_tradeability.append(_normalize_tradeability(base, fallback=fallback_score))

    df["tradeability_score"] = normalized_tradeability
    df["abs_delta"] = df["delta"].abs()
    df["spread_pct"] = df["spread_pct"].fillna(999.0)
    df["open_interest"] = df["open_interest"].fillna(0)
    df["dte"] = df["dte"].fillna(0)

    ordered = df.sort_values(
        by=["tradeability_score", "spread_pct", "open_interest", "dte"],
        ascending=[False, True, False, True],
        kind="mergesort",
    )

    picks: List[pd.Series] = []
    used_indexes: set[int] = set()
    slot_targets: List[float | None] = []
    targets = list(target_deltas) if target_deltas else [ordered["abs_delta"].median()]

    for slot, target in enumerate(targets, start=1):
        if len(picks) >= count:
            break
        remaining = ordered.loc[~ordered.index.isin(used_indexes)].copy()
        if remaining.empty:
            break
        remaining["delta_fit"] = (remaining["abs_delta"] - float(target)).abs()
        selected_idx = remaining["delta_fit"].idxmin()
        selected = ordered.loc[selected_idx]
        picks.append(selected)
        used_indexes.add(selected_idx)
        slot_targets.append(float(target))

    if len(picks) < count:
        for row_idx, row in ordered.iterrows():
            if len(picks) >= count:
                break
            if row_idx in used_indexes:
                continue
            picks.append(row)
            used_indexes.add(row_idx)
            slot_targets.append(None)

    return SelectionResult(rows=picks[:count], targets=slot_targets[:count])


def grade_option_pick(
    tradeability: float | None,
    spread_pct: float | None,
    oi: float | None,
    delta_fit: float | None,
) -> str:
    """Return qualitative rating (green/yellow/red) for a contract."""
    tradeability_score = _normalize_tradeability(tradeability, fallback=0.0)
    spread_val = None
    if spread_pct is not None:
        spread_val = float(spread_pct)
    oi_val = None
    if oi is not None:
        try:
            oi_val = float(oi)
        except (TypeError, ValueError):
            oi_val = None
    delta_gap = float(delta_fit) if delta_fit is not None else None

    if (
        tradeability_score >= 80.0
        and (spread_val is None or spread_val <= 8.0)
        and (oi_val is None or oi_val >= 1000.0)
        and (delta_gap is None or delta_gap <= 0.08)
    ):
        return "green"

    if (
        tradeability_score >= 60.0
        and (spread_val is None or spread_val <= 12.0)
        and (oi_val is None or oi_val >= 300.0)
    ):
        return "yellow"

    return "red"


def reason_tokens(
    tradeability: float | None,
    spread_pct: float | None,
    oi: float | None,
    delta_fit: float | None,
) -> List[str]:
    tokens: List[str] = []
    if tradeability is not None:
        if tradeability >= 80.0:
            tokens.append("tradeability_strong")
        elif tradeability >= 60.0:
            tokens.append("tradeability_ok")
        else:
            tokens.append("tradeability_low")
    if spread_pct is not None:
        tokens.append("spread_ok" if spread_pct <= 12.0 else "spread_wide")
    if oi is not None:
        tokens.append("oi_ok" if oi >= 300.0 else "oi_low")
    if delta_fit is not None:
        tokens.append("delta_fit_ok" if delta_fit <= 0.1 else "delta_fit_wide")
    return tokens


__all__ = [
    "filter_chain",
    "pick_best_contract",
    "select_top_n",
    "target_delta_by_style",
    "grade_option_pick",
    "style_guardrail_rules",
    "reason_tokens",
]
