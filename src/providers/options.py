from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import pandas as pd

from ..contract_selector import grade_option_pick, select_top_n, target_delta_by_style, style_guardrail_rules, reason_tokens
from ..market_clock import MarketClock
from ..polygon_options import fetch_polygon_option_chain, fetch_polygon_option_chain_asof
from ..services.chart_utils import infer_session_label


_ET = ZoneInfo("America/New_York")
_MARKET_CLOCK = MarketClock()
_OI_LADDER = [1000, 700, 500, 300]

_RELAXATION_SEQUENCE: Sequence[Tuple[str, Optional[float]]] = (
    ("delta", None),
    ("dte", None),
    ("spread", None),
    ("oi", 700.0),
    ("oi", 500.0),
    ("oi", 300.0),
)


def _normalize_style(style: Any) -> str:
    token = str(style or "").strip().lower()
    if "scalp" in token:
        return "scalp"
    if "swing" in token:
        return "swing"
    if "leap" in token:
        return "leaps"
    return "intraday"


def _normalize_direction(plan: Mapping[str, Any]) -> str:
    direction = str(plan.get("direction") or plan.get("bias") or "long").lower()
    return "short" if direction == "short" else "long"


def _resolve_quote_context(as_of: datetime | None) -> Tuple[datetime, str]:
    if as_of is None:
        now = datetime.now(timezone.utc)
    else:
        if as_of.tzinfo is None:
            now = as_of.replace(tzinfo=timezone.utc)
        else:
            now = as_of.astimezone(timezone.utc)
    session = infer_session_label(now)
    if session == "live":
        return now, "regular_open"
    et_time = now.astimezone(_ET)
    prev_close_et = _MARKET_CLOCK.last_rth_close(at=et_time)
    prev_close_utc = prev_close_et.astimezone(timezone.utc)
    return prev_close_utc, "regular_close"


def _normalize_chain(chain: pd.DataFrame, option_type: str) -> pd.DataFrame:
    if chain is None or chain.empty:
        return pd.DataFrame()
    df = chain.copy()
    df["option_type"] = df.get("option_type", "").astype(str).str.lower()
    df = df[df["option_type"] == option_type]
    if df.empty:
        return df
    numeric_cols = [
        "strike",
        "delta",
        "dte",
        "spread_pct",
        "open_interest",
        "volume",
        "bid",
        "ask",
        "mid",
        "tradeability",
        "liquidity_score",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "mid" not in df or df["mid"].isna().all():
        if "bid" in df.columns and "ask" in df.columns:
            df["mid"] = (pd.to_numeric(df["bid"], errors="coerce") + pd.to_numeric(df["ask"], errors="coerce")) / 2.0
    if "spread_pct" in df.columns:
        spreads = pd.to_numeric(df["spread_pct"], errors="coerce")
        needs_percent = spreads <= 1.0
        df.loc[needs_percent, "spread_pct"] = spreads[needs_percent] * 100.0
    return df


def _first_failure(row: pd.Series, rules: Dict[str, float]) -> Tuple[Optional[str], Optional[str]]:
    symbol = str(row.get("symbol") or "")
    delta_val = row.get("delta")
    if pd.isna(delta_val):
        return "DELTA_MISSING", symbol
    try:
        abs_delta = abs(float(delta_val))
    except (TypeError, ValueError):
        return "DELTA_MISSING", symbol
    if abs_delta < rules["delta_low"]:
        return "DELTA_TOO_LOW", symbol
    if abs_delta > rules["delta_high"]:
        return "DELTA_TOO_HIGH", symbol
    dte_val = row.get("dte")
    if pd.isna(dte_val):
        return "DTE_MISSING", symbol
    try:
        dte_val = float(dte_val)
    except (TypeError, ValueError):
        return "DTE_MISSING", symbol
    if dte_val < rules["dte_low"]:
        return "DTE_TOO_SHORT", symbol
    if dte_val > rules["dte_high"]:
        return "DTE_TOO_LONG", symbol
    spread = row.get("spread_pct")
    if pd.isna(spread):
        return "SPREAD_MISSING", symbol
    if float(spread) > rules["max_spread_pct"]:
        return "SPREAD_TOO_WIDE", symbol
    oi_val = row.get("open_interest")
    if pd.isna(oi_val):
        return "OPEN_INTEREST_MISSING", symbol
    try:
        if float(oi_val) < rules["min_open_interest"]:
            return "OPEN_INTEREST_TOO_LOW", symbol
    except (TypeError, ValueError):
        return "OPEN_INTEREST_MISSING", symbol
    volume = row.get("volume")
    try:
        if float(volume) < rules["min_volume"]:
            return "VOLUME_TOO_LOW", symbol
    except (TypeError, ValueError):
        return "VOLUME_MISSING", symbol
    return None, None


def _dedupe_rejections(rejections: Sequence[Tuple[str, str]]) -> List[Dict[str, str]]:
    dedup: Dict[Tuple[str, str], Dict[str, str]] = {}
    for reason, symbol in rejections:
        if not reason:
            continue
        sym = (symbol or "").upper()
        key = (sym, reason)
        if key not in dedup:
            dedup[key] = {"symbol": sym or "UNKNOWN", "reason": reason}
    return list(dedup.values())


def _build_options_note(relax_flags: List[str], rejected: List[Dict[str, str]], fallback_used: bool) -> Optional[str]:
    if fallback_used and rejected:
        reasons = ", ".join(sorted({item["reason"] for item in rejected}))
        return f"Contracts unavailable ({reasons}); review guardrail_flags."
    if relax_flags:
        labels = ", ".join(relax_flags)
        return f"Contracts relaxed ({labels}); review guardrail_flags."
    if rejected:
        reasons = ", ".join(sorted({item["reason"] for item in rejected}))
        return f"Contracts filtered ({reasons})"
    return None


async def select_contracts(symbol: str, as_of: datetime, plan: Mapping[str, Any]) -> Dict[str, Any]:
    """Select option contracts for a plan using Polygon snapshot data."""

    direction = _normalize_direction(plan)
    option_type = "call" if direction == "long" else "put"
    style = _normalize_style(plan.get("style"))
    strategy_id = plan.get("strategy_id")
    desired_targets = target_delta_by_style(style, strategy_id)
    desired_count = max(2, min(3, len(desired_targets)))

    as_of_resolved, quote_session = _resolve_quote_context(as_of)
    market_closed = not _MARKET_CLOCK.is_rth_open(at=as_of_resolved.astimezone(_ET))
    use_asof = quote_session == "regular_close" or market_closed
    if use_asof:
        chain = await fetch_polygon_option_chain_asof(symbol, as_of_resolved)
    else:
        chain = await fetch_polygon_option_chain(symbol)
    if chain is None or chain.empty:
        return {
            "options_contracts": [],
            "rejected_contracts": [],
            "options_note": "Polygon option chain unavailable",
            "options_quote_session": quote_session,
            "options_as_of": as_of_resolved.isoformat(),
        }

    normalized_chain = _normalize_chain(chain, option_type)
    if normalized_chain.empty:
        return {
            "options_contracts": [],
            "rejected_contracts": [{"symbol": symbol.upper(), "reason": "OPTION_SIDE_UNAVAILABLE"}],
            "options_note": f"No {option_type} contracts available",
            "options_quote_session": quote_session,
            "options_as_of": as_of_resolved.isoformat(),
        }

    rules = style_guardrail_rules(style)
    rules.pop("style_key", None)
    base_spread_cap = float(rules.get("max_spread_pct", 8.0))
    rules["min_open_interest"] = float(_OI_LADDER[0])

    relax_flags: List[str] = []
    rejection_records: List[Tuple[str, str]] = []
    filtered = normalized_chain.copy()
    selection = select_top_n(pd.DataFrame(), [], 0)  # placeholder

    relax_index = 0
    while True:
        filtered, stage_rejections = _run_filters(normalized_chain, rules)
        rejection_records.extend(stage_rejections)
        selection = select_top_n(filtered, desired_targets, desired_count)
        if len(selection.rows) >= desired_count:
            break
        if relax_index >= len(_RELAXATION_SEQUENCE):
            break
        relax_type, relax_value = _RELAXATION_SEQUENCE[relax_index]
        relax_index += 1
        if relax_type == "delta":
            rules["delta_low"] = max(0.0, rules["delta_low"] - 0.05)
            rules["delta_high"] = min(1.0, rules["delta_high"] + 0.05)
            relax_flags.append("DELTA_WINDOW_RELAXED")
        elif relax_type == "dte":
            rules["dte_low"] = max(0.0, rules["dte_low"] - 2.0)
            rules["dte_high"] = rules["dte_high"] + 2.0
            relax_flags.append("DTE_RELAXED")
        elif relax_type == "spread":
            cap_limit = base_spread_cap + 4.0
            rules["max_spread_pct"] = min(cap_limit, rules["max_spread_pct"] + 2.0)
            relax_flags.append("SPREAD_RELAXED")
        elif relax_type == "oi":
            rules["min_open_interest"] = float(relax_value or rules["min_open_interest"])
            relax_flags.append(f"OPEN_INTEREST_RELAXED_{int(rules['min_open_interest'])}")

    fallback_used = False
    if len(selection.rows) < desired_count and not normalized_chain.empty:
        df_fallback = normalized_chain.copy()
        df_fallback["delta"] = df_fallback["delta"].fillna(0.0)
        existing_flags = df_fallback.get("guardrail_flags")
        if isinstance(existing_flags, pd.Series):
            df_fallback["guardrail_flags"] = existing_flags.apply(
                lambda flags: list(flags) + ["DELTA_MISSING"] if isinstance(flags, (list, tuple, set)) else ["DELTA_MISSING"]
            )
        else:
            df_fallback["guardrail_flags"] = [["DELTA_MISSING"] for _ in range(len(df_fallback))]
        selection = select_top_n(df_fallback, desired_targets, desired_count)
        relax_flags.append("DELTA_MISSING_FALLBACK")
        fallback_used = True
    if len(selection.rows) < desired_count and not normalized_chain.empty:
        selection = select_top_n(normalized_chain, desired_targets, desired_count)
        fallback_used = True
        relax_flags.append("GUARDRAIL_FALLBACK")

    rejected_contracts = _dedupe_rejections(rejection_records)
    as_of_iso = as_of_resolved.isoformat()
    options_contracts: List[Dict[str, Any]] = []
    for idx, row in enumerate(selection.rows):
        contract = _serialize_contract(
            row,
            quote_session,
            as_of_iso,
            selection.targets[idx] if idx < len(selection.targets) else None,
        )
        if relax_flags:
            existing_contract_flags = contract.get("guardrail_flags", [])
            merged_flags = list(dict.fromkeys(list(existing_contract_flags) + list(relax_flags)))
            contract["guardrail_flags"] = merged_flags
        options_contracts.append(contract)

    note = _build_options_note(relax_flags, rejected_contracts, fallback_used)

    return {
        "options_contracts": options_contracts,
        "rejected_contracts": rejected_contracts,
        "options_note": note,
        "options_quote_session": quote_session,
        "options_as_of": as_of_iso,
    }


def _run_filters(chain: pd.DataFrame, rules: Dict[str, float]) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
    keep_indices: List[int] = []
    rejections: List[Tuple[str, str]] = []
    for idx, row in chain.iterrows():
        reason, symbol = _first_failure(row, rules)
        if reason:
            rejections.append((reason, symbol or row.get("symbol") or ""))
            continue
        keep_indices.append(idx)
    filtered = chain.loc[keep_indices].copy() if keep_indices else chain.iloc[0:0].copy()
    return filtered, rejections


def _serialize_contract(row: pd.Series, quote_session: str, as_of_timestamp: str, target_delta: Optional[float]) -> Dict[str, Any]:
    delta_val = row.get("delta")
    abs_delta = abs(float(delta_val)) if delta_val is not None else None
    tradeability = float(row.get("tradeability_score")) if row.get("tradeability_score") is not None else None
    spread_pct = float(row.get("spread_pct")) if row.get("spread_pct") is not None else None
    oi = float(row.get("open_interest")) if row.get("open_interest") is not None else None
    delta_fit = None
    if abs_delta is not None and target_delta is not None:
        delta_fit = abs(abs_delta - float(target_delta))
    rating = grade_option_pick(tradeability, spread_pct, oi, delta_fit)
    reasons = reason_tokens(tradeability, spread_pct, oi, delta_fit)

    contract: Dict[str, Any] = {
        "symbol": row.get("symbol"),
        "option_type": row.get("option_type"),
        "strike": _safe_number(row.get("strike")),
        "expiration_date": row.get("expiration_date"),
        "delta": _safe_number(row.get("delta")),
        "dte": _safe_number(row.get("dte")),
        "bid": _safe_number(row.get("bid")),
        "ask": _safe_number(row.get("ask")),
        "mid": _safe_number(row.get("mid")),
        "spread_pct": _safe_number(row.get("spread_pct")),
        "open_interest": _safe_number(row.get("open_interest")),
        "volume": _safe_number(row.get("volume")),
        "tradeability_score": _safe_number(row.get("tradeability_score")),
        "quote_session": quote_session,
        "as_of_timestamp": as_of_timestamp,
        "rating": rating,
        "reasons": reasons,
    }
    row_flags = row.get("guardrail_flags")
    if isinstance(row_flags, (list, tuple, set)):
        contract["guardrail_flags"] = list(row_flags)
        if "DELTA_MISSING" in contract["guardrail_flags"]:
            contract["rating"] = "yellow"
    if tradeability is not None:
        contract["tradeability"] = tradeability
    if delta_fit is not None:
        contract["delta_fit"] = round(float(delta_fit), 4)
    return {key: value for key, value in contract.items() if value is not None}


def _safe_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(num) or math.isinf(num):
        return None
    return num


__all__ = ["select_contracts"]
