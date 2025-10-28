from __future__ import annotations

import logging
import math
from datetime import datetime, timezone, timedelta
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
_LOGGER = logging.getLogger("options.select")

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
    log = _LOGGER
    log.info("[%s] selecting contracts as_of=%s", symbol, as_of_resolved.isoformat())
    market_closed = not _MARKET_CLOCK.is_rth_open(at=as_of_resolved.astimezone(_ET))
    use_asof = quote_session == "regular_close" or market_closed
    if use_asof:
        chain = await fetch_polygon_option_chain_asof(symbol, as_of_resolved)
    else:
        chain = await fetch_polygon_option_chain(symbol)
    chain_row_count = 0 if chain is None else len(chain)
    chain_columns = list(chain.columns) if isinstance(chain, pd.DataFrame) else None
    if (
        use_asof
        and isinstance(chain, pd.DataFrame)
        and not chain.empty
        and ("delta" not in chain.columns or chain["delta"].isna().mean() >= 0.8)
    ):
        prior_cutoff = as_of_resolved - timedelta(minutes=5)
        if prior_cutoff.date() == as_of_resolved.date():
            fallback_chain = await fetch_polygon_option_chain_asof(symbol, prior_cutoff)
            if isinstance(fallback_chain, pd.DataFrame) and not fallback_chain.empty:
                _LOGGER.warning(
                    "as-of option chain delta missing, using fallback window symbol=%s original_rows=%d fallback_rows=%d",
                    symbol,
                    chain_row_count,
                    len(fallback_chain),
                )
                chain = fallback_chain
                chain_row_count = len(chain)
                chain_columns = list(chain.columns)
    log.info(
        "[%s] Chain length=%d cols=%s session=%s use_asof=%s",
        symbol,
        chain_row_count,
        chain_columns,
        quote_session,
        use_asof,
    )
    if chain is None or chain.empty:
        log.warning("[%s] No option chain data at %s", symbol, as_of_resolved.isoformat())
        return {
            "options_contracts": [],
            "rejected_contracts": [],
            "options_note": "Polygon option chain unavailable",
            "options_quote_session": quote_session,
            "options_as_of": as_of_resolved.isoformat(),
        }

    normalized_chain = _normalize_chain(chain, option_type)
    log.info("filter option_type=%s: %d -> %d", option_type, chain_row_count, len(normalized_chain))
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
    initial_dte_range = (float(rules["dte_low"]), float(rules["dte_high"]))
    initial_delta_range = (float(rules["delta_low"]), float(rules["delta_high"]))
    initial_min_volume = float(rules.get("min_volume", 0.0))
    rules["min_open_interest"] = float(_OI_LADDER[0])
    initial_min_open_interest = rules["min_open_interest"]
    after_hours_relaxed = False
    if market_closed:
        rules["max_spread_pct"] = max(base_spread_cap, 400.0)
        rules["min_open_interest"] = min(rules["min_open_interest"], 50.0)
        rules["min_volume"] = 0.0
        after_hours_relaxed = True
    base_spread_cap = float(rules.get("max_spread_pct", base_spread_cap))

    diagnostics = _log_guardrail_pipeline(
        normalized_chain,
        dte_range=initial_dte_range,
        delta_range=initial_delta_range,
        max_spread_pct=base_spread_cap,
        min_open_interest=initial_min_open_interest,
        min_volume=initial_min_volume,
    )
    missing_delta_candidates = diagnostics.get("missing_delta")

    relax_flags: List[str] = []
    if after_hours_relaxed:
        relax_flags.append("AFTER_HOURS_RELAXED")
    rejection_records: List[Tuple[str, str]] = []
    filtered = normalized_chain.copy()
    selection = select_top_n(pd.DataFrame(), [], 0)  # placeholder
    relaxation_sequence = list(_RELAXATION_SEQUENCE)
    if after_hours_relaxed:
        relaxation_sequence.extend([("spread", None), ("spread", None)])

    relax_index = 0
    while True:
        filtered, stage_rejections = _run_filters(normalized_chain, rules)
        rejection_records.extend(stage_rejections)
        selection = select_top_n(filtered, desired_targets, desired_count)
        if len(selection.rows) >= desired_count:
            break
        if relax_index >= len(relaxation_sequence):
            break
        relax_type, relax_value = relaxation_sequence[relax_index]
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
            cap_limit = max(base_spread_cap, rules["max_spread_pct"]) + (10.0 if after_hours_relaxed else 2.0)
            rules["max_spread_pct"] = min(cap_limit, rules["max_spread_pct"] + (10.0 if after_hours_relaxed else 2.0))
            relax_flags.append("SPREAD_RELAXED")
        elif relax_type == "oi":
            rules["min_open_interest"] = float(relax_value or rules["min_open_interest"])
            relax_flags.append(f"OPEN_INTEREST_RELAXED_{int(rules['min_open_interest'])}")

    def _relaxed_contract_fallback(df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], Optional[str], Optional[str]]:
        if df is None or df.empty:
            return [], None, None
        fallback_frame = df.copy()
        numeric_cols = [
            "dte",
            "delta",
            "spread_pct",
            "open_interest",
            "volume",
            "bid",
            "ask",
            "mid",
        ]
        for col in numeric_cols:
            if col in fallback_frame.columns:
                fallback_frame[col] = pd.to_numeric(fallback_frame[col], errors="coerce")
        for required_col in ("dte", "delta", "spread_pct", "open_interest"):
            if required_col not in fallback_frame.columns:
                fallback_frame[required_col] = pd.Series([float("nan")] * len(fallback_frame), index=fallback_frame.index)
        if "mid" not in fallback_frame or fallback_frame["mid"].isna().all():
            if "bid" in fallback_frame.columns and "ask" in fallback_frame.columns:
                fallback_frame["mid"] = (
                    pd.to_numeric(fallback_frame["bid"], errors="coerce")
                    + pd.to_numeric(fallback_frame["ask"], errors="coerce")
                ) / 2.0
        if "spread_pct" in fallback_frame.columns:
            spreads = fallback_frame["spread_pct"]
            needs_percent = spreads <= 1.0
            mask = needs_percent.fillna(False)
            fallback_frame.loc[mask, "spread_pct"] = spreads[mask] * 100.0
        pre_count = len(fallback_frame)
        dte_ok = fallback_frame[
            fallback_frame["dte"].between(1, 45, inclusive="both")
        ].copy()
        delta_ok = dte_ok[dte_ok["delta"].abs().between(0.15, 0.55, inclusive="both")].copy()
        spread_ok = delta_ok[delta_ok["spread_pct"] <= 15.0].copy()
        oi_ok = spread_ok[spread_ok["open_interest"] >= 50.0].copy()
        log.info(
            "[%s] Relaxed filters counts pre=%d dte=%d delta=%d spread=%d oi=%d",
            symbol,
            pre_count,
            len(dte_ok),
            len(delta_ok),
            len(spread_ok),
            len(oi_ok),
        )
        if oi_ok.empty:
            missing_delta = dte_ok[dte_ok["delta"].isna()].copy()
            if not missing_delta.empty:
                relaxed = missing_delta.sort_values(by=["spread_pct", "open_interest"], ascending=[True, False]).head(3)
                contracts: List[Dict[str, Any]] = []
                for _, row in relaxed.iterrows():
                    contract = _serialize_contract(row, quote_session, as_of_resolved.isoformat(), None)
                    flags = list(dict.fromkeys((contract.get("guardrail_flags") or []) + ["DELTA_MISSING_FALLBACK"]))
                    contract["guardrail_flags"] = flags
                    contract["status"] = "degraded"
                    contract["reason"] = "delta_missing_fallback"
                    contract["rating"] = contract.get("rating") or "yellow"
                    contracts.append(contract)
                return contracts, "Delta missing at close — relaxed filter fallback", "DELTA_MISSING_FALLBACK"
            log.error("[%s] No eligible contracts after relaxed filters", symbol)
            return [], "No contracts met filters", None
        selected = oi_ok.sort_values(by=["spread_pct", "open_interest"], ascending=[True, False]).head(3)
        contracts = []
        for _, row in selected.iterrows():
            contract = _serialize_contract(row, quote_session, as_of_resolved.isoformat(), None)
            flags = list(dict.fromkeys((contract.get("guardrail_flags") or []) + ["RELAXED_SIMPLE_FILTERS"]))
            contract["guardrail_flags"] = flags
            contract.setdefault("status", "relaxed")
            contract.setdefault("reason", "relaxed_filter_fallback")
            if not contract.get("rating"):
                contract["rating"] = "yellow"
            contracts.append(contract)
        return contracts, "Filters relaxed — using liquidity fallback", "RELAXED_SIMPLE_FILTERS"

    fallback_used = False
    delta_missing_fallback_used = False
    if (
        len(selection.rows) < desired_count
        and missing_delta_candidates is not None
        and not missing_delta_candidates.empty
    ):
        df_fallback = missing_delta_candidates.copy()
        if desired_targets:
            fallback_deltas = pd.Series(
                [float(desired_targets[min(idx, len(desired_targets) - 1)]) for idx in range(len(df_fallback))],
                index=df_fallback.index,
            )
            df_fallback["delta"] = df_fallback["delta"].fillna(fallback_deltas)
        else:
            df_fallback["delta"] = df_fallback["delta"].fillna(0.5)
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
        delta_missing_fallback_used = True
        log.warning(
            "delta-missing fallback used symbol=%s as_of=%s count=%d",
            symbol,
            as_of_resolved.isoformat(),
            len(selection.rows),
        )
    if not delta_missing_fallback_used and len(selection.rows) < desired_count and not normalized_chain.empty:
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
        if delta_missing_fallback_used:
            contract.setdefault("status", "degraded")
            contract.setdefault("reason", "delta_missing_fallback")
        options_contracts.append(contract)

    note = _build_options_note(relax_flags, rejected_contracts, fallback_used)

    target_min = 3
    if len(options_contracts) < target_min and not normalized_chain.empty:
        fallback_contracts, fallback_note, fallback_flag = _relaxed_contract_fallback(normalized_chain)
        if fallback_contracts:
            log.warning(
                "[%s] Relaxed fallback supplying %d contracts (original=%d)",
                symbol,
                len(fallback_contracts),
                len(options_contracts),
            )
            options_contracts = fallback_contracts
            if fallback_flag and fallback_flag not in relax_flags:
                relax_flags.append(fallback_flag)
            if fallback_note:
                note = fallback_note
        elif not options_contracts and fallback_note and not note:
            note = fallback_note

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


def _log_guardrail_pipeline(
    frame: pd.DataFrame,
    *,
    dte_range: Tuple[float, float],
    delta_range: Tuple[float, float],
    max_spread_pct: float,
    min_open_interest: float,
    min_volume: float,
) -> Dict[str, pd.DataFrame]:
    if frame is None or frame.empty:
        _LOGGER.info("filter pipeline skipped: frame empty")
        return {"eligible": pd.DataFrame(), "missing_delta": pd.DataFrame()}

    def _ensure_numeric(df: pd.DataFrame, column: str) -> None:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        else:
            df[column] = pd.Series([float("nan")] * len(df), index=df.index)

    diagnostic = frame.copy()

    _ensure_numeric(diagnostic, "dte")
    _ensure_numeric(diagnostic, "delta")
    _ensure_numeric(diagnostic, "spread_pct")
    _ensure_numeric(diagnostic, "open_interest")
    _ensure_numeric(diagnostic, "volume")

    dte_low, dte_high = dte_range
    delta_low, delta_high = delta_range

    dte_ok = diagnostic[
        (diagnostic["dte"] >= float(dte_low))
        & (diagnostic["dte"] <= float(dte_high))
    ].copy()
    _LOGGER.info("filter DTE: %d -> %d", len(diagnostic), len(dte_ok))

    missing_delta = dte_ok[dte_ok["delta"].isna()].copy()
    if not missing_delta.empty:
        missing_delta = missing_delta[
            (missing_delta["spread_pct"] <= float(max_spread_pct))
            & (missing_delta["open_interest"] >= float(min_open_interest))
            & (missing_delta["volume"] >= float(min_volume))
        ].copy()
    delta_ok = dte_ok[
        dte_ok["delta"].abs().between(float(delta_low), float(delta_high), inclusive="both")
    ].copy()
    _LOGGER.info(
        "filter DELTA: %d -> %d (delta nulls=%d)",
        len(dte_ok),
        len(delta_ok),
        int(len(missing_delta)),
    )

    spread_ok = delta_ok[delta_ok["spread_pct"] <= float(max_spread_pct)].copy()
    _LOGGER.info("filter SPREAD: %d -> %d", len(delta_ok), len(spread_ok))

    oi_ok = spread_ok[spread_ok["open_interest"] >= float(min_open_interest)].copy()
    vol_ok = oi_ok[oi_ok["volume"] >= float(min_volume)].copy()
    _LOGGER.info("filter OI/VOL: %d -> %d", len(oi_ok), len(vol_ok))

    return {"eligible": vol_ok, "missing_delta": missing_delta}


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
