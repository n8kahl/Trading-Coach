"""Polygon option chain helpers.

This module centralizes the logic for fetching Polygon snapshot option chains
and shaping them into pandas DataFrames that downstream components can filter
and rank.  It also provides convenience functions for producing GPT-friendly
summaries (best contract plus a few alternatives) driven by strategy rules.
"""

from __future__ import annotations

from datetime import datetime
import logging
import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx
import numpy as np
import pandas as pd

from .config import get_settings
from .contract_selector import filter_chain
from .data_sources import fetch_polygon_ohlcv

logger = logging.getLogger(__name__)

POLYGON_BASE_URL = "https://api.polygon.io"


def _safe_number(value: Any) -> Optional[float]:
    """Convert numeric-ish values to float, guarding against NaNs/Infs."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        val = float(value)
    elif isinstance(value, (np.integer, np.floating)):
        val = float(value)
    elif isinstance(value, str):
        try:
            val = float(value)
        except ValueError:
            return None
    else:
        return None
    if math.isnan(val) or math.isinf(val):
        return None
    return val


def _safe_int(value: Any) -> Optional[int]:
    num = _safe_number(value)
    if num is None:
        return None
    return int(round(num))


def _normalize_contract_type(contract_type: Optional[str]) -> Optional[str]:
    if not contract_type:
        return None
    token = contract_type.strip().lower()
    if token in {"call", "c"}:
        return "call"
    if token in {"put", "p"}:
        return "put"
    return token or None


async def fetch_polygon_option_chain(symbol: str, expiration: str | None = None, *, limit: int = 400) -> pd.DataFrame:
    """Fetch Polygon's snapshot option chain for the given underlying ticker."""
    settings = get_settings()
    api_key = settings.polygon_api_key
    if not api_key:
        return pd.DataFrame()

    params: Dict[str, Any] = {
        "limit": limit,
        "order": "asc",
        "sort": "expiration_date",
        "include_greeks": "true",
        "include_volatility": "true",
        "apiKey": api_key,
    }
    if expiration:
        params["expiration_date"] = expiration

    url = f"{POLYGON_BASE_URL}/v3/snapshot/options/{symbol.upper()}"
    timeout = httpx.Timeout(8.0, connect=4.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("Polygon option snapshot failed for %s: %s", symbol, exc)
            return pd.DataFrame()

    payload = resp.json()
    results = payload.get("results") or []
    if not results:
        return pd.DataFrame()
    return _normalize_option_results(results, fallback_symbol=symbol.upper())


def _normalize_as_of(as_of: datetime | str | pd.Timestamp | None) -> pd.Timestamp | None:
    if not as_of:
        return None
    if isinstance(as_of, pd.Timestamp):
        ts = as_of
    elif isinstance(as_of, datetime):
        ts = pd.Timestamp(as_of)
    elif isinstance(as_of, str):
        try:
            ts = pd.Timestamp(as_of)
        except Exception:
            return None
    else:
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


async def fetch_polygon_option_chain_asof(
    symbol: str,
    as_of: datetime | str | pd.Timestamp | None,
    *,
    expiration: str | None = None,
    limit: int = 400,
) -> pd.DataFrame:
    """Return Polygon option chain data clamped to the provided `as_of` timestamp."""

    frame = await fetch_polygon_option_chain(symbol, expiration=expiration, limit=limit)
    cutoff = _normalize_as_of(as_of)
    if frame.empty or cutoff is None:
        return frame

    filtered = frame.copy()
    timestamp_columns = []
    if "last_updated" in filtered.columns:
        timestamp_columns.append("last_updated")
    if "underlying_last_updated" in filtered.columns:
        timestamp_columns.append("underlying_last_updated")

    for column in timestamp_columns:
        try:
            timestamps = pd.to_datetime(filtered[column], utc=True, errors="coerce")
        except Exception:
            continue
        mask = timestamps.isna() | (timestamps <= cutoff)
        filtered = filtered.loc[mask]

    return filtered


async def last_price_asof(
    symbol: str,
    as_of: datetime | str | pd.Timestamp | None,
    *,
    timeframe: str = "1",
) -> Optional[float]:
    """Return the latest Polygon close at or before `as_of` for the requested symbol."""

    frame = await fetch_polygon_ohlcv(symbol, timeframe)
    if frame is None or frame.empty:
        return None

    cutoff = _normalize_as_of(as_of)
    if cutoff is not None:
        frame = frame.loc[frame.index <= cutoff]
        if frame.empty:
            return None

    try:
        return float(frame["close"].iloc[-1])
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def _normalize_option_results(results: Iterable[Dict[str, Any]], *, fallback_symbol: str) -> pd.DataFrame:
    now_date = pd.Timestamp.utcnow().date()
    records: List[Dict[str, Any]] = []

    for item in results:
        details = item.get("details") or {}
        last_quote = item.get("last_quote") or {}
        greeks = item.get("greeks") or {}
        day = item.get("day") or {}
        underlying = item.get("underlying_asset") or {}

        option_symbol = details.get("ticker") or details.get("symbol")
        expiration_date = details.get("expiration_date")
        strike_price = _safe_number(details.get("strike_price"))
        contract_type = _normalize_contract_type(details.get("contract_type"))

        bid = _safe_number(last_quote.get("bid") or last_quote.get("bid_price"))
        ask = _safe_number(last_quote.get("ask") or last_quote.get("ask_price"))
        bid_size = _safe_int(last_quote.get("bid_size"))
        ask_size = _safe_int(last_quote.get("ask_size"))

        mid = None
        spread_pct = None
        if bid is not None and ask is not None and ask >= bid:
            mid = (bid + ask) / 2.0
            spread = ask - bid
            if mid > 0:
                spread_pct = spread / mid

        delta = _safe_number(greeks.get("delta"))
        gamma = _safe_number(greeks.get("gamma"))
        theta = _safe_number(greeks.get("theta"))
        vega = _safe_number(greeks.get("vega"))
        rho = _safe_number(greeks.get("rho"))

        open_interest = _safe_int(item.get("open_interest") or day.get("open_interest"))
        volume = _safe_int(day.get("volume"))
        iv = _safe_number(item.get("implied_volatility") or item.get("iv"))

        dte = None
        if expiration_date:
            try:
                exp_ts = pd.Timestamp(expiration_date)
                exp_date = exp_ts.date()
                dte = max((exp_date - now_date).days, 0)
            except Exception:  # pragma: no cover - defensive
                dte = None

        last_quote_time = last_quote.get("last_updated") or item.get("updated")
        underlying_price = _safe_number(underlying.get("price"))
        underlying_symbol = underlying.get("ticker") or fallback_symbol
        underlying_updated = underlying.get("last_updated")

        break_even = None
        if mid is not None and strike_price is not None:
            if contract_type == "call":
                break_even = strike_price + mid
            elif contract_type == "put":
                break_even = strike_price - mid

        records.append(
            {
                "symbol": option_symbol,
                "expiration_date": expiration_date,
                "strike": strike_price,
                "option_type": contract_type,
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "spread_pct": spread_pct,
                "bid_size": bid_size,
                "ask_size": ask_size,
                "delta": delta,
                "gamma": gamma,
                "theta": theta,
                "vega": vega,
                "rho": rho,
                "open_interest": open_interest,
                "volume": volume,
                "implied_volatility": iv,
                "dte": float(dte) if dte is not None else np.nan,
                "last_updated": last_quote_time,
                "underlying_symbol": underlying_symbol,
                "underlying_price": underlying_price,
                "underlying_last_updated": underlying_updated,
                "break_even": break_even,
            }
        )

    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        return frame
    numeric_cols = ["strike", "bid", "ask", "mid", "spread_pct", "delta", "gamma", "theta", "vega", "rho", "open_interest", "volume", "implied_volatility", "dte", "underlying_price", "break_even"]
    for col in numeric_cols:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def _rank_contracts(frame: pd.DataFrame, prefer_delta: float | None) -> pd.DataFrame:
    if frame.empty:
        return frame
    ranked = frame.copy()
    ranked["_spread_sort"] = ranked["spread_pct"].replace([np.inf, -np.inf], np.nan).fillna(999.0)
    ranked["_dte_sort"] = ranked["dte"].fillna(999.0)
    ranked["_oi_sort"] = ranked["open_interest"].fillna(0.0)
    sort_cols: List[str] = []
    ascending: List[bool] = []
    if prefer_delta is not None:
        ranked["_delta_score"] = (ranked["delta"].abs() - abs(prefer_delta)).abs().fillna(999.0)
        sort_cols.append("_delta_score")
        ascending.append(True)
    sort_cols.extend(["_spread_sort", "_dte_sort", "_oi_sort"])
    ascending.extend([True, True, False])
    ranked = ranked.sort_values(by=sort_cols, ascending=ascending)
    return ranked.drop(columns=[col for col in ["_delta_score", "_spread_sort", "_dte_sort", "_oi_sort"] if col in ranked.columns])


def _serialize_contract(row: pd.Series) -> Dict[str, Any]:
    data = {
        "symbol": row.get("symbol"),
        "expiration": row.get("expiration_date"),
        "option_type": row.get("option_type"),
        "strike": _safe_number(row.get("strike")),
        "bid": _safe_number(row.get("bid")),
        "ask": _safe_number(row.get("ask")),
        "mid": _safe_number(row.get("mid")),
        "spread_pct": _safe_number(row.get("spread_pct")),
        "bid_size": _safe_int(row.get("bid_size")),
        "ask_size": _safe_int(row.get("ask_size")),
        "delta": _safe_number(row.get("delta")),
        "gamma": _safe_number(row.get("gamma")),
        "theta": _safe_number(row.get("theta")),
        "vega": _safe_number(row.get("vega")),
        "rho": _safe_number(row.get("rho")),
        "open_interest": _safe_int(row.get("open_interest")),
        "volume": _safe_int(row.get("volume")),
        "dte": _safe_number(row.get("dte")),
        "implied_volatility": _safe_number(row.get("implied_volatility")),
        "break_even": _safe_number(row.get("break_even")),
        "last_updated": row.get("last_updated"),
    }
    return {key: value for key, value in data.items() if value is not None}


def _summarize_underlying(chain: pd.DataFrame) -> Optional[Dict[str, Any]]:
    if chain.empty:
        return None
    symbol_series = chain.get("underlying_symbol")
    price_series = chain.get("underlying_price")
    updated_series = chain.get("underlying_last_updated")
    symbol = symbol_series.dropna().iloc[0] if symbol_series is not None and not symbol_series.dropna().empty else None
    price = _safe_number(price_series.dropna().iloc[0]) if price_series is not None and not price_series.dropna().empty else None
    updated = updated_series.dropna().iloc[0] if updated_series is not None and not updated_series.dropna().empty else None
    if symbol is None and price is None:
        return None
    payload: Dict[str, Any] = {}
    if symbol is not None:
        payload["symbol"] = symbol
    if price is not None:
        payload["price"] = price
    if updated is not None:
        payload["last_updated"] = updated
    return payload or None


def summarize_polygon_chain(
    chain: pd.DataFrame,
    rules: Dict[str, Any] | None = None,
    *,
    top_n: int = 3,
) -> Optional[Dict[str, Any]]:
    """Produce a summary suitable for GPT consumption given a Polygon chain."""
    if chain is None or chain.empty:
        return None

    consider = chain
    filters_applied = False
    if rules:
        consider = filter_chain(chain, rules)
        filters_applied = True

    delta_range = rules.get("delta_range") if rules else None
    prefer_delta = None
    if isinstance(delta_range, (list, tuple)) and len(delta_range) == 2:
        try:
            prefer_delta = (float(delta_range[0]) + float(delta_range[1])) / 2.0
        except (TypeError, ValueError):
            prefer_delta = None

    ranked = _rank_contracts(consider, prefer_delta)
    filtered_size = len(ranked)
    if ranked.empty:
        # Fallback to the full chain if filtering removed everything
        ranked = _rank_contracts(chain, prefer_delta)
        filters_applied = False
        filtered_size = len(ranked)

    if ranked.empty:
        return None

    best_row = ranked.iloc[0]
    alt_rows = ranked.iloc[1 : max(1, min(top_n, len(ranked)))].copy()

    summary: Dict[str, Any] = {
        "source": "polygon",
        "filters_applied": filters_applied and bool(rules),
        "filter_rules": dict(rules) if isinstance(rules, dict) else None,
        "chain_size": int(len(chain)),
        "considered_size": int(filtered_size),
        "best": _serialize_contract(best_row),
        "alternatives": [_serialize_contract(row) for _, row in alt_rows.iterrows()],
    }

    underlying = _summarize_underlying(chain)
    if underlying:
        summary["underlying"] = underlying

    return summary
