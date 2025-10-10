"""Async Tradier client helpers used for option contract pricing."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import httpx
import pandas as pd

from .config import get_settings

logger = logging.getLogger(__name__)


class TradierNotConfiguredError(RuntimeError):
    """Raised when Tradier credentials are missing."""


@dataclass(slots=True)
class TradierConfig:
    token: str
    base_url: str


_CACHE_TTL_SECONDS = 15.0
_CHAIN_CACHE: Dict[Tuple[str, str | None], Tuple[float, pd.DataFrame]] = {}
_QUOTE_CACHE: Dict[Tuple[str, ...], Tuple[float, Dict[str, Dict[str, Any]]]] = {}


def _get_tradier_config() -> TradierConfig:
    settings = get_settings()
    token = settings.tradier_token
    if not token:
        raise TradierNotConfiguredError("Tradier API token is not configured.")
    base_url = settings.tradier_base_url.rstrip("/")
    return TradierConfig(token=token, base_url=base_url)


async def _fetch_json(client: httpx.AsyncClient, url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    response = await client.get(url, params=params)
    response.raise_for_status()
    return response.json()


async def _resolve_expiration(client: httpx.AsyncClient, symbol: str, requested: str | None) -> str:
    if requested:
        return requested
    data = await _fetch_json(
        client,
        "/v1/markets/options/expirations",
        params={"symbol": symbol, "includeAllRoots": "true", "strikes": "false"},
    )
    expirations = data.get("expirations", {})
    dates = expirations.get("date") if isinstance(expirations, dict) else []
    if isinstance(dates, str):
        return dates
    if isinstance(dates, list) and dates:
        return dates[0]
    raise RuntimeError(f"No option expirations returned for {symbol}")


def _chain_to_dataframe(options: List[Dict[str, Any]]) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for opt in options:
        greeks = opt.get("greeks") or {}
        bid = opt.get("bid")
        ask = opt.get("ask")
        mid = None
        spread_pct = None
        if isinstance(bid, (int, float)) and isinstance(ask, (int, float)) and (bid or ask):
            mid = (bid + ask) / 2 if bid is not None and ask is not None else None
            if mid and mid > 0:
                spread_pct = (ask - bid) / mid
        expiration = opt.get("expiration_date")
        dte = None
        if expiration:
            try:
                expiry_ts = pd.Timestamp(expiration).normalize()
                dte = (expiry_ts - pd.Timestamp.utcnow().normalize()).days
            except Exception:  # pragma: no cover - defensive
                dte = None
        records.append(
            {
                "symbol": opt.get("symbol"),
                "underlying": opt.get("underlying"),
                "option_type": opt.get("option_type"),
                "strike": opt.get("strike"),
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "delta": greeks.get("delta"),
                "gamma": greeks.get("gamma"),
                "theta": greeks.get("theta"),
                "vega": greeks.get("vega"),
                "iv": greeks.get("iv"),
                "volume": opt.get("volume") or 0,
                "open_interest": opt.get("open_interest") or 0,
                "dte": dte,
                "expiration_date": expiration,
                "spread_pct": spread_pct if spread_pct is not None else 9.99,
            }
        )
    return pd.DataFrame.from_records(records)


async def fetch_option_chain(symbol: str, expiration: str | None = None) -> pd.DataFrame:
    """Retrieve a Tradier option chain as a pandas DataFrame."""
    config = _get_tradier_config()
    headers = {
        "Authorization": f"Bearer {config.token}",
        "Accept": "application/json",
    }
    async with httpx.AsyncClient(base_url=config.base_url, headers=headers, timeout=10.0) as client:
        resolved_expiry = await _resolve_expiration(client, symbol, expiration)
        chain_payload = await _fetch_json(
            client,
            "/v1/markets/options/chains",
            params={"symbol": symbol, "expiration": resolved_expiry},
        )
    options = chain_payload.get("options", {}).get("option", [])
    if not options:
        return pd.DataFrame()
    if isinstance(options, dict):
        options = [options]
    frame = _chain_to_dataframe(options)
    return frame


async def select_tradier_contract(symbol: str, prefer_delta: float | None = 0.5) -> Dict[str, Any] | None:
    """Fetch the option chain and return a single contract suggestion."""
    from .contract_selector import filter_chain, pick_best_contract  # late import to avoid cycle

    chain = await fetch_option_chain(symbol)
    if chain.empty:
        logger.warning("Tradier chain empty for %s", symbol)
        return None

    rules = {
        "dte_range": (0, 90),
        "delta_range": (0.2, 0.8),
        "max_spread_pct": 0.25,
        "min_open_interest": 10,
        "min_volume": 1,
    }
    filtered = filter_chain(chain, rules)
    best = pick_best_contract(filtered, prefer_delta=prefer_delta)
    if best is None:
        logger.info("No contract matched filters for %s", symbol)
        return None
    return {
        "symbol": best.get("symbol"),
        "strike": best.get("strike"),
        "expiration": best.get("expiration_date"),
        "option_type": best.get("option_type"),
        "bid": best.get("bid"),
        "ask": best.get("ask"),
        "mid": best.get("mid"),
        "delta": best.get("delta"),
        "dte": best.get("dte"),
        "spread_pct": best.get("spread_pct"),
    }


def _chunk_symbols(symbols: Iterable[str], size: int = 40) -> List[List[str]]:
    chunk: List[str] = []
    chunks: List[List[str]] = []
    for sym in symbols:
        chunk.append(sym)
        if len(chunk) >= size:
            chunks.append(chunk)
            chunk = []
    if chunk:
        chunks.append(chunk)
    return chunks


def _normalize_quote_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    greeks = entry.get("greeks") or {}
    summary = {
        "symbol": entry.get("symbol"),
        "bid": entry.get("bid"),
        "ask": entry.get("ask"),
        "last": entry.get("last"),
        "volume": entry.get("volume"),
        "open_interest": entry.get("open_interest"),
        "strike": entry.get("strike"),
        "option_type": entry.get("option_type"),
        "expiration_date": entry.get("expiration_date"),
        "description": entry.get("description"),
        "delta": greeks.get("delta"),
        "gamma": greeks.get("gamma"),
        "theta": greeks.get("theta"),
        "vega": greeks.get("vega"),
        "rho": greeks.get("rho"),
        "iv": greeks.get("iv"),
    }
    return summary


async def fetch_option_chain_cached(symbol: str, expiration: str | None = None) -> pd.DataFrame:
    key = (symbol.upper(), expiration)
    now = time.monotonic()
    cached = _CHAIN_CACHE.get(key)
    if cached and now - cached[0] < _CACHE_TTL_SECONDS:
        return cached[1].copy()
    frame = await fetch_option_chain(symbol, expiration)
    _CHAIN_CACHE[key] = (now, frame.copy())
    return frame


async def fetch_option_quotes(symbols: List[str]) -> Dict[str, Dict[str, Any]]:
    cleaned = sorted({sym for sym in symbols if sym})
    if not cleaned:
        return {}
    key = tuple(cleaned)
    now = time.monotonic()
    cached = _QUOTE_CACHE.get(key)
    if cached and now - cached[0] < _CACHE_TTL_SECONDS:
        return {k: dict(v) for k, v in cached[1].items()}

    config = _get_tradier_config()
    headers = {
        "Authorization": f"Bearer {config.token}",
        "Accept": "application/json",
    }
    results: Dict[str, Dict[str, Any]] = {}
    async with httpx.AsyncClient(base_url=config.base_url, headers=headers, timeout=10.0) as client:
        for chunk in _chunk_symbols(cleaned):
            params = {
                "symbols": ",".join(chunk),
                "greeks": "true",
            }
            try:
                payload = await _fetch_json(client, "/v1/markets/options/quotes", params=params)
            except httpx.HTTPError as exc:
                logger.warning("Tradier quotes fetch failed for %s: %s", chunk, exc)
                continue
            quotes = payload.get("quotes", {}).get("quote")
            if not quotes:
                continue
            if isinstance(quotes, dict):
                quotes = [quotes]
            for entry in quotes:
                normalized = _normalize_quote_entry(entry)
                symbol = normalized.get("symbol")
                if symbol:
                    results[symbol] = normalized

    _QUOTE_CACHE[key] = (now, {k: dict(v) for k, v in results.items()})
    return results
