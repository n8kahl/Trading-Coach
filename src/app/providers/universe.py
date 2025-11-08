"""Ticker universe helpers backed by Polygon reference data.

This module builds a liquid universe suitable for scanning when the client
does not supply an explicit ticker list.  Results are cached briefly to avoid
excess API usage and are weighted by sector relative strength derived from
SPDR ETFs.
"""

from __future__ import annotations

import asyncio
import math
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, parse_qs

import httpx

from src.config import get_settings, get_massive_api_key

_POLYGON_BASE = "https://api.massive.com"
INDEX_PRIORITY = ['SPX', 'NDX']
_UNIVERSE_CACHE: Optional[Tuple[float, List[Dict[str, Any]]]] = None
_UNIVERSE_CACHE_TTL = 900.0  # 15 minutes
_SECTOR_CACHE: Optional[Tuple[float, Dict[str, float]]] = None
_SECTOR_CACHE_TTL = 180.0
_MAX_REFERENCE_RESULTS = 320

_SECTOR_FOCUS = {
    "communication services": "XLC",
    "consumer discretionary": "XLY",
    "consumer staples": "XLP",
    "energy": "XLE",
    "financials": "XLF",
    "health care": "XLV",
    "healthcare": "XLV",
    "industrials": "XLI",
    "materials": "XLB",
    "real estate": "XLRE",
    "technology": "XLK",
    "information technology": "XLK",
    "utilities": "XLU",
}

_SECTOR_SYNONYMS = {
    "tech": "technology",
    "it": "technology",
    "health": "healthcare",
    "health care": "healthcare",
    "financial": "financials",
    "industrials": "industrials",
    "industrial": "industrials",
    "energy": "energy",
    "utilities": "utilities",
    "consumer discretionary": "consumer discretionary",
    "consumer staples": "consumer staples",
    "communications": "communication services",
    "communication": "communication services",
    "realestate": "real estate",
    "real-estate": "real estate",
}


def _normalise_sector(value: str | None) -> Optional[str]:
    if not value:
        return None
    token = value.strip().lower()
    if not token:
        return None
    if token in _SECTOR_SYNONYMS:
        token = _SECTOR_SYNONYMS[token]
    return token


def _extract_cursor(next_url: str | None) -> Optional[str]:
    if not next_url:
        return None
    parsed = urlparse(next_url)
    query = parse_qs(parsed.query)
    cursor_list = query.get("cursor")
    if not cursor_list:
        return None
    return cursor_list[0]


async def _fetch_top_list(api_key: str) -> List[Dict[str, Any]]:
    global _UNIVERSE_CACHE

    now = time.monotonic()
    cached = _UNIVERSE_CACHE
    if cached and now - cached[0] < _UNIVERSE_CACHE_TTL:
        return list(cached[1])

    url = f"{_POLYGON_BASE}/v3/reference/tickers"
    params = {
        "market": "stocks",
        "active": "true",
        "order": "desc",
        "sort": "market_cap",
        "limit": 100,
    }
    results: List[Dict[str, Any]] = []
    cursor: Optional[str] = None

    timeout = httpx.Timeout(10.0, connect=4.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        while len(results) < _MAX_REFERENCE_RESULTS:
            req_params = dict(params)
            if cursor:
                req_params["cursor"] = cursor
            try:
                resp = await client.get(url, params=req_params, headers={"Authorization": f"Bearer {api_key}"})
                resp.raise_for_status()
            except httpx.HTTPError:
                break
            payload = resp.json()
            page = payload.get("results") or []
            results.extend(page)
            cursor = _extract_cursor(payload.get("next_url"))
            if not cursor:
                break

    if results:
        _UNIVERSE_CACHE = (time.monotonic(), results[: _MAX_REFERENCE_RESULTS])
    return results[: _MAX_REFERENCE_RESULTS]


async def _percent_change(api_key: str, client: httpx.AsyncClient, symbol: str) -> Optional[float]:
    params = {"adjusted": "true"}
    url = f"{_POLYGON_BASE}/v2/aggs/ticker/{symbol.upper()}/prev"
    try:
        resp = await client.get(url, params=params, headers={"Authorization": f"Bearer {api_key}"})
        resp.raise_for_status()
    except httpx.HTTPError:
        return None
    data = resp.json()
    results = data.get("results") or []
    if not results:
        return None
    row = results[0]
    open_price = row.get("o")
    close_price = row.get("c")
    if not isinstance(open_price, (int, float)) or not isinstance(close_price, (int, float)):
        return None
    if open_price <= 0:
        return None
    try:
        return float(close_price) / float(open_price) - 1.0
    except ZeroDivisionError:
        return None


async def _sector_biases(api_key: str) -> Dict[str, float]:
    global _SECTOR_CACHE

    now = time.monotonic()
    cached = _SECTOR_CACHE
    if cached and now - cached[0] < _SECTOR_CACHE_TTL:
        return dict(cached[1])

    symbols = set(_SECTOR_FOCUS.values())
    symbols.update({"SPY", "QQQ"})

    timeout = httpx.Timeout(6.0, connect=3.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        tasks = [asyncio.create_task(_percent_change(api_key, client, symbol)) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    changes: Dict[str, float] = {}
    for symbol, outcome in zip(symbols, results):
        if isinstance(outcome, Exception) or outcome is None:
            continue
        changes[symbol.upper()] = float(outcome)

    spy_change = changes.get("SPY", 0.0)
    qqq_change = changes.get("QQQ", spy_change)
    biases: Dict[str, float] = {}
    for sector_name, etf in _SECTOR_FOCUS.items():
        sector_change = changes.get(etf.upper())
        if sector_change is None:
            continue
        relative = sector_change - (qqq_change if etf.upper() == 'QQQ' else spy_change)
        # Damp the impact so sectors are nudged, not dominated.
        biases[sector_name] = max(-0.20, min(0.20, relative * 5.0))

    _SECTOR_CACHE = (time.monotonic(), biases)
    return biases


def _style_filter(style: Optional[str], market_cap: float) -> bool:
    if not style:
        return True
    token = style.lower()
    if token in {"scalp", "intraday"}:
        return market_cap >= 5_000_000_000  # $5B+
    if token in {"swing"}:
        return market_cap >= 3_000_000_000  # $3B+
    if token in {"leap", "leaps"}:
        return market_cap >= 20_000_000_000  # $20B+
    return True


async def load_universe(
    *,
    style: Optional[str],
    sector: Optional[str],
    limit: int,
) -> List[str]:
    """Return a list of tickers sized to `limit`, weighted by Polygon data."""

    settings = get_settings()
    api_key = get_massive_api_key(settings)
    if not api_key:
        # Without Polygon, fall back to a static set of large-cap names.
        fallback = [
            "SPY",
            "QQQ",
            "IWM",
            "DIA",
            "AAPL",
            "MSFT",
            "NVDA",
            "META",
            "GOOG",
            "AMZN",
            "TSLA",
            "NFLX",
            "UNH",
            "JPM",
            "XOM",
            "BAC",
            "AMD",
            "KO",
            "PG",
            "AVGO",
        ]
        return fallback[: max(10, min(limit, len(fallback)))]

    reference = await _fetch_top_list(api_key)
    if not reference:
        return []

    biases = await _sector_biases(api_key)
    sector_filter = _normalise_sector(sector)
    style_token = style.lower() if isinstance(style, str) else None

    scored: List[Tuple[float, str]] = []
    for row in reference:
        ticker = str(row.get("ticker") or "").upper()
        if not ticker:
            continue
        if ticker in INDEX_PRIORITY:
            scored.append((1.0, ticker))
            continue
        if ticker.endswith(".W") or ticker.endswith(".U"):
            continue
        primary_exchange = (row.get("primary_exchange") or "").upper()
        if primary_exchange.startswith("OTC"):
            continue
        market_cap = float(row.get("market_cap") or 0.0)
        if not math.isfinite(market_cap) or market_cap <= 0:
            continue
        if not _style_filter(style_token, market_cap):
            continue

        sector_name = _normalise_sector(row.get("sector") or row.get("sic_description"))
        if sector_filter and sector_name != sector_filter:
            continue

        weight = market_cap
        if sector_name and sector_name in biases:
            weight *= 1.0 + biases[sector_name]
        scored.append((weight, ticker))

    if not scored:
        # If filters removed everything, relax sector constraint.
        if sector_filter:
            return await load_universe(style=style, sector=None, limit=limit)
        return []

    scored.sort(key=lambda item: item[0], reverse=True)
    unique: Dict[str, None] = {}
    for _, ticker in scored:
        if ticker not in unique:
            unique[ticker] = None
        if len(unique) >= limit:
            break

    return list(unique.keys())


__all__ = ["load_universe"]
