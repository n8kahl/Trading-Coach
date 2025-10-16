"""Sector and relative strength context helpers."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, Optional, Tuple
import time

import httpx

from config import get_settings

_POLYGON_BASE = "https://api.polygon.io"
_CACHE_TTL = 60.0
_CHANGE_CACHE: Dict[str, Tuple[float, Optional[float]]] = {}

_SECTOR_MAP: Dict[str, str] = {
    "AAPL": "XLK",
    "MSFT": "XLK",
    "TSLA": "XLY",
    "NVDA": "XLK",
    "META": "XLC",
    "AMZN": "XLY",
}


def _symbol_sector(symbol: str) -> str:
    return _SECTOR_MAP.get(symbol.upper(), "SPY")


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _daily_change(symbol: str) -> Optional[float]:
    settings = get_settings()
    api_key = settings.polygon_api_key
    if not api_key:
        return None
    key = symbol.upper()
    now = time.monotonic()
    cached = _CHANGE_CACHE.get(key)
    if cached and now - cached[0] < _CACHE_TTL:
        return cached[1]
    end = date.today()
    start = end - timedelta(days=6)
    url = f"{_POLYGON_BASE}/v2/aggs/ticker/{symbol.upper()}/range/1/day/{start}/{end}"
    params = {"adjusted": "true", "sort": "desc", "limit": 2, "apiKey": api_key}
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            results = resp.json().get("results") or []
    except httpx.HTTPError:
        return None
    if len(results) < 2:
        return None
    try:
        latest = float(results[0].get("c"))
        prior = float(results[1].get("c"))
    except (TypeError, ValueError):
        return None
    if prior <= 0:
        result = None
    else:
        result = (latest / prior) - 1.0
    _CHANGE_CACHE[key] = (now, result)
    return result


def sector_strength(symbol: str, as_of: str | None = None) -> Dict[str, float | str | None]:
    """Return a lightweight sector relative strength snapshot."""

    sector = _symbol_sector(symbol)
    sector_change = _daily_change(sector) or 0.0
    spy_change = _daily_change("SPY") or 0.0
    rel = sector_change - spy_change
    rel_vs_spy = round(_clamp(0.5 + rel / 0.04), 3)
    zscore = round(rel / 0.01, 2)
    return {
        "sector": sector,
        "rel_vs_spy": rel_vs_spy,
        "zscore": zscore,
    }


def peer_rel_strength(symbol: str, as_of: str | None = None) -> Dict[str, float]:
    """Return relative strength versus the benchmark."""

    symbol_change = _daily_change(symbol) or 0.0
    spy_change = _daily_change("SPY") or 0.0
    diff = symbol_change - spy_change
    rs = round(_clamp(0.5 + diff / 0.04), 3)
    return {"rs_vs_benchmark": rs}


__all__ = ["sector_strength", "peer_rel_strength"]
