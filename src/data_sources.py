"""Shared data-source helpers (Polygon, etc.)."""

from __future__ import annotations

import asyncio
import math
from typing import Optional

import httpx
import pandas as pd

from .config import get_settings

_YAHOO_INTERVAL_MAP = {
    "1m": ("5d", "1m"),
    "5m": ("5d", "5m"),
    "15m": ("1mo", "15m"),
    "1h": ("3mo", "60m"),
    "d": ("1y", "1d"),
}


def _canonical_timeframe(timeframe: str) -> str:
    token = (timeframe or "5").strip().lower()
    if token in {"1", "1m", "1min"}:
        return "1m"
    if token in {"3", "3m"}:
        return "1m"
    if token in {"5", "5m", "5min"}:
        return "5m"
    if token in {"10", "10m"}:
        return "5m"
    if token in {"15", "15m", "15min"}:
        return "15m"
    if token in {"30", "30m"}:
        return "15m"
    if token in {"45", "45m"}:
        return "15m"
    if token in {"60", "1h", "60m", "h", "1hr"}:
        return "1h"
    if token in {"d", "1d", "day", "daily"}:
        return "d"
    if token.isdigit():
        minutes = int(token)
        if minutes <= 1:
            return "1m"
        if minutes <= 5:
            return "5m"
        if minutes <= 15:
            return "15m"
        if minutes >= 60:
            return "1h"
    return "5m"


async def fetch_polygon_ohlcv(symbol: str, timeframe: str, *, max_days: Optional[int] = None) -> pd.DataFrame | None:
    """Fetch OHLCV candles from Polygon."""
    settings = get_settings()
    api_key = settings.polygon_api_key
    if not api_key:
        return None

    tf = (timeframe or "5").upper()
    if tf == "D":
        multiplier, timespan = 1, "day"
        default_days = 250
    else:
        try:
            minutes = int(tf)
        except ValueError:
            minutes = 5
        multiplier, timespan = minutes, "minute"
        total_minutes = max(minutes * 600, minutes * 5)
        default_days = max(math.ceil(total_minutes / (60 * 6)) + 2, 5)

    days_back = max_days if max_days is not None else default_days

    now = pd.Timestamp.utcnow()
    end = now + pd.Timedelta(minutes=multiplier)
    start = now - pd.Timedelta(days=days_back)
    frm = start.normalize().date().isoformat()
    to = (end.normalize() + pd.Timedelta(days=1)).date().isoformat()

    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol.upper()}/range/{multiplier}/{timespan}/{frm}/{to}"
    params = {
        "adjusted": "true",
        "sort": "desc",
        "limit": 5000,
        "apiKey": api_key,
    }
    timeout = httpx.Timeout(8.0, connect=4.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
        except httpx.HTTPError:
            return None

    data = resp.json()
    results = data.get("results")
    if not results:
        return None

    frame = pd.DataFrame(results)
    frame = frame[["t", "o", "h", "l", "c", "v"]].rename(
        columns={"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    frame = frame.set_index("timestamp").sort_index()
    return frame


async def fetch_yahoo_ohlcv(symbol: str, timeframe: str) -> pd.DataFrame | None:
    """Fetch OHLCV candles from Yahoo Finance as a secondary source."""

    interval = _canonical_timeframe(timeframe)
    range_span, yahoo_interval = _YAHOO_INTERVAL_MAP.get(interval, ("5d", "5m"))

    def _request() -> pd.DataFrame | None:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol.upper()}"
        params = {
            "interval": yahoo_interval,
            "range": range_span,
            "includePrePost": "false",
        }
        try:
            with httpx.Client(timeout=6.0) as client:
                resp = client.get(url, params=params)
                resp.raise_for_status()
        except httpx.HTTPError:
            return None

        payload = resp.json()
        try:
            chart = payload["chart"]
            if chart.get("error"):
                return None
            result = chart["result"][0]
        except (KeyError, IndexError, TypeError, ValueError):
            return None

        timestamps = result.get("timestamp")
        if not timestamps:
            return None
        quote_payloads = result.get("indicators", {}).get("quote") or []
        if not quote_payloads:
            return None
        quote = quote_payloads[0]
        frame = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(timestamps, unit="s", utc=True),
                "open": quote.get("open"),
                "high": quote.get("high"),
                "low": quote.get("low"),
                "close": quote.get("close"),
                "volume": quote.get("volume"),
            }
        ).dropna()
        if frame.empty:
            return None
        frame = frame.set_index("timestamp").sort_index()
        return frame

    return await asyncio.to_thread(_request)
