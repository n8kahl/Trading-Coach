"""Shared data-source helpers (Polygon, etc.)."""

from __future__ import annotations

import asyncio
import math
from typing import Optional, Tuple

import httpx
import pandas as pd

from .config import get_settings

_CLIENT_LOCK = asyncio.Lock()
_POLYGON_CLIENT: httpx.AsyncClient | None = None
_DEFAULT_TIMEOUT = httpx.Timeout(8.0, connect=4.0)


def _parse_polygon_timeframe(timeframe: str) -> Tuple[int, str, int]:
    token = (timeframe or "5").strip().lower()
    if token in {"d", "1d", "day", "daily"}:
        return 1, "day", 250

    if token.endswith("h") and token[:-1].isdigit():
        minutes = max(int(token[:-1]), 1) * 60
    elif token.endswith("m") and token[:-1].isdigit():
        minutes = max(int(token[:-1]), 1)
    else:
        try:
            minutes = max(int(token), 1)
        except ValueError:
            minutes = 5

    if minutes >= 1440:
        return 1, "day", 250

    if minutes >= 60 and minutes % 60 == 0:
        multiplier = minutes // 60
        timespan = "hour"
    else:
        multiplier = minutes
        timespan = "minute"

    total_minutes = max(minutes * 600, minutes * 5)
    default_days = max(math.ceil(total_minutes / (60 * 6)) + 2, 5)
    return multiplier, timespan, default_days


async def _get_polygon_client() -> httpx.AsyncClient:
    global _POLYGON_CLIENT
    if _POLYGON_CLIENT is None or _POLYGON_CLIENT.is_closed:
        async with _CLIENT_LOCK:
            if _POLYGON_CLIENT is None or _POLYGON_CLIENT.is_closed:
                _POLYGON_CLIENT = httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT)
    return _POLYGON_CLIENT


async def fetch_polygon_ohlcv(symbol: str, timeframe: str, *, max_days: Optional[int] = None) -> pd.DataFrame | None:
    """Fetch OHLCV candles from Polygon."""
    settings = get_settings()
    api_key = settings.polygon_api_key
    if not api_key:
        return None
    multiplier, timespan, default_days = _parse_polygon_timeframe(timeframe)

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
    client = await _get_polygon_client()
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
    column_map = {"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close"}
    if "v" in frame.columns:
        column_map["v"] = "volume"
    available_columns = [col for col in column_map.keys() if col in frame.columns]
    frame = frame[available_columns].rename(columns={key: column_map[key] for key in available_columns})
    if "volume" not in frame.columns:
        frame["volume"] = 0.0
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    frame = frame.set_index("timestamp").sort_index()
    return frame
