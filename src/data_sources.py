"""Shared data-source helpers (Polygon, etc.)."""

from __future__ import annotations

import math
from datetime import datetime, time, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import httpx
import pandas as pd

from .config import get_settings

_ET = ZoneInfo("America/New_York")


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


async def last_price_asof(symbol: str, as_of: datetime) -> Optional[float]:
    """Return the last Polygon close on or before the provided timestamp."""

    if as_of is None:
        return None

    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=_ET)
    as_of_utc = as_of.astimezone(timezone.utc)
    day_start_et = datetime.combine(as_of.astimezone(_ET).date(), time(0, 0), tzinfo=_ET)
    day_start_utc = day_start_et.astimezone(timezone.utc)

    intraday = await fetch_polygon_ohlcv(symbol, "1", max_days=7)
    if intraday is not None and not intraday.empty:
        window = intraday[(intraday.index >= day_start_utc) & (intraday.index <= as_of_utc)]
        if not window.empty:
            try:
                return float(window["close"].iloc[-1])
            except Exception:
                pass

    # Fallback to the most recent daily close on or before as_of
    daily = await fetch_polygon_ohlcv(symbol, "D", max_days=10)
    if daily is None or daily.empty:
        return None
    recent = daily[daily.index <= as_of_utc]
    if recent.empty:
        try:
            return float(daily["close"].iloc[-1])
        except Exception:
            return None
    try:
        return float(recent["close"].iloc[-1])
    except Exception:
        return None
