"""Shared data-source helpers (Polygon, etc.)."""

from __future__ import annotations

import asyncio
import math
import os
from typing import Optional, Tuple

import httpx
import pandas as pd
import logging

from .config import get_settings, get_massive_api_key

_CLIENT_LOCK = asyncio.Lock()
_POLYGON_CLIENT: httpx.AsyncClient | None = None
_DEFAULT_TIMEOUT = httpx.Timeout(8.0, connect=4.0)
_MASSIVE_BASE = os.getenv("MARKETDATA_BASE_URL", "https://api.massive.com").rstrip("/")
_POLYGON_BASE = os.getenv("POLYGON_BASE_URL", "https://api.polygon.io").rstrip("/")
_BASE_URL = _MASSIVE_BASE or _POLYGON_BASE
logger = logging.getLogger(__name__)


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


async def fetch_polygon_ohlcv(
    symbol: str,
    timeframe: str,
    *,
    max_days: Optional[int] = None,
    include_extended: bool = False,
) -> pd.DataFrame | None:
    """Fetch OHLCV candles from Polygon."""
    settings = get_settings()
    api_key = get_massive_api_key(settings)
    if not api_key:
        return None
    multiplier, timespan, default_days = _parse_polygon_timeframe(timeframe)

    days_back = max_days if max_days is not None else default_days

    now = pd.Timestamp.utcnow()
    end = now + pd.Timedelta(minutes=multiplier)
    start = now - pd.Timedelta(days=days_back)
    frm = start.normalize().date().isoformat()
    to = (end.normalize() + pd.Timedelta(days=1)).date().isoformat()

    client = await _get_polygon_client()

    async def _fetch_range(start_str: str, end_str: str) -> pd.DataFrame | None:
        params = {
            "adjusted": "true",
            "sort": "desc",
            "limit": 5000,
            "apiKey": api_key,
        }
        if include_extended:
            params["include_extended"] = "true"
        url = f"{_BASE_URL}/v2/aggs/ticker/{symbol.upper()}/range/{multiplier}/{timespan}/{start_str}/{end_str}"
        logger.debug(
            "massive_ohlcv_request",
            extra={
                "symbol": symbol.upper(),
                "timeframe": timeframe,
                "url": url,
                "start": start_str,
                "end": end_str,
            },
        )
        try:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning(
                "massive_ohlcv_http_error",
                extra={
                    "symbol": symbol.upper(),
                    "timeframe": timeframe,
                    "error": str(exc),
                },
            )
            return None
        data = resp.json()
        results = data.get("results")
        if not results:
            logger.info(
                "massive_ohlcv_empty",
                extra={
                    "symbol": symbol.upper(),
                    "timeframe": timeframe,
                    "start": start_str,
                    "end": end_str,
                },
            )
            return None
        frame = pd.DataFrame(results)
        if frame.empty:
            return None
        return frame

    frame = await _fetch_range(frm, to)
    if frame is None or frame.empty:
        try:
            end_ts = pd.Timestamp(to)
            start_ts = pd.Timestamp(frm)
        except Exception:
            end_ts = pd.Timestamp.utcnow()
            start_ts = end_ts - pd.Timedelta(days=days_back or 5)
        window = end_ts - start_ts
        for offset in range(1, 6):
            shifted_end = end_ts - pd.Timedelta(days=offset)
            shifted_start = shifted_end - window
            logger.info(
                "massive_ohlcv_backfill_attempt",
                extra={
                    "symbol": symbol.upper(),
                    "timeframe": timeframe,
                    "attempt": offset,
                    "start": shifted_start.date().isoformat(),
                    "end": shifted_end.date().isoformat(),
                },
            )
            fallback_frame = await _fetch_range(
                shifted_start.date().isoformat(),
                shifted_end.date().isoformat(),
            )
            if fallback_frame is not None and not fallback_frame.empty:
                frame = fallback_frame
                logger.info(
                    "massive_ohlcv_backfill_success",
                    extra={
                        "symbol": symbol.upper(),
                        "timeframe": timeframe,
                        "attempt": offset,
                    },
                )
                break

    if frame is None or frame.empty:
        logger.error(
            "massive_ohlcv_unavailable",
            extra={
                "symbol": symbol.upper(),
                "timeframe": timeframe,
            },
        )
        return None

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


__all__ = ["fetch_polygon_ohlcv"]
