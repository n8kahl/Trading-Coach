"""Polygon client utilities for planning-mode scans.

This module centralises access to Polygon's reference and aggregates APIs,
adding lightweight caching, retry handling, and concurrency limits suitable for
planning-mode scan workloads.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import httpx
import pandas as pd

from ..config import get_settings
from ..data_sources import fetch_polygon_ohlcv, fetch_yahoo_ohlcv

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = httpx.Timeout(8.0, connect=4.0, read=8.0)


def _now_ts() -> float:
    return time.monotonic()


def _cache_key(symbol: str, timeframe: str) -> Tuple[str, str]:
    return (symbol.upper(), timeframe.lower())


@dataclass
class AggregatesResult:
    """Container for multi-window aggregate results."""

    symbol: str
    windows: Dict[str, pd.DataFrame]


class PolygonAggregatesClient:
    """Client that wraps Polygon data access with caching and throttling."""

    def __init__(
        self,
        *,
        max_concurrency: int = 8,
        cache_ttl: float = 900.0,
        request_timeout: httpx.Timeout | None = None,
    ) -> None:
        self._settings = get_settings()
        self._api_key = (self._settings.polygon_api_key or "").strip()
        self._sem = asyncio.Semaphore(max(1, max_concurrency))
        self._cache_ttl = max(60.0, cache_ttl)
        self._agg_cache: MutableMapping[Tuple[str, str], Tuple[float, pd.DataFrame | None]] = {}
        self._json_cache: MutableMapping[Tuple[str, str], Tuple[float, Any]] = {}
        self._timeout = request_timeout or _DEFAULT_TIMEOUT
        self._client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()
        self._max_retries = 3

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            async with self._client_lock:
                if self._client is None or self._client.is_closed:
                    self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    # ------------------------------------------------------------------ #
    # Reference data helpers
    # ------------------------------------------------------------------ #

    async def fetch_index_constituents(self, index_ticker: str) -> List[str]:
        """Return the ticker symbols for a Polygon index (e.g., I:SPX)."""

        if not self._api_key:
            return []
        cache_key = ("index", index_ticker.upper())
        cached = self._json_cache.get(cache_key)
        if cached and _now_ts() - cached[0] <= self._cache_ttl:
            return list(cached[1])

        endpoint = f"https://api.polygon.io/v2/reference/indices/{index_ticker.upper()}/constituents"
        params = {"apiKey": self._api_key, "limit": 1000}
        client = await self._get_client()
        attempt = 0
        while attempt < self._max_retries:
            attempt += 1
            try:
                async with self._sem:
                    resp = await client.get(endpoint, params=params)
                resp.raise_for_status()
                payload = resp.json()
                results = payload.get("results") or []
                symbols = [str(item.get("ticker") or "").strip().upper() for item in results if item.get("ticker")]
                symbols = [sym for sym in symbols if sym]
                self._json_cache[cache_key] = (_now_ts(), symbols)
                return symbols
            except httpx.HTTPError as exc:
                logger.warning("Polygon index constituent fetch failed (attempt %d): %s", attempt, exc)
                await asyncio.sleep(0.75 * attempt)
        return []

    async def fetch_top_market_cap(self, limit: int) -> List[str]:
        """Return the highest market-cap tickers available from Polygon."""

        if not self._api_key:
            return []
        limit = max(1, min(limit, 1000))
        cache_key = ("top", str(limit))
        cached = self._json_cache.get(cache_key)
        if cached and _now_ts() - cached[0] <= self._cache_ttl:
            return list(cached[1])

        endpoint = "https://api.polygon.io/v3/reference/tickers"
        params = {
            "market": "stocks",
            "active": "true",
            "sort": "market_cap",
            "order": "desc",
            "limit": limit,
            "apiKey": self._api_key,
        }
        client = await self._get_client()
        attempt = 0
        symbols: List[str] = []
        cursor: Optional[str] = None
        while len(symbols) < limit and attempt < self._max_retries:
            attempt += 1
            payload: Dict[str, Any] | None = None
            try:
                query = dict(params)
                if cursor:
                    query["cursor"] = cursor
                async with self._sem:
                    resp = await client.get(endpoint, params=query)
                resp.raise_for_status()
                payload = resp.json()
            except httpx.HTTPError as exc:
                logger.warning("Polygon reference tickers fetch failed (attempt %d): %s", attempt, exc)
                await asyncio.sleep(0.75 * attempt)
                continue

            results = (payload or {}).get("results") or []
            for item in results:
                ticker = (item.get("ticker") or "").strip().upper()
                if ticker:
                    symbols.append(ticker)
                    if len(symbols) >= limit:
                        break
            cursor = (payload or {}).get("next_url")
            if not cursor:
                break

        if symbols:
            self._json_cache[cache_key] = (_now_ts(), symbols[:limit])
        return symbols[:limit]

    # ------------------------------------------------------------------ #
    # Aggregates helpers
    # ------------------------------------------------------------------ #

    async def _fetch_timeframe(self, symbol: str, timeframe: str) -> pd.DataFrame | None:
        cache_key = _cache_key(symbol, timeframe)
        cached = self._agg_cache.get(cache_key)
        if cached and _now_ts() - cached[0] <= self._cache_ttl:
            frame = cached[1]
            return frame.copy() if frame is not None else None

        if not self._api_key:
            logger.debug("Polygon API key missing; aggregates unavailable for %s", symbol)
            return None

        attempt = 0
        frame: pd.DataFrame | None = None
        while attempt < self._max_retries:
            attempt += 1
            try:
                async with self._sem:
                    frame = await fetch_polygon_ohlcv(symbol, timeframe)
                if frame is not None:
                    break
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Polygon aggregates fetch error for %s/%s (attempt %d): %s", symbol, timeframe, attempt, exc)
            await asyncio.sleep(0.5 * attempt)

        if frame is None:
            try:
                frame = await fetch_yahoo_ohlcv(symbol, timeframe)
                if frame is not None:
                    logger.debug("Yahoo fallback used for %s/%s", symbol, timeframe)
                    self._json_cache.setdefault(("fallback", symbol.upper()), (_now_ts(), True))
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Yahoo fallback failed for %s/%s: %s", symbol, timeframe, exc)

        self._agg_cache[cache_key] = (_now_ts(), frame)
        return frame.copy() if frame is not None else None

    async def fetch_symbol_windows(self, symbol: str, timeframes: Sequence[str]) -> AggregatesResult:
        tasks = [self._fetch_timeframe(symbol, tf) for tf in timeframes]
        results = await asyncio.gather(*tasks)
        windows: Dict[str, pd.DataFrame] = {}
        for tf, frame in zip(timeframes, results):
            if frame is not None and not frame.empty:
                windows[tf.lower()] = frame
        return AggregatesResult(symbol=symbol.upper(), windows=windows)

    async def fetch_many(self, symbols: Sequence[str], timeframes: Sequence[str]) -> Dict[str, AggregatesResult]:
        unique_symbols = []
        seen = set()
        for sym in symbols:
            token = (sym or "").strip().upper()
            if token and token not in seen:
                seen.add(token)
                unique_symbols.append(token)
        tasks = [self.fetch_symbol_windows(sym, timeframes) for sym in unique_symbols]
        results = await asyncio.gather(*tasks)
        return {result.symbol: result for result in results}


__all__ = ["PolygonAggregatesClient", "AggregatesResult"]
