"""Index feed health monitoring for Polygon + Tradier."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import httpx

from config import get_settings
from .index_common import POLYGON_INDEX_TICKERS


@dataclass(slots=True)
class FeedStatus:
    source: str
    symbol: str
    status: str  # healthy | degraded | failed
    updated_at: Optional[str] = None
    latency_ms: Optional[float] = None
    error: Optional[str] = None
    details: Dict[str, Any] = None  # type: ignore[assignment]

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "source": self.source,
            "symbol": self.symbol,
            "status": self.status,
            "updated_at": self.updated_at,
            "latency_ms": self.latency_ms,
        }
        if self.error:
            payload["error"] = self.error
        if self.details:
            payload["details"] = dict(self.details)
        return payload


_POLYGON_CACHE: Dict[str, Tuple[float, Dict[str, Any], FeedStatus]] = {}
_TRADIER_CACHE: Dict[Tuple[str, Optional[str]], Tuple[float, Dict[str, Any], FeedStatus]] = {}
_UNIVERSAL_CACHE: Dict[str, Tuple[float, Dict[str, Any], FeedStatus]] = {}

_CACHE_TTL = 30.0


async def polygon_index_snapshot(symbol: str, *, force_refresh: bool = False) -> Tuple[Dict[str, Any] | None, FeedStatus]:
    """Fetch Polygon index option snapshot and return diagnostics."""
    settings = get_settings()
    api_key = settings.polygon_api_key
    base = symbol.upper()
    polygon_symbol = POLYGON_INDEX_TICKERS.get(base, symbol.upper())
    cache_key = base
    cached = _POLYGON_CACHE.get(cache_key)
    now = time.monotonic()
    if cached and not force_refresh and now - cached[0] < _CACHE_TTL:
        return dict(cached[1]), cached[2]

    if not api_key:
        status = FeedStatus(
            source="polygon",
            symbol=base,
            status="failed",
            error="missing_api_key",
            details={"message": "Polygon API key required"},
        )
        return None, status

    url = f"https://api.polygon.io/v3/snapshot/options/{polygon_symbol}"
    params = {"apiKey": api_key}
    timeout = httpx.Timeout(6.0, connect=3.0)
    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.get(url, params=params)
            latency = (time.perf_counter() - started) * 1000.0
            if resp.status_code == 429:
                status = FeedStatus(
                    source="polygon",
                    symbol=base,
                    status="degraded",
                    latency_ms=latency,
                    error="rate_limited",
                    details={"status_code": resp.status_code},
                )
                _POLYGON_CACHE[cache_key] = (now, {}, status)
                return None, status
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            latency = (time.perf_counter() - started) * 1000.0
            status = FeedStatus(
                source="polygon",
                symbol=base,
                status="failed",
                latency_ms=latency,
                error=f"http_{exc.response.status_code}",
                details={"status_code": exc.response.status_code},
            )
            _POLYGON_CACHE[cache_key] = (now, {}, status)
            return None, status
        except httpx.HTTPError as exc:
            latency = (time.perf_counter() - started) * 1000.0
            status = FeedStatus(
                source="polygon",
                symbol=base,
                status="failed",
                latency_ms=latency,
                error="network_error",
                details={"message": str(exc)},
            )
            _POLYGON_CACHE[cache_key] = (now, {}, status)
            return None, status

    payload = resp.json()
    results = payload.get("results") or []
    if not results:
        latency = (time.perf_counter() - started) * 1000.0
        status = FeedStatus(
            source="polygon",
            symbol=base,
            status="degraded",
            latency_ms=latency,
            error="empty_payload",
        )
        _POLYGON_CACHE[cache_key] = (now, payload, status)
        return payload, status

    first = results[0]
    updated_at = None
    try:
        updated_at = first.get("last_quote", {}).get("last_updated") or first.get("last_trade", {}).get("sip_timestamp")
    except Exception:  # pragma: no cover - defensive
        updated_at = None
    latency = (time.perf_counter() - started) * 1000.0
    status = FeedStatus(
        source="polygon",
        symbol=base,
        status="healthy",
        latency_ms=latency,
        updated_at=updated_at,
        details={"contracts": len(results)},
    )
    _POLYGON_CACHE[cache_key] = (now, payload, status)
    return payload, status


async def tradier_index_greeks(symbol: str, expiration: Optional[str]) -> Tuple[Dict[str, Any] | None, FeedStatus]:
    """Fetch Tradier ORATS greeks for an index chain."""
    base = symbol.upper()
    cache_key = (base, expiration)
    now = time.monotonic()
    cached = _TRADIER_CACHE.get(cache_key)
    if cached and now - cached[0] < _CACHE_TTL:
        return dict(cached[1]), cached[2]

    settings = get_settings()
    token = settings.tradier_token
    if not token:
        status = FeedStatus(
            source="tradier",
            symbol=base,
            status="failed",
            error="missing_token",
            details={"message": "Tradier token missing"},
        )
        return None, status

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
    }
    params = {"symbol": base}
    if expiration:
        params["expiration"] = expiration

    timeout = httpx.Timeout(6.0, connect=3.0)
    url = "https://api.tradier.com/v1/markets/options/greeks"
    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.get(url, headers=headers, params=params)
            latency = (time.perf_counter() - started) * 1000.0
            if resp.status_code == 429:
                status = FeedStatus(
                    source="tradier",
                    symbol=base,
                    status="degraded",
                    latency_ms=latency,
                    error="rate_limited",
                    details={"status_code": resp.status_code},
                )
                _TRADIER_CACHE[cache_key] = (now, {}, status)
                return None, status
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            latency = (time.perf_counter() - started) * 1000.0
            status = FeedStatus(
                source="tradier",
                symbol=base,
                status="failed",
                latency_ms=latency,
                error=f"http_{exc.response.status_code}",
                details={"status_code": exc.response.status_code},
            )
            _TRADIER_CACHE[cache_key] = (now, {}, status)
            return None, status
        except httpx.HTTPError as exc:
            latency = (time.perf_counter() - started) * 1000.0
            status = FeedStatus(
                source="tradier",
                symbol=base,
                status="failed",
                latency_ms=latency,
                error="network_error",
                details={"message": str(exc)},
            )
            _TRADIER_CACHE[cache_key] = (now, {}, status)
            return None, status

    payload = resp.json()
    greeks = payload.get("greeks")
    if not greeks:
        latency = (time.perf_counter() - started) * 1000.0
        status = FeedStatus(
            source="tradier",
            symbol=base,
            status="degraded",
            latency_ms=latency,
            error="empty_payload",
        )
        _TRADIER_CACHE[cache_key] = (now, payload, status)
        return payload, status

    updated_at = None
    try:
        first = greeks[0] if isinstance(greeks, list) else greeks
        updated_at = first.get("updated") or first.get("updated_at")
    except Exception:  # pragma: no cover - defensive
        updated_at = None
    latency = (time.perf_counter() - started) * 1000.0
    status = FeedStatus(
        source="tradier",
        symbol=base,
        status="healthy",
        latency_ms=latency,
        updated_at=updated_at,
        details={"count": len(greeks) if isinstance(greeks, list) else 1},
    )
    _TRADIER_CACHE[cache_key] = (now, payload, status)
    return payload, status


async def polygon_universal_snapshot() -> Tuple[Dict[str, Any] | None, FeedStatus]:
    """Fetch universal snapshot for quick sanity checks on index + ETF proxies."""
    cache_key = "universal"
    now = time.monotonic()
    cached = _UNIVERSAL_CACHE.get(cache_key)
    if cached and now - cached[0] < _CACHE_TTL:
        return dict(cached[1]), cached[2]

    settings = get_settings()
    api_key = settings.polygon_api_key
    if not api_key:
        status = FeedStatus(
            source="polygon",
            symbol="multi",
            status="failed",
            error="missing_api_key",
        )
        return None, status

    params = {
        "ticker.any_of": "I:SPX,I:NDX,SPY,QQQ",
        "apiKey": api_key,
    }
    url = "https://api.polygon.io/v3/snapshot"
    timeout = httpx.Timeout(6.0, connect=3.0)
    started = time.perf_counter()
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.get(url, params=params)
            latency = (time.perf_counter() - started) * 1000.0
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            latency = (time.perf_counter() - started) * 1000.0
            status = FeedStatus(
                source="polygon",
                symbol="multi",
                status="failed",
                latency_ms=latency,
                error=f"http_{exc.response.status_code}",
                details={"status_code": exc.response.status_code},
            )
            _UNIVERSAL_CACHE[cache_key] = (now, {}, status)
            return None, status
        except httpx.HTTPError as exc:
            latency = (time.perf_counter() - started) * 1000.0
            status = FeedStatus(
                source="polygon",
                symbol="multi",
                status="failed",
                latency_ms=latency,
                error="network_error",
                details={"message": str(exc)},
            )
            _UNIVERSAL_CACHE[cache_key] = (now, {}, status)
            return None, status

    payload = resp.json()
    results = payload.get("results") or []
    latency = (time.perf_counter() - started) * 1000.0
    status = FeedStatus(
        source="polygon",
        symbol="multi",
        status="healthy" if results else "degraded",
        latency_ms=latency,
        details={"count": len(results)},
    )
    _UNIVERSAL_CACHE[cache_key] = (now, payload, status)
    return payload, status


__all__ = ["FeedStatus", "polygon_index_snapshot", "tradier_index_greeks", "polygon_universal_snapshot"]
