"""Finnhub-backed enrichment microservice.

This FastAPI app fetches sentiment, macro events, and earnings metadata for a
symbol and exposes them under a single `/enrich/{symbol}` endpoint. Responses
are cached briefly to avoid hitting rate limits.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import os
import time
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

load_dotenv()

# Load lazily; don't crash import if missing. We'll 503 at request time.
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

FINNHUB_BASE_URL = "https://finnhub.io/api/v1"


class TTLCache:
    """Simple in-memory TTL cache for upstream responses."""

    def __init__(self, ttl_s: int = 300) -> None:
        self._ttl = ttl_s
        self._store: Dict[str, tuple[Any, float]] = {}

    def get(self, key: str) -> Any | None:
        cached = self._store.get(key)
        if not cached:
            return None
        value, expires_at = cached
        if time.time() > expires_at:
            self._store.pop(key, None)
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        self._store[key] = (value, time.time() + self._ttl)


cache = TTLCache(ttl_s=300)

app = FastAPI(title="Context Enrichment (Finnhub)")


logger = logging.getLogger("enrich_service")
if not logger.handlers:
    logger.addHandler(logging.NullHandler())


def _make_cache_key(name: str, params: Dict[str, Any]) -> str:
    filtered = tuple(sorted((k, v) for k, v in params.items() if k.lower() != "token"))
    return f"{name}|{filtered}"


async def _fetch_finnhub_json(
    client: httpx.AsyncClient,
    name: str,
    path: str,
    params: Dict[str, Any],
) -> Dict[str, Any]:
    """Fetch Finnhub JSON with logging and basic caching."""

    cache_key = _make_cache_key(name, params)
    cached = cache.get(cache_key)
    if cached is not None:
        logger.info("Finnhub cache hit %s status=%s", name, cached.get("status"))
        return cached

    sanitized = {k: ("***" if k.lower() == "token" else v) for k, v in params.items()}
    url = f"{FINNHUB_BASE_URL}{path}"
    logger.info("Finnhub request %s url=%s params=%s", name, url, {k: v for k, v in sanitized.items() if k != "token"})

    try:
        response = await client.get(url, params=params, timeout=15.0)
    except Exception as exc:
        logger.warning("Finnhub request failed %s error=%s", name, exc)
        return {"data": None, "error": str(exc)[:200], "status": None}

    logger.info(
        "Finnhub response %s status=%s bytes=%s",
        name,
        response.status_code,
        len(response.content),
    )

    if response.status_code != 200:
        body_excerpt = response.text[:200]
        logger.warning("Finnhub error %s status=%s body=%s", name, response.status_code, body_excerpt)
        return {
            "data": None,
            "error": f"status {response.status_code}: {body_excerpt}",
            "status": response.status_code,
        }

    payload = response.json()
    result = {"data": payload, "error": None, "status": response.status_code}
    cache.set(cache_key, result)
    return result


def _extract_events(data: Any) -> list[Dict[str, Any]]:
    if isinstance(data, dict):
        calendar = data.get("earningsCalendar") or data.get("data")
        if isinstance(calendar, list):
            return calendar
    if isinstance(data, list):
        return data
    return []


def _extract_earnings(data: Any) -> list[Dict[str, Any]]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # Some Finnhub responses nest the list under `data`
        entries = data.get("data") or data.get("earnings")
        if isinstance(entries, list):
            return entries
    return []


def _extract_sentiment(data: Any) -> Dict[str, Any]:
    if isinstance(data, dict):
        return data
    return {}


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


@app.get("/enrich/{symbol}")
async def enrich(symbol: str) -> Dict[str, Any]:
    symbol_clean = symbol.strip().upper()
    if not symbol_clean:
        raise HTTPException(status_code=400, detail="Symbol is required")

    if not FINNHUB_API_KEY:
        raise HTTPException(status_code=503, detail="FINNHUB_API_KEY not configured")

    endpoint_specs = {
        "earnings": ("/stock/earnings", {"symbol": symbol_clean, "token": FINNHUB_API_KEY}),
        "events": ("/calendar/earnings", {"symbol": symbol_clean, "token": FINNHUB_API_KEY}),
        "sentiment": ("/news-sentiment", {"symbol": symbol_clean, "token": FINNHUB_API_KEY}),
    }

    async with httpx.AsyncClient(timeout=15.0) as client:
        names = list(endpoint_specs.keys())
        responses = await asyncio.gather(
            *[
                _fetch_finnhub_json(client, name, path, params)
                for name, (path, params) in endpoint_specs.items()
            ]
        )
    results = dict(zip(names, responses))

    earnings_result = results["earnings"]
    events_result = results["events"]
    sentiment_result = results["sentiment"]

    earnings_data = _extract_earnings(earnings_result["data"])
    events_data = _extract_events(events_result["data"])
    sentiment_data = _extract_sentiment(sentiment_result["data"])

    payload: Dict[str, Any] = {
        "symbol": symbol_clean,
        "earnings": earnings_data,
        "events": events_data,
        "sentiment": sentiment_data,
        "finnhub_status": "ok" if any(res.get("error") is None for res in results.values()) else "error",
        "fetched_at": _now_iso(),
    }

    if earnings_result.get("error"):
        payload["earnings_error"] = earnings_result["error"]
        payload["earnings_status"] = earnings_result.get("status")
    if events_result.get("error"):
        payload["events_error"] = events_result["error"]
        payload["events_status"] = events_result.get("status")
    if sentiment_result.get("error"):
        payload["sentiment_error"] = sentiment_result["error"]
        payload["sentiment_status"] = sentiment_result.get("status")

    finnhub_requests: Dict[str, Dict[str, Any]] = {}
    for name, res in results.items():
        entry: Dict[str, Any] = {}
        if res.get("status") is not None:
            entry["status"] = res["status"]
        if res.get("error"):
            entry["error"] = res["error"]
        finnhub_requests[name] = entry or {"status": None}
    payload["finnhub_requests"] = finnhub_requests

    return payload


@app.get("/healthz")
async def healthcheck() -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "configured": bool(FINNHUB_API_KEY),
        "ok": False,
        "status": None,
        "checked_at": None,
    }

    if not FINNHUB_API_KEY:
        summary["error"] = "FINNHUB_API_KEY missing"
        return {"status": "ok", "finnhub": summary}

    cache_key = "health:quote:SPY"
    cached = cache.get(cache_key)
    if cached is not None:
        return {"status": "ok", "finnhub": dict(cached)}

    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            params = {"symbol": "SPY", "token": FINNHUB_API_KEY}
            logger.info("Finnhub request healthz url=%s/quote params=%s", FINNHUB_BASE_URL, {"symbol": "SPY"})
            resp = await client.get(f"{FINNHUB_BASE_URL}/quote", params=params)
            logger.info("Finnhub response healthz status=%s bytes=%s", resp.status_code, len(resp.content))
            summary["status"] = resp.status_code
            summary["ok"] = resp.status_code == 200
            summary["checked_at"] = _now_iso()
            if not summary["ok"]:
                summary["error"] = resp.text[:200]
    except Exception as exc:  # pragma: no cover
        summary["ok"] = False
        summary["status"] = None
        summary["checked_at"] = _now_iso()
        summary["error"] = str(exc)[:200]

    cache.set(cache_key, summary)
    return {"status": "ok", "finnhub": summary}


# ---------------------------------------------------------------------------
# Futures snapshot proxies (ETF + VIX) — /gpt/futures-snapshot
# ---------------------------------------------------------------------------

PROXY_MAP: Dict[str, str] = {
    "es_proxy": "SPY",
    "nq_proxy": "QQQ",
    "ym_proxy": "DIA",
    "rty_proxy": "IWM",
    "vix": "CBOE:VIX",
}


def _market_phase_chicago(now: Optional[dt.datetime] = None) -> str:
    tz = ZoneInfo("America/Chicago")
    dt_now = (now or dt.datetime.now(dt.timezone.utc)).astimezone(tz)
    if dt_now.weekday() >= 5:  # Sat/Sun
        return "closed"
    minutes = dt_now.hour * 60 + dt_now.minute
    reg_open, reg_close = 8 * 60 + 30, 15 * 60  # 08:30–15:00
    pre_open = 3 * 60  # 03:00
    after_close = 19 * 60  # 19:00
    if pre_open <= minutes < reg_open:
        return "premarket"
    if reg_open <= minutes < reg_close:
        return "regular"
    if reg_close <= minutes < after_close:
        return "afterhours"
    return "closed"


async def _fetch_quote_symbol(client: httpx.AsyncClient, symbol: str) -> Dict[str, Any]:
    try:
        resp = await client.get(
            f"{FINNHUB_BASE_URL}/quote",
            params={"symbol": symbol, "token": FINNHUB_API_KEY},
            timeout=8.0,
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail="finnhub quote error")
        q = resp.json() or {}
        c, pc = q.get("c"), q.get("pc")
        pct = None
        if isinstance(c, (int, float)) and isinstance(pc, (int, float)) and pc not in (0, None):
            try:
                pct = (c / pc) - 1.0
            except Exception:
                pct = None
        return {
            "symbol": symbol,
            "last": c,
            "percent": pct,
            "time_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
            "stale": False,
        }
    except Exception:
        return {
            "symbol": symbol,
            "last": None,
            "percent": None,
            "time_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
            "stale": True,
        }


_futures_cache: Dict[str, Any] = {"data": None, "ts": 0.0}


@app.get("/gpt/futures-snapshot")
async def futures_snapshot(resp: Dict[str, Any] = None) -> Dict[str, Any]:
    now_ts = time.time()
    # 3-minute cache
    cached = _futures_cache.get("data")
    ts = float(_futures_cache.get("ts") or 0)
    if cached and (now_ts - ts < 180):
        payload = dict(cached)
        payload["stale_seconds"] = int(now_ts - ts)
        return payload

    if not FINNHUB_API_KEY:
        # Mirror prior behavior: 503 when key missing
        return {
            "code": "UNAVAILABLE",
            "message": "FINNHUB_API_KEY missing",
        }

    out: Dict[str, Any] = {}
    async with httpx.AsyncClient(timeout=8.0) as client:
        for key, sym in PROXY_MAP.items():
            out[key] = await _fetch_quote_symbol(client, sym)

    out["market_phase"] = _market_phase_chicago()
    out["stale_seconds"] = 0
    _futures_cache["data"], _futures_cache["ts"] = out, now_ts
    return out
