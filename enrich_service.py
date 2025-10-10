"""Finnhub-backed enrichment microservice.

This FastAPI app fetches sentiment, macro events, and earnings metadata for a
symbol and exposes them under a single `/enrich/{symbol}` endpoint. Responses
are cached briefly to avoid hitting rate limits.
"""

from __future__ import annotations

import datetime as dt
import os
import time
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

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


async def _get_json(
    client: httpx.AsyncClient, url: str, params: Dict[str, Any]
) -> Dict[str, Any]:
    """Fetch JSON with a small TTL cache."""

    cache_key = f"{url}|{sorted(params.items())}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    response = await client.get(url, params=params, timeout=15.0)
    if response.status_code != 200:
        raise HTTPException(
            status_code=502, detail=f"Upstream error {response.status_code}: {response.text[:200]}"
        )
    payload = response.json()
    cache.set(cache_key, payload)
    return payload


class SentimentBlock(BaseModel):
    symbol_sentiment: float
    news_count_24h: Optional[int] = None
    news_bias_24h: Optional[float] = None
    social_sentiment: Optional[float] = None
    headline_risk: Optional[str] = None


class EventsBlock(BaseModel):
    next_fomc_minutes: Optional[int] = None
    next_cpi_minutes: Optional[int] = None
    next_nfp_minutes: Optional[int] = None
    within_event_window: Optional[bool] = None
    label: Optional[str] = None


class EarningsBlock(BaseModel):
    next_earnings_at: Optional[str] = None
    dte_to_earnings: Optional[float] = None
    pre_or_post: Optional[str] = None
    earnings_flag: Optional[str] = None
    expected_move_pct: Optional[float] = None


class EnrichmentResponse(BaseModel):
    symbol: str
    sentiment: Optional[SentimentBlock] = None
    events: Optional[EventsBlock] = None
    earnings: Optional[EarningsBlock] = None


def _minutes_until(timestamp_utc: dt.datetime) -> int:
    now = dt.datetime.now(dt.timezone.utc)
    delta = timestamp_utc - now
    return max(0, int(delta.total_seconds() // 60))


def _iso_str(timestamp_utc: dt.datetime) -> str:
    return timestamp_utc.replace(microsecond=0).isoformat()


async def _build_sentiment(client: httpx.AsyncClient, symbol: str) -> SentimentBlock:
    params = {"symbol": symbol, "token": FINNHUB_API_KEY}
    data = await _get_json(client, f"{FINNHUB_BASE_URL}/news-sentiment", params)

    score = float(data.get("companyNewsScore") or 0.0)
    symbol_sentiment = (score - 0.5) * 2.0  # map 0..1 to -1..1

    buzz = data.get("buzz") or {}
    news_count = int(buzz.get("articlesInLastWeek") or 0)

    sentiment = data.get("sentiment") or {}
    bullish = float(sentiment.get("bullishPercent") or 0.5)
    news_bias = (bullish - 0.5) * 2.0

    headline_risk = "normal"
    if news_count > 50 or abs(symbol_sentiment) > 0.6:
        headline_risk = "elevated"

    return SentimentBlock(
        symbol_sentiment=round(symbol_sentiment, 3),
        news_count_24h=news_count,
        news_bias_24h=round(news_bias, 3),
        social_sentiment=None,
        headline_risk=headline_risk,
    )


async def _build_events(client: httpx.AsyncClient) -> EventsBlock:
    today = dt.date.today().isoformat()
    params = {"from": today, "to": today, "token": FINNHUB_API_KEY}
    data = await _get_json(client, f"{FINNHUB_BASE_URL}/calendar/economic", params)
    calendar = data.get("economicCalendar") or []

    def next_minutes(name_tokens: tuple[str, ...]) -> Optional[int]:
        future: list[int] = []
        for item in calendar:
            title = (item.get("event") or "").lower()
            if not any(token in title for token in name_tokens):
                continue
            date_str = item.get("date")
            time_str = item.get("time") or "00:00"
            if not date_str:
                continue
            try:
                timestamp = dt.datetime.fromisoformat(f"{date_str}T{time_str}:00+00:00")
            except ValueError:
                continue
            minutes = _minutes_until(timestamp)
            if minutes >= 0:
                future.append(minutes)
        return min(future) if future else None

    next_cpi = next_minutes(("cpi", "consumer price"))
    next_fomc = next_minutes(("fomc", "federal reserve", "interest rate decision"))
    next_nfp = next_minutes(("non-farm", "nonfarm", "payroll"))

    upcoming = [value for value in (next_cpi, next_fomc, next_nfp) if value is not None]
    within_window = any(value <= 90 for value in upcoming) if upcoming else False

    return EventsBlock(
        next_fomc_minutes=next_fomc,
        next_cpi_minutes=next_cpi,
        next_nfp_minutes=next_nfp,
        within_event_window=within_window,
        label="watch" if within_window else "none",
    )


async def _build_earnings(client: httpx.AsyncClient, symbol: str) -> EarningsBlock:
    today = dt.date.today()
    params = {
        "from": today.isoformat(),
        "to": (today + dt.timedelta(days=60)).isoformat(),
        "token": FINNHUB_API_KEY,
    }
    data = await _get_json(client, f"{FINNHUB_BASE_URL}/calendar/earnings", params)
    calendar = data.get("earningsCalendar") or []

    upcoming: Optional[Dict[str, Any]] = None
    for item in calendar:
        if (item.get("symbol") or "").upper() == symbol.upper():
            upcoming = item
            break

    if not upcoming:
        return EarningsBlock(pre_or_post="none", earnings_flag="none")

    date_str = upcoming.get("date") or today.isoformat()
    hour_str = upcoming.get("hour") or "20:00"
    try:
        timestamp = dt.datetime.fromisoformat(f"{date_str}T{hour_str}:00+00:00")
    except ValueError:
        timestamp = dt.datetime.fromisoformat(f"{today.isoformat()}T20:00:00+00:00")

    dte = (timestamp.date() - today).days
    flag = "none"
    if dte == 0:
        flag = "today"
    elif 1 <= dte <= 7:
        flag = "near"

    pre_or_post = "none"
    try:
        hour = int(hour_str.split(":")[0])
        if hour < 13:
            pre_or_post = "pre"
            if dte == 0:
                flag = "before_open"
        else:
            pre_or_post = "post"
            if dte == 0:
                flag = "after_close"
    except ValueError:
        pass

    return EarningsBlock(
        next_earnings_at=_iso_str(timestamp),
        dte_to_earnings=float(dte),
        pre_or_post=pre_or_post,
        earnings_flag=flag,
    )


@app.get("/enrich/{symbol}", response_model=EnrichmentResponse)
async def enrich(symbol: str) -> EnrichmentResponse:
    symbol = symbol.strip().upper()
    if not symbol:
        raise HTTPException(status_code=400, detail="Symbol is required")

    if not FINNHUB_API_KEY:
        raise HTTPException(status_code=503, detail="FINNHUB_API_KEY not configured")

    async with httpx.AsyncClient() as client:
        sentiment = await _build_sentiment(client, symbol)
        events = await _build_events(client)
        earnings = await _build_earnings(client, symbol)

    return EnrichmentResponse(symbol=symbol, sentiment=sentiment, events=events, earnings=earnings)


@app.get("/healthz")
async def healthcheck(ping: bool = Query(False, description="Ping Finnhub to verify reachability")) -> Dict[str, Any]:
    """Basic process health plus optional Finnhub connectivity check.

    - Always returns `status: ok` if the server is running.
    - Returns a `finnhub` block indicating whether the API key is configured and,
      if `ping=true`, whether a recent upstream call succeeded (cached briefly).
    """

    finnhub_info: Dict[str, Any] = {
        "configured": bool(FINNHUB_API_KEY),
        "ok": None,
        "last_checked": None,
        "message": None,
    }

    if FINNHUB_API_KEY and ping:
        cache_key = "health:finnhub"
        cached = cache.get(cache_key)
        if cached is not None:
            finnhub_info.update(cached)
        else:
            # Perform a lightweight upstream call and cache the result
            try:
                async with httpx.AsyncClient(timeout=8.0) as client:
                    # Use news-sentiment on a liquid symbol as a simple probe
                    params = {"symbol": "AAPL", "token": FINNHUB_API_KEY}
                    resp = await client.get(f"{FINNHUB_BASE_URL}/news-sentiment", params=params)
                    ok = resp.status_code == 200
                    finfo = {
                        "configured": True,
                        "ok": bool(ok),
                        "last_checked": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
                        "message": None if ok else f"status={resp.status_code}",
                    }
                    cache.set(cache_key, finfo)
                    finnhub_info.update(finfo)
            except Exception as exc:  # pragma: no cover
                finfo = {
                    "configured": True,
                    "ok": False,
                    "last_checked": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
                    "message": str(exc)[:200],
                }
                cache.set(cache_key, finfo)
                finnhub_info.update(finfo)

    return {
        "status": "ok",
        "finnhub": finnhub_info,
    }


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
