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
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

load_dotenv()

# Load lazily; don't crash import if missing. We'll 503 at request time.
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
TE_API_KEY = os.getenv("TE_API_KEY")

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
    logger.info(
        "Finnhub request %s url=%s params=%s",
        name,
        url,
        {k: v for k, v in sanitized.items() if k != "token"},
    )

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
        logger.warning(
            "Finnhub error %s status=%s body=%s",
            name,
            response.status_code,
            body_excerpt,
        )
        return {
            "data": None,
            "error": f"status {response.status_code}: {body_excerpt}",
            "status": response.status_code,
        }

    payload = response.json()
    result = {"data": payload, "error": None, "status": response.status_code}
    cache.set(cache_key, result)
    return result


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


async def _build_events(
    client: httpx.AsyncClient,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Build macro events summary using Trading Economics (primary) with Finnhub fallback."""

    summary: Dict[str, Any] = {
        "label": "none",
        "next_fomc_minutes": None,
        "next_cpi_minutes": None,
        "next_nfp_minutes": None,
        "source": "tradingeconomics",
    }
    meta: Dict[str, Any] = {
        "finnhub_status": None,
        "finnhub_error": None,
        "fallback_used": False,  # will now indicate Finnhub fallback
        "fallback_error": None,
    }

    today = dt.date.today()
    calendar: List[Dict[str, Any]] = []

    # ---- PRIMARY: Trading Economics (US, 14d) ----
    te_rows: List[Dict[str, Any]] = []
    if TE_API_KEY:
        te_url = "https://api.tradingeconomics.com/calendar"
        te_params = {
            "country": "united states",
            "from": today.isoformat(),
            "to": (today + dt.timedelta(days=14)).isoformat(),
            "c": TE_API_KEY,
        }
        try:
            resp = await client.get(te_url, params=te_params, timeout=8.0)
            resp.raise_for_status()
            te_rows = resp.json()
        except Exception as exc:  # pragma: no cover
            te_rows = []
            meta["fallback_error"] = str(exc)[:200]  # keep for visibility
        normalized: List[Dict[str, Any]] = []
        if isinstance(te_rows, list):
            for row in te_rows:
                if not isinstance(row, dict):
                    continue
                event_name = (row.get("Event") or "").strip()
                if not event_name:
                    continue
                when_raw = row.get("Date")
                date_token = None
                time_token = None
                if isinstance(when_raw, str) and "T" in when_raw:
                    date_token, _, remainder = when_raw.partition("T")
                    time_token = remainder[:5]
                normalized.append(
                    {
                        "event": event_name,
                        "country": row.get("Country"),
                        "datetime": when_raw,
                        "date": date_token,
                        "time": time_token,
                        "_source": "tradingeconomics",
                    }
                )
        if normalized:
            calendar = normalized
            summary["source"] = "tradingeconomics"
        else:
            meta["fallback_error"] = meta.get("fallback_error") or "te_empty_or_error"

    # ---- FALLBACK: Finnhub economic calendar (US, 14d) if TE failed/empty ----
    if not calendar and FINNHUB_API_KEY:
        fin_params = {
            "from": today.isoformat(),
            "to": (today + dt.timedelta(days=14)).isoformat(),
            "country": "US",
            "token": FINNHUB_API_KEY,
        }
        finnhub_result = await _fetch_finnhub_json(
            client,
            "events_economic",
            "/calendar/economic",
            fin_params,
        )
        meta["finnhub_status"] = finnhub_result.get("status")
        if finnhub_result.get("error"):
            meta["finnhub_error"] = finnhub_result["error"]
        data = finnhub_result.get("data")
        if isinstance(data, dict):
            calendar = data.get("economicCalendar") or []
        elif isinstance(data, list):
            calendar = data
        if calendar:
            summary["source"] = "finnhub"
            meta["fallback_used"] = True
        elif not meta.get("fallback_error"):
            meta["fallback_error"] = "finnhub_empty"

    def _event_timestamp(item: Dict[str, Any]) -> Optional[dt.datetime]:
        datetime_token = item.get("datetime")
        if isinstance(datetime_token, str):
            try:
                iso_token = datetime_token.replace("Z", "+00:00")
                if "+" not in iso_token[10:]:
                    iso_token = f"{iso_token}+00:00"
                return dt.datetime.fromisoformat(iso_token)
            except ValueError:
                pass
        date_str = item.get("date")
        time_str = item.get("time") or "00:00"
        if not date_str:
            return None
        if isinstance(time_str, str) and time_str.upper() in {"", "TBD", "N/A"}:
            time_str = "00:00"
        try:
            return dt.datetime.fromisoformat(f"{date_str}T{time_str}:00+00:00")
        except ValueError:
            return None

    def next_minutes(name_tokens: tuple[str, ...]) -> Optional[int]:
        now_dt = dt.datetime.now(dt.timezone.utc)
        future: List[int] = []
        for item in calendar:
            title = (item.get("event") or item.get("Event") or "").lower()
            if not any(token in title for token in name_tokens):
                continue
            timestamp = _event_timestamp(item)
            if not timestamp:
                continue
            minutes = int((timestamp - now_dt).total_seconds() // 60)
            if minutes >= -30:
                future.append(minutes if minutes >= 0 else 0)
        return min(future) if future else None

    next_cpi = next_minutes(("cpi", "consumer price"))
    next_fomc = next_minutes(
        (
            "fomc",
            "federal reserve",
            "fed",
            "interest rate decision",
            "policy",
            "statement",
            "press",
        )
    )
    next_nfp = next_minutes(("non-farm", "nonfarm", "payroll"))

    if next_fomc is not None:
        summary["label"] = "policy_watch"
    elif next_cpi is not None:
        summary["label"] = "inflation_watch"
    elif next_nfp is not None:
        summary["label"] = "labor_watch"

    summary["next_fomc_minutes"] = next_fomc
    summary["next_cpi_minutes"] = next_cpi
    summary["next_nfp_minutes"] = next_nfp

    return summary, meta


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
        "earnings": (
            "/stock/earnings",
            {"symbol": symbol_clean, "token": FINNHUB_API_KEY},
        ),
        "sentiment": (
            "/news-sentiment",
            {"symbol": symbol_clean, "token": FINNHUB_API_KEY},
        ),
    }

    async with httpx.AsyncClient(timeout=15.0) as client:
        names = list(endpoint_specs.keys())
        responses = await asyncio.gather(
            *[
                _fetch_finnhub_json(client, name, path, params)
                for name, (path, params) in endpoint_specs.items()
            ]
        )
        events_summary, events_meta = await _build_events(client)
    results = dict(zip(names, responses))

    earnings_result = results["earnings"]
    sentiment_result = results["sentiment"]

    earnings_data = _extract_earnings(earnings_result["data"])
    sentiment_data = _extract_sentiment(sentiment_result["data"])

    finnhub_ok_flags = [
        earnings_result.get("error") is None,
        sentiment_result.get("error") is None,
        events_meta.get("finnhub_error") is None,
    ]

    payload: Dict[str, Any] = {
        "symbol": symbol_clean,
        "earnings": earnings_data,
        "events": events_summary,
        "sentiment": sentiment_data,
        "finnhub_status": "ok" if any(finnhub_ok_flags) else "error",
        "fetched_at": _now_iso(),
    }

    if earnings_result.get("error"):
        payload["earnings_error"] = earnings_result["error"]
        payload["earnings_status"] = earnings_result.get("status")
    if sentiment_result.get("error"):
        payload["sentiment_error"] = sentiment_result["error"]
        payload["sentiment_status"] = sentiment_result.get("status")
    if events_meta.get("finnhub_error"):
        payload["events_error"] = events_meta["finnhub_error"]
    if events_meta.get("finnhub_status") is not None:
        payload["events_status"] = events_meta["finnhub_status"]
    if events_meta.get("fallback_error"):
        payload["events_fallback_error"] = events_meta["fallback_error"]
    payload["events_fallback_used"] = events_meta.get("fallback_used", False)

    finnhub_requests: Dict[str, Dict[str, Any]] = {}
    for name, res in results.items():
        entry: Dict[str, Any] = {}
        if res.get("status") is not None:
            entry["status"] = res["status"]
        if res.get("error"):
            entry["error"] = res["error"]
        finnhub_requests[name] = entry or {"status": None}
    event_entry: Dict[str, Any] = {}
    if events_meta.get("finnhub_status") is not None:
        event_entry["status"] = events_meta["finnhub_status"]
    if events_meta.get("finnhub_error"):
        event_entry["error"] = events_meta["finnhub_error"]
    if events_meta.get("fallback_used"):
        event_entry["fallback"] = "finnhub"
    finnhub_requests["events_macro"] = event_entry or {"status": None}
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
            logger.info(
                "Finnhub request healthz url=%s/quote params=%s",
                FINNHUB_BASE_URL,
                {"symbol": "SPY"},
            )
            resp = await client.get(f"{FINNHUB_BASE_URL}/quote", params=params)
            logger.info(
                "Finnhub response healthz status=%s bytes=%s",
                resp.status_code,
                len(resp.content),
            )
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
            raise HTTPException(
                status_code=resp.status_code, detail="finnhub quote error"
            )
        q = resp.json() or {}
        c, pc = q.get("c"), q.get("pc")
        pct = None
        if (
            isinstance(c, (int, float))
            and isinstance(pc, (int, float))
            and pc not in (0, None)
        ):
            try:
                pct = (c / pc) - 1.0
            except Exception:
                pct = None
        return {
            "symbol": symbol,
            "last": c,
            "percent": pct,
            "time_utc": dt.datetime.now(dt.timezone.utc)
            .replace(microsecond=0)
            .isoformat(),
            "stale": False,
        }
    except Exception:
        return {
            "symbol": symbol,
            "last": None,
            "percent": None,
            "time_utc": dt.datetime.now(dt.timezone.utc)
            .replace(microsecond=0)
            .isoformat(),
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
