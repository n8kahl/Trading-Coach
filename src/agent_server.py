"""Trading Coach backend tailored for GPT Actions integrations.

The service now focuses on a lean surface area that lets a custom GPT pull
ranked setups (with richer level-aware targets) and render interactive charts
driven by the same OHLCV data. Legacy endpoints for watchlists, notes, and
trade-following have been removed to keep the API aligned with the coaching
workflow.
"""

from __future__ import annotations

import asyncio
import json
import math
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, AliasChoices
from pydantic import ConfigDict
from urllib.parse import urlencode, quote
from fastapi.responses import StreamingResponse

from .config import get_settings
from .calculations import atr, ema, bollinger_bands, keltner_channels, adx, vwap
from .charts_api import router as charts_router, get_candles, normalize_interval
from .gpt_sentiment import router as gpt_sentiment_router
from .scanner import scan_market
from .tradier import (
    TradierNotConfiguredError,
    fetch_option_chain,
    fetch_option_chain_cached,
    fetch_option_quotes,
    select_tradier_contract,
)
from .polygon_options import fetch_polygon_option_chain, summarize_polygon_chain
from .context_overlays import compute_context_overlays


logger = logging.getLogger(__name__)

ALLOWED_CHART_KEYS = {
    "symbol",
    "interval",
    "view",
    "ema",
    "vwap",
    "range",
    "theme",
    "studies",
    "levels",
    "entry",
    "stop",
    "tp",
    "t1",
    "t2",
    "t3",
    "notes",
    "strategy",
    "direction",
    "atr",
    "title",
    "scale_plan",
    "supply",
    "demand",
    "liquidity",
    "fvg",
    "avwap",
}


DATA_SYMBOL_ALIASES: Dict[str, List[str]] = {
    "SPX": ["^GSPC"],
    "^SPX": ["^GSPC"],
    "INDEX:SPX": ["^GSPC"],
    "SP500": ["^GSPC"],
}


def _data_symbol_candidates(symbol: str) -> List[str]:
    primary = symbol or ""
    token = primary.upper()
    aliases = DATA_SYMBOL_ALIASES.get(token, [])
    if isinstance(aliases, str):
        aliases = [aliases]
    candidates: List[str] = [primary]
    for alias in aliases:
        if alias and alias not in candidates:
            candidates.append(alias)
    return candidates


def _normalize_chart_symbol(value: str) -> str:
    token = (value or "").strip()
    if ":" in token:
        return token
    return token.upper()


def _normalize_chart_interval(value: str) -> str:
    token = (value or "").strip().lower()
    if not token:
        return "1"
    if token.endswith("m"):
        return str(int(token.rstrip("m") or "1"))
    if token.endswith("h"):
        hours = int(token.rstrip("h") or "1")
        return str(hours * 60)
    if token in {"d", "1d"}:
        return "1D"
    return token.upper()


class ChartParams(BaseModel):
    symbol: str
    interval: str

    model_config = ConfigDict(extra="allow")


class ChartLinks(BaseModel):
    interactive: str


app = FastAPI(
    title="Trading Coach GPT Backend",
    description="Backend utilities for a custom GPT that offers trading guidance.",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


STATIC_ROOT = (Path(__file__).resolve().parent.parent / "static").resolve()
TV_STATIC_DIR = STATIC_ROOT / "tv"
if TV_STATIC_DIR.exists():
    app.mount("/tv", StaticFiles(directory=str(TV_STATIC_DIR), html=True), name="tv")


# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------

class AuthedUser(BaseModel):
    user_id: str


async def require_api_key(
    authorization: str | None = Header(default=None, alias="Authorization"),
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
) -> AuthedUser:
    """Optional API key check for GPT Actions.

    If `BACKEND_API_KEY` is set we enforce it. Otherwise the app falls back to
    a permissive mode that uses `X-User-Id` (or `anonymous`) to scope data.
    """

    settings = get_settings()
    expected = settings.backend_api_key

    if expected:
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing bearer token")
        token = authorization.split(" ", 1)[1]
        if token != expected:
            raise HTTPException(status_code=401, detail="Invalid bearer token")

    user_id = x_user_id or "anonymous"
    return AuthedUser(user_id=user_id)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ScanUniverse(BaseModel):
    tickers: List[str] = Field(..., description="Ticker symbols to analyse")
    style: str | None = Field(
        default=None,
        description="Optional style filter: 'scalp', 'intraday', 'swing', or 'leap'.",
    )


class ContractsRequest(BaseModel):
    symbol: str
    side: str | None = None
    style: str | None = None
    min_dte: int | None = None
    max_dte: int | None = None
    min_delta: float | None = None
    max_delta: float | None = None
    max_spread_pct: float | None = None
    min_oi: int | None = None
    max_price: float | None = None
    risk_amount: float | None = None
    expiry: str | None = None
    bias: str | None = None

    model_config = ConfigDict(extra="allow")


class PlanRequest(BaseModel):
    symbol: str
    style: str | None = None


class PlanResponse(BaseModel):
    plan_id: str | None = None
    version: int | None = None
    idea_url: str | None = None
    warnings: List[str] | None = None
    symbol: str
    style: str | None = None
    strategy_id: str | None = None
    description: str | None = None
    score: float | None = None
    plan: Dict[str, Any] | None = None
    charts: Dict[str, Any] | None = None
    key_levels: Dict[str, Any] | None = None
    market_snapshot: Dict[str, Any] | None = None
    features: Dict[str, Any] | None = None
    options: Dict[str, Any] | None = None
    calc_notes: Dict[str, Any] | None = None
    htf: Dict[str, Any] | None = None
    decimals: int | None = None
    data_quality: Dict[str, Any] | None = None
    debug: Dict[str, Any] | None = None


class IdeaStoreRequest(BaseModel):
    plan: Dict[str, Any]
    summary: Dict[str, Any] | None = None
    volatility_regime: Dict[str, Any] | None = None
    htf: Dict[str, Any] | None = None
    data_quality: Dict[str, Any] | None = None
    chart_url: str | None = None
    options: Dict[str, Any] | None = None
    why_this_works: List[str] | None = None
    invalidation: List[str] | None = None
    risk_note: str | None = None


class IdeaStoreResponse(BaseModel):
    plan_id: str
    idea_url: str


class StreamPushRequest(BaseModel):
    symbol: str
    event: Dict[str, Any]


class MultiContextRequest(BaseModel):
    symbol: str
    intervals: List[str] = Field(
        ..., validation_alias=AliasChoices("intervals", "frames")
    )
    lookback: int | None = None
    include_series: bool = False


class MultiContextResponse(BaseModel):
    symbol: str
    contexts: List[Dict[str, Any]]
    volatility_regime: Dict[str, Any]
    sentiment: Dict[str, Any] | None = None
    events: Dict[str, Any] | None = None
    earnings: Dict[str, Any] | None = None
    summary: Dict[str, Any] | None = None
    decimals: int | None = None
    data_quality: Dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


async def _fetch_context_enrichment(symbol: str) -> Dict[str, Any] | None:
    """Fetch sentiment/event/earnings enrichment data when the sidecar is configured."""

    try:
        settings = get_settings()
        base_url = getattr(settings, "enrichment_service_url", None) or ""
    except Exception:
        base_url = ""

    base_url = base_url.strip()
    if not base_url:
        return None

    url = f"{base_url.rstrip('/')}/enrich/{quote(symbol)}"
    timeout = httpx.Timeout(5.0, connect=2.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.get(url)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("context enrichment fetch failed for %s: %s", symbol, exc)
            return None
        except Exception as exc:  # pragma: no cover - network errors
            logger.warning("context enrichment fetch error for %s: %s", symbol, exc)
            return None

    try:
        return resp.json()
    except ValueError:
        logger.warning("context enrichment returned invalid JSON for %s", symbol)
        return None


def _style_for_strategy(strategy_id: str) -> str:
    sid = strategy_id.lower()
    if "power" in sid:
        return "scalp"
    if "gap" in sid:
        return "scalp"
    if "midday" in sid:
        return "intraday"
    if "pmcc" in sid or "leap" in sid:
        return "leap"
    if "orb" in sid:
        return "scalp"
    if "vwap" in sid or "inside" in sid:
        return "intraday"
    return "swing"


def _normalize_style(style: str | None) -> str | None:
    if style is None:
        return None
    normalized = style.strip().lower()
    if not normalized:
        return None
    if normalized == "leaps":
        normalized = "leap"
    if normalized in {"power_hour", "powerhour", "power-hour", "power hour"}:
        normalized = "scalp"
    return normalized


def _build_idea_url(base: str, plan_id: str, version: int) -> str:
    base_clean = base.rstrip("/")
    return f"{base_clean}/idea/{plan_id}?v={version}"


def _extract_plan_core(first: Dict[str, Any], plan_id: str, version: int, decimals: int | None) -> Dict[str, Any]:
    plan_block = first.get("plan") or {}
    charts = first.get("charts") or {}
    core = {
        "plan_id": plan_id,
        "version": version,
        "symbol": first.get("symbol"),
        "style": first.get("style"),
        "bias": plan_block.get("direction"),
        "setup": first.get("strategy_id"),
        "entry": plan_block.get("entry"),
        "stop": plan_block.get("stop"),
        "targets": plan_block.get("targets"),
        "rr_to_t1": plan_block.get("risk_reward"),
        "confidence": plan_block.get("confidence"),
        "decimals": decimals,
        "charts_params": charts.get("params") if isinstance(charts, dict) else None,
    }
    return core


def _build_snapshot_summary(first: Dict[str, Any]) -> Dict[str, Any]:
    features = first.get("features") or {}
    snapshot = first.get("market_snapshot") or {}
    volatility = snapshot.get("volatility") or {}
    trend = (snapshot.get("trend") or {}).get("ema_stack")
    summary = {
        "frames_used": [],
        "confluence_score": features.get("plan_confidence"),
        "trend_notes": {"primary": trend} if trend else {},
        "volatility_regime": volatility,
        "expected_move_horizon": volatility.get("expected_move_horizon"),
        "nearby_levels": list((first.get("key_levels") or {}).keys()),
    }
    return summary


async def _store_idea_snapshot(plan_id: str, snapshot: Dict[str, Any]) -> None:
    async with _IDEA_LOCK:
        versions = _IDEA_STORE.setdefault(plan_id, [])
        versions.append(snapshot)


async def _publish_stream_event(symbol: str, event: Dict[str, Any]) -> None:
    async with _STREAM_LOCK:
        queues = list(_STREAM_SUBSCRIBERS.get(symbol, []))
    payload = json.dumps({"symbol": symbol, "event": event})
    for queue in queues:
        try:
            queue.put_nowait(payload)
        except asyncio.QueueFull:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            queue.put_nowait(payload)


async def _get_idea_snapshot(plan_id: str, version: Optional[int] = None) -> Dict[str, Any]:
    async with _IDEA_LOCK:
        versions = _IDEA_STORE.get(plan_id)
        if not versions:
            raise HTTPException(status_code=404, detail="Plan not found")
        if version is None:
            return versions[-1]
        for snap in versions:
            plan = snap.get("plan") or {}
            if plan.get("version") == version:
                return snap
        raise HTTPException(status_code=404, detail="Plan version not found")


async def _stream_generator(symbol: str) -> Any:
    queue: asyncio.Queue[str] = asyncio.Queue(maxsize=100)
    async with _STREAM_LOCK:
        _STREAM_SUBSCRIBERS.setdefault(symbol, []).append(queue)
    try:
        while True:
            data = await queue.get()
            yield f"data: {data}\n\n"
    except asyncio.CancelledError:
        raise
    finally:
        async with _STREAM_LOCK:
            subscribers = _STREAM_SUBSCRIBERS.get(symbol, [])
            if queue in subscribers:
                subscribers.remove(queue)
            if not subscribers:
                _STREAM_SUBSCRIBERS.pop(symbol, None)


async def _simulate_generator(symbol: str, params: Dict[str, Any]) -> Any:
    lookback = max(int(params.get("minutes", 30)), 5)
    try:
        bars = get_candles(symbol, "1", lookback=lookback)
    except Exception as exc:
        yield f"data: {json.dumps({"error": str(exc)})}\n\n"
        return
    if bars.empty:
        yield f"data: {json.dumps({"error": "No data"})}\n\n"
        return
    entry = float(params.get("entry"))
    stop = float(params.get("stop"))
    tp1 = float(params.get("tp1"))
    tp2 = params.get("tp2")
    direction = params.get("direction", "long")
    state = "AWAIT_TRIGGER"
    for _, row in bars.iterrows():
        price = float(row["close"])
        event = {
            "type": "bar",
            "state": state,
            "price": price,
            "time": row["time"],
        }
        if state == "AWAIT_TRIGGER":
            if (direction == "long" and price >= entry) or (direction == "short" and price <= entry):
                state = "IN_TRADE"
                event["coaching"] = "Trigger crossed — manage fills"
        elif state == "IN_TRADE":
            if (direction == "long" and price >= tp1) or (direction == "short" and price <= tp1):
                state = "MANAGE"
                event["coaching"] = "TP1 hit — scale and trail to BE"
            elif (direction == "long" and price <= stop) or (direction == "short" and price >= stop):
                state = "EXITED"
                event["coaching"] = "Stopped out"
        elif state == "MANAGE":
            if tp2 and ((direction == "long" and price >= float(tp2)) or (direction == "short" and price <= float(tp2))):
                state = "EXITED"
                event["coaching"] = "TP2 hit — flat"
            elif (direction == "long" and price <= entry) or (direction == "short" and price >= entry):
                state = "EXITED"
                event["coaching"] = "Back to entry — flat"
        elif state == "EXITED":
            event["coaching"] = "Trade complete"
        yield f"data: {json.dumps(event)}\n\n"
        await asyncio.sleep(0.15)



async def _fetch_polygon_ohlcv(symbol: str, timeframe: str) -> pd.DataFrame | None:
    """Fetch OHLCV from Polygon if `POLYGON_API_KEY` is configured.

    timeframe: minutes string like '1','5','15','60' or 'D' for daily.
    """
    settings = get_settings()
    api_key = settings.polygon_api_key
    if not api_key:
        return None

    tf = (timeframe or "5").upper()
    if tf == "D":
        multiplier, timespan = 1, "day"
        days_back = 120
    else:
        try:
            minutes = int(tf)
        except ValueError:
            minutes = 5
        multiplier, timespan = minutes, "minute"
        total_minutes = max(minutes * 500, minutes * 5)
        days_back = max(math.ceil(total_minutes / (60 * 6)) + 2, 5)

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
        except httpx.HTTPError as exc:
            logger.warning("Polygon fetch failed for %s(%s %s): %s", symbol, multiplier, timespan, exc)
            return None

    data = resp.json()
    results = data.get("results")
    if not results:
        return None
    frame = pd.DataFrame(results)
    # Polygon keys: t (ms), o/h/l/c, v
    frame = frame[["t", "o", "h", "l", "c", "v"]].rename(
        columns={"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
    )
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True)
    frame = frame.set_index("timestamp").sort_index()
    return frame.dropna()


def _is_stale_frame(frame: pd.DataFrame, timeframe: str) -> bool:
    if frame.empty:
        return True
    last_ts = frame.index[-1]
    if last_ts.tzinfo is None:
        last_ts = last_ts.tz_localize("UTC")
    else:
        last_ts = last_ts.tz_convert("UTC")
    now = pd.Timestamp.utcnow()
    age = now - last_ts
    tf = (timeframe or "5").lower()
    if tf.isdigit():
        return age > pd.Timedelta(hours=36)
    if tf in {"d", "1d", "day"}:
        return age > pd.Timedelta(days=10)
    return age > pd.Timedelta(days=10)


async def _load_remote_ohlcv(symbol: str, timeframe: str) -> pd.DataFrame | None:
    """Fetch recent OHLCV, preferring Polygon when available, else Yahoo Finance."""
    candidates = _data_symbol_candidates(symbol)

    # Try Polygon first for each candidate symbol
    fresh_polygon: pd.DataFrame | None = None
    stale_polygon: pd.DataFrame | None = None
    for candidate in candidates:
        poly = await _fetch_polygon_ohlcv(candidate, timeframe)
        if poly is None or poly.empty:
            continue
        if not _is_stale_frame(poly, timeframe):
            fresh_polygon = poly
            break
        stale_polygon = poly if stale_polygon is None else stale_polygon

    if fresh_polygon is not None:
        return fresh_polygon

    if stale_polygon is not None:
        logger.warning("Polygon data is stale for %s; attempting Yahoo fallback.", symbol)

    tf = timeframe or "5"
    interval_map = {
        "1": ("1d", "1m"),
        "3": ("5d", "2m"),
        "5": ("5d", "5m"),
        "15": ("1mo", "15m"),
        "30": ("1mo", "30m"),
        "60": ("6mo", "60m"),
        "120": ("1y", "2h"),
        "240": ("2y", "4h"),
        "D": ("1y", "1d"),
    }
    range_span, interval = interval_map.get(tf, ("5d", "5m"))
    timeout = httpx.Timeout(6.0, connect=3.0)
    for candidate in candidates:
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{candidate}"
        params = {"interval": interval, "range": range_span, "includePrePost": "false"}

        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                response = await client.get(url, params=params)
                response.raise_for_status()
            except httpx.HTTPError as exc:
                logger.warning("Yahoo Finance fetch failed for %s: %s", candidate, exc)
                continue

        payload = response.json()
        try:
            chart = payload["chart"]
            if chart.get("error"):
                raise ValueError(chart["error"])
            result = chart["result"][0]
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            logger.warning("Unexpected Yahoo Finance payload for %s: %s", candidate, exc)
            continue

        timestamps = result.get("timestamp")
        if not timestamps:
            logger.warning("Yahoo Finance returned no timestamps for %s", candidate)
            continue

        quote = result["indicators"]["quote"][0]
        o = quote.get("open")
        h = quote.get("high")
        l = quote.get("low")
        c = quote.get("close")
        v = quote.get("volume")
        if not all([o, h, l, c, v]):
            logger.warning("Incomplete OHLCV data for %s", candidate)
            continue

        frame = pd.DataFrame(
            {
                "timestamp": [int(ts) for ts in timestamps],
                "open": o,
                "high": h,
                "low": l,
                "close": c,
                "volume": v,
            }
        ).dropna()

        frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="s", utc=True)
        frame = frame.set_index("timestamp")
        if not frame.empty:
            return frame

    if stale_polygon is not None:
        return stale_polygon

    return None


async def _collect_market_data(tickers: List[str], timeframe: str = "5") -> Dict[str, pd.DataFrame]:
    """Fetch OHLCV for a list of tickers from Polygon or Yahoo Finance."""
    tasks = [_load_remote_ohlcv(ticker, timeframe) for ticker in tickers]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    market_data: Dict[str, pd.DataFrame] = {}

    for ticker, result in zip(tickers, results):
        frame: pd.DataFrame | None = None
        if isinstance(result, Exception):
            logger.warning("Data fetch raised for %s: %s", ticker, result)
        elif isinstance(result, pd.DataFrame) and not result.empty:
            frame = result
        if frame is None:
            logger.warning("No market data available for %s", ticker)
            continue
        market_data[ticker] = frame

    return market_data


def _resample_ohlcv(frame: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Downsample OHLCV data to the requested timeframe (in minutes)."""
    if frame.empty:
        return frame

    tf = (timeframe or "5").lower()
    if tf.isdigit():
        minutes = int(tf)
        if minutes <= 1:
            return frame
        rule = f"{minutes}T"
    elif tf in {"d", "1d", "day"}:
        rule = "1D"
    else:
        return frame

    if not isinstance(frame.index, pd.DatetimeIndex):
        frame = frame.copy()
        frame.index = pd.to_datetime(frame.index)

    resampled = (
        frame.resample(rule)
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna()
    )
    return resampled


def _extract_key_levels(history: pd.DataFrame) -> Dict[str, float]:
    """Derive intraday and higher-timeframe reference levels."""
    if history.empty:
        return {}

    df = history.sort_index()
    today = df.index[-1].date()
    session_df = df[df.index.date == today]
    prev_session_df: pd.DataFrame | None = None
    session_dates = list(dict.fromkeys(df.index.date))
    if len(session_dates) >= 2:
        prev_date = session_dates[-2]
        prev_session_df = df[df.index.date == prev_date]

    opening_slice = session_df.head(min(len(session_df), 3)) if not session_df.empty else df.head(min(len(df), 3))
    prev_row = df.iloc[-2] if len(df) > 1 else df.iloc[-1]
    today_open = float(session_df["open"].iloc[0]) if not session_df.empty else float(df["open"].iloc[0])

    levels: Dict[str, float | None] = {
        "session_high": float(session_df["high"].max()) if not session_df.empty else float(df["high"].max()),
        "session_low": float(session_df["low"].min()) if not session_df.empty else float(df["low"].min()),
        "opening_range_high": float(opening_slice["high"].max()) if not opening_slice.empty else float(df["high"].iloc[0]),
        "opening_range_low": float(opening_slice["low"].min()) if not opening_slice.empty else float(df["low"].iloc[0]),
        "prev_close": float(prev_session_df["close"].iloc[-1]) if prev_session_df is not None and not prev_session_df.empty else float(prev_row["close"]),
        "prev_high": float(prev_session_df["high"].max()) if prev_session_df is not None and not prev_session_df.empty else float(prev_row["high"]),
        "prev_low": float(prev_session_df["low"].min()) if prev_session_df is not None and not prev_session_df.empty else float(prev_row["low"]),
        "today_open": today_open,
    }
    if prev_session_df is not None and not prev_session_df.empty:
        gap_fill_level = levels["prev_close"]
        if gap_fill_level and abs(today_open - gap_fill_level) >= max(0.1, gap_fill_level * 0.001):
            levels["gap_fill"] = gap_fill_level
        else:
            levels["gap_fill"] = None
    else:
        levels["gap_fill"] = None

    return {key: round(val, 2) for key, val in levels.items() if val is not None and np.isfinite(val)}

def _infer_bar_interval(history: pd.DataFrame) -> int:
    """Return approximate bar interval in minutes based on timestamp spacing."""
    idx = history.index
    if len(idx) < 2:
        return 1
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    deltas = idx.to_series().diff().dropna()
    if deltas.empty:
        return 1
    median_seconds = deltas.dt.total_seconds().median()
    if not np.isfinite(median_seconds) or median_seconds <= 0:
        return 1
    return max(1, int(round(median_seconds / 60.0)))


def _session_phase(ts: pd.Timestamp) -> str:
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    et = ts.tz_convert("America/New_York")
    h, m = et.hour, et.minute
    wd = et.weekday()
    if wd >= 5:
        return "off"
    if h < 9 or (h == 9 and m < 30):
        return "premarket"
    if h == 9 and 30 <= m < 60:
        return "open_drive"
    if h == 10 or (h == 11 and m < 30):
        return "morning"
    if (h == 11 and m >= 30) or (12 <= h < 14):
        return "midday"
    if h == 14:
        return "afternoon"
    if h == 15:
        return "power_hour"
    if h >= 16:
        return "postmarket"
    return "other"


def _minutes_until_close(ts: pd.Timestamp) -> int:
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    et = ts.tz_convert("America/New_York")
    close = et.replace(hour=16, minute=0, second=0, microsecond=0)
    delta = close - et
    minutes = int(delta.total_seconds() // 60)
    return max(minutes, 0)


def _safe_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return float(value)
    if isinstance(value, (np.floating, np.integer)):
        val = float(value)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    return None


def _build_market_snapshot(history: pd.DataFrame, key_levels: Dict[str, float]) -> Dict[str, Any]:
    df = history.sort_index().tail(600)
    latest = df.iloc[-1]
    ts = df.index[-1]
    if ts.tzinfo is None:
        ts_utc = ts.tz_localize("UTC")
    else:
        ts_utc = ts.tz_convert("UTC")

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"] if "volume" in df.columns else pd.Series(dtype=float)

    atr_series = atr(high, low, close, period=14)
    atr_value = float(atr_series.iloc[-1]) if not atr_series.empty else float("nan")

    ema9_val = ema(close, 9).iloc[-1] if len(close) >= 9 else float("nan")
    ema20_val = ema(close, 20).iloc[-1] if len(close) >= 20 else float("nan")
    ema50_val = ema(close, 50).iloc[-1] if len(close) >= 50 else float("nan")
    adx14_series = adx(high, low, close, 14)
    adx14_val = float(adx14_series.iloc[-1]) if not adx14_series.empty else float("nan")
    vwap_series = vwap(close, volume) if not volume.empty else pd.Series(dtype=float)
    vwap_val = float(vwap_series.iloc[-1]) if not vwap_series.empty else float("nan")

    bb_upper, bb_lower = bollinger_bands(close, period=20, width=2.0)
    kc_upper, kc_lower = keltner_channels(close, high, low, period=20, atr_factor=1.5)
    bb_width = None
    kc_width = None
    in_squeeze = None
    if not bb_upper.empty and not bb_lower.empty:
        upper = float(bb_upper.iloc[-1])
        lower = float(bb_lower.iloc[-1])
        if np.isfinite(upper) and np.isfinite(lower):
            bb_width = upper - lower
    if not kc_upper.empty and not kc_lower.empty:
        upper = float(kc_upper.iloc[-1])
        lower = float(kc_lower.iloc[-1])
        if np.isfinite(upper) and np.isfinite(lower):
            kc_width = upper - lower
    if bb_width is not None and kc_width is not None and np.isfinite(bb_width) and np.isfinite(kc_width):
        in_squeeze = bb_width < kc_width

    prev_close_series = close.shift(1)
    true_range = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close_series).abs(),
            (low - prev_close_series).abs(),
        ],
        axis=1,
    ).max(axis=1)
    tr_median = float(true_range.tail(20).median()) if not true_range.empty else float("nan")

    bar_interval = _infer_bar_interval(df)
    horizon_minutes = 30 if bar_interval <= 2 else 60
    horizon_bars = max(1, int(horizon_minutes / max(bar_interval, 1)))
    expected_move = None
    if np.isfinite(tr_median):
        expected_move = tr_median * horizon_bars

    prev_close_level = key_levels.get("prev_close")
    gap_points = None
    gap_percent = None
    gap_direction = None
    if prev_close_level:
        gap_points = float(latest["close"]) - float(prev_close_level)
        if prev_close_level:
            gap_percent = (gap_points / float(prev_close_level)) * 100.0
        if gap_points > 0:
            gap_direction = "up"
        elif gap_points < 0:
            gap_direction = "down"
        else:
            gap_direction = "flat"

    if np.isfinite(ema9_val) and np.isfinite(ema20_val) and np.isfinite(ema50_val):
        if ema9_val > ema20_val > ema50_val:
            ema_stack = "bullish"
        elif ema9_val < ema20_val < ema50_val:
            ema_stack = "bearish"
        else:
            ema_stack = "mixed"
    else:
        ema_stack = "unknown"

    session_phase = _session_phase(ts)
    minutes_to_close = _minutes_until_close(ts)

    recent_closes = [float(val) for val in close.tail(10).tolist()]
    recent_returns = []
    if len(recent_closes) >= 2:
        recent_returns = [round(recent_closes[i] - recent_closes[i - 1], 4) for i in range(1, len(recent_closes))]

    snapshot = {
        "timestamp_utc": ts_utc.isoformat(),
        "price": {
            "open": float(latest["open"]),
            "high": float(latest["high"]),
            "low": float(latest["low"]),
            "close": float(latest["close"]),
            "volume": float(latest.get("volume", 0.0)),
        },
        "indicators": {
            "ema9": _safe_number(ema9_val),
            "ema20": _safe_number(ema20_val),
            "ema50": _safe_number(ema50_val),
            "vwap": _safe_number(vwap_val),
            "atr14": _safe_number(atr_value),
            "adx14": _safe_number(adx14_val),
        },
        "volatility": {
            "true_range_median": _safe_number(tr_median),
            "bollinger_width": _safe_number(bb_width),
            "keltner_width": _safe_number(kc_width),
            "in_squeeze": in_squeeze,
            "expected_move_horizon": _safe_number(expected_move),
        },
        "levels": key_levels,
        "session": {
            "phase": session_phase,
            "minutes_to_close": minutes_to_close,
            "bar_interval_minutes": bar_interval,
        },
        "trend": {
            "ema_stack": ema_stack,
            "direction_hint": None,
        },
        "gap": {
            "points": _safe_number(gap_points),
            "percent": _safe_number(gap_percent),
            "direction": gap_direction,
        },
        "recent": {
            "closes": recent_closes,
            "close_deltas": recent_returns,
        },
    }
    return snapshot


def _serialize_features(features: Dict[str, Any]) -> Dict[str, Any]:
    serialized: Dict[str, Any] = {}
    for key, value in features.items():
        if isinstance(value, (np.floating, np.integer)):
            serialized[key] = float(value)
        elif isinstance(value, (list, tuple)):
            flattened: List[Any] = []
            numeric = True
            for item in value:
                if isinstance(item, (np.floating, np.integer, float, int)):
                    flattened.append(float(item))
                else:
                    numeric = False
                    break
            if numeric:
                serialized[key] = flattened
            else:
                serialized[key] = list(value)
        elif isinstance(value, (float, int, str, bool)) or value is None:
            serialized[key] = value
        else:
            try:
                serialized[key] = float(value)
            except (TypeError, ValueError):
                serialized[key] = str(value)
    return serialized


def _series_points(series: pd.Series, limit: int = 200) -> List[Dict[str, Any]]:
    points: List[Dict[str, Any]] = []
    if series is None:
        return points
    tail = series.dropna().tail(limit)
    for ts, val in tail.items():
        stamp = pd.Timestamp(ts)
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize("UTC")
        else:
            stamp = stamp.tz_convert("UTC")
        points.append({"time": stamp.isoformat(), "value": float(val)})
    return points
# Strategy utilities ---------------------------------------------------------

def _direction_for_strategy(strategy_id: str) -> str:
    sid = strategy_id.lower()
    if "short" in sid or "put" in sid:
        return "short"
    return "long"


def _indicators_for_strategy(strategy_id: str) -> List[str]:
    sid = strategy_id.lower()
    if "vwap" in sid:
        return ["VWAP", "EMA9", "EMA20"]
    if "orb" in sid:
        return ["VWAP", "EMA9", "EMA20"]
    if "power" in sid:
        return ["VWAP", "EMA9", "EMA20", "EMA50"]
    if "gap" in sid:
        return ["VWAP", "EMA9", "EMA20"]
    if "midday" in sid:
        return ["VWAP", "EMA9", "EMA20"]
    if "adx" in sid:
        return ["VWAP", "ADX"]
    return ["VWAP"]


def _timeframe_for_style(style: str | None) -> str:
    normalized = _normalize_style(style) or ""
    mapping = {"scalp": "1", "intraday": "5", "swing": "60", "leap": "1D"}
    return mapping.get(normalized, "5")


def _view_for_style(style: str | None) -> str:
    normalized = _normalize_style(style) or ""
    mapping = {"scalp": "30m", "intraday": "1d", "swing": "5d", "leap": "fit"}
    return mapping.get(normalized, "fit")


TV_SUPPORTED_RESOLUTIONS = ["1", "3", "5", "15", "30", "60", "120", "240", "1D"]


def _resolution_to_timeframe(resolution: str) -> str | None:
    token = (resolution or "").strip().upper()
    if not token:
        return None
    if token.endswith("M") and token[:-1].isdigit():
        return token[:-1]
    if token.endswith("H") and token[:-1].isdigit():
        try:
            return str(int(token[:-1]) * 60)
        except Exception:
            return None
    if token.endswith("D"):
        return "D"
    if token.isdigit():
        return token
    return None


def _resolution_to_minutes(resolution: str) -> int:
    token = (resolution or "").strip().upper()
    if token.endswith("D"):
        days = int("".join(ch for ch in token if ch.isdigit()) or "1")
        return days * 24 * 60
    if token.isdigit():
        return int(token)
    return 1


def _price_scale_for(price: float | None) -> int:
    if price is None or not math.isfinite(price) or price <= 0:
        return 100
    text = f"{price:.6f}".rstrip("0")
    if "." in text:
        decimals = len(text.split(".")[1])
    else:
        decimals = 0
    decimals = max(0, min(decimals, 6))
    return int(10 ** decimals)


def _extract_levels_for_chart(key_levels: Dict[str, float]) -> List[str]:
    order = [
        "session_high",
        "session_low",
        "opening_range_high",
        "opening_range_low",
        "prev_high",
        "prev_low",
        "prev_close",
        "gap_fill",
    ]
    levels: List[str] = []
    for key in order:
        value = key_levels.get(key)
        if value is None:
            continue
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            levels.append(f"{float(value):.2f}")
    return levels


def _float_to_token(value: float | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return f"{float(value):.2f}"
    return None


def _encode_overlay_params(overlays: Dict[str, Any]) -> Dict[str, str]:
    payload: Dict[str, str] = {}
    supply = overlays.get("supply_zones") or []
    demand = overlays.get("demand_zones") or []
    liquidity = overlays.get("liquidity_pools") or []
    fvgs = overlays.get("fvg") or []
    avwap_bundle = overlays.get("avwap") or {}

    def _format_zone(entry: Dict[str, Any]) -> str | None:
        low_token = _float_to_token(entry.get("low"))
        high_token = _float_to_token(entry.get("high"))
        timeframe = entry.get("timeframe") or ""
        strength = entry.get("strength") or ""
        if not low_token or not high_token:
            return None
        label = timeframe.replace(";", "").replace("@", "").replace("|", "").replace(",", "")
        strength_label = strength.replace(";", "").replace("@", "").replace("|", "").replace(",", "")
        return f"{label}@{low_token}-{high_token}@{strength_label}".strip("@")

    def _format_liquidity(entry: Dict[str, Any]) -> str | None:
        level_token = _float_to_token(entry.get("level"))
        if not level_token:
            return None
        label = entry.get("type") or ""
        label = label.replace(";", "").replace("@", "").replace("|", "").replace(",", "")
        tf = entry.get("timeframe") or ""
        tf = tf.replace(";", "").replace("@", "").replace("|", "").replace(",", "")
        density = entry.get("density")
        density_token = ""
        if isinstance(density, (int, float)) and math.isfinite(float(density)):
            density_token = f"{float(density):.2f}"
        pieces = [label, level_token]
        if tf:
            pieces.append(tf)
        if density_token:
            pieces.append(density_token)
        return "@".join(pieces)

    def _format_fvg(entry: Dict[str, Any]) -> str | None:
        low_token = _float_to_token(entry.get("low"))
        high_token = _float_to_token(entry.get("high"))
        if not low_token or not high_token:
            return None
        timeframe = entry.get("timeframe") or ""
        age = entry.get("age")
        age_token = ""
        if isinstance(age, (int, float)) and math.isfinite(float(age)):
            age_token = str(int(age))
        tf_clean = timeframe.replace(";", "").replace("@", "").replace("|", "").replace(",", "")
        pieces = [low_token, high_token]
        if tf_clean:
            pieces.append(tf_clean)
        if age_token:
            pieces.append(age_token)
        return "@".join(pieces)

    def _format_avwap(label: str, value: Any) -> str | None:
        token = _float_to_token(value if isinstance(value, (int, float)) else None)
        if not token:
            return None
        clean_label = label.replace(";", "").replace("@", "").replace("|", "").replace(",", "")
        return f"{clean_label}@{token}"

    supply_tokens = [token for token in (_format_zone(item) for item in supply[:6]) if token]
    demand_tokens = [token for token in (_format_zone(item) for item in demand[:6]) if token]
    liquidity_tokens = [token for token in (_format_liquidity(item) for item in liquidity[:8]) if token]
    fvg_tokens = [token for token in (_format_fvg(item) for item in fvgs[:6]) if token]
    avwap_tokens = [token for token in (_format_avwap(label, value) for label, value in avwap_bundle.items()) if token]

    if supply_tokens:
        payload["supply"] = ";".join(supply_tokens)
    if demand_tokens:
        payload["demand"] = ";".join(demand_tokens)
    if liquidity_tokens:
        payload["liquidity"] = ";".join(liquidity_tokens)
    if fvg_tokens:
        payload["fvg"] = ";".join(fvg_tokens)
    if avwap_tokens:
        payload["avwap"] = ";".join(avwap_tokens)
    return payload


CONTRACT_STYLE_DEFAULTS: Dict[str, Dict[str, float | int]] = {
    "scalp": {"min_dte": 0, "max_dte": 2, "min_delta": 0.55, "max_delta": 0.65, "max_spread_pct": 8.0, "min_oi": 500},
    "intraday": {"min_dte": 1, "max_dte": 5, "min_delta": 0.45, "max_delta": 0.55, "max_spread_pct": 10.0, "min_oi": 500},
    "swing": {"min_dte": 7, "max_dte": 45, "min_delta": 0.30, "max_delta": 0.55, "max_spread_pct": 12.0, "min_oi": 500},
    "leaps": {"min_dte": 180, "max_dte": 1200, "min_delta": 0.25, "max_delta": 0.45, "max_spread_pct": 12.0, "min_oi": 500},
}


CONTRACT_STYLE_TARGET_DELTA: Dict[str, float] = {
    "scalp": 0.60,
    "intraday": 0.50,
    "swing": 0.40,
    "leaps": 0.35,
}


def _normalize_contract_style(style: str | None) -> str:
    token = (style or "").strip().lower()
    if token in CONTRACT_STYLE_DEFAULTS:
        return token
    if token in {"leap", "leaps"}:
        return "leaps"
    if token in {"swing", "swingtrade", "swing_trade"}:
        return "swing"
    if token in {"scalp", "0dte", "short"}:
        return "scalp"
    return "intraday"


def _style_default_bounds(style: str) -> Dict[str, float | int]:
    return dict(CONTRACT_STYLE_DEFAULTS.get(style, CONTRACT_STYLE_DEFAULTS["intraday"]))


def _style_target_delta(style: str) -> float:
    return CONTRACT_STYLE_TARGET_DELTA.get(style, 0.50)


def _format_strike(strike: Any) -> str:
    try:
        value = float(strike)
    except (TypeError, ValueError):
        return str(strike)
    text = f"{value:.2f}"
    text = text.rstrip("0").rstrip(".")
    return text or f"{value:.0f}"


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _norm(value: float, lower: float, upper: float) -> float:
    if lower == upper:
        return 0.0
    span = float(upper - lower)
    if span <= 0:
        return 0.0
    ratio = (value - lower) / span
    return _clamp(ratio, 0.0, 1.0)


def _tradeability_score(
    *,
    spread_pct: float,
    delta: float,
    style: str,
    oi: float,
    iv_rank: float | None,
    theta: float | None,
) -> float:
    target_delta = _style_target_delta(style)
    delta_gap = min(abs(abs(delta) - target_delta), 0.5)
    spread_score = 1.0 - _norm(spread_pct, 2.0, 12.0)
    delta_score = 1.0 - delta_gap / 0.5
    oi_score = _norm(math.log10(max(oi, 1.0)), 2.0, 4.0)
    vr = iv_rank if iv_rank is not None and math.isfinite(iv_rank) else 55.0
    vol_score = 1.0 - _norm(vr, 40.0, 70.0)
    base = 0.4 * spread_score + 0.3 * delta_score + 0.2 * oi_score + 0.1 * vol_score
    score = max(0.0, min(1.0, base))
    if style == "leaps" and theta is not None and math.isfinite(theta) and theta > -0.05:
        score = min(1.0, score + 0.05)
    return round(score * 100.0, 1)


def _compute_price(bid: float | None, ask: float | None, last: float | None, fallback: float | None) -> float | None:
    mid_value: float | None = None
    if isinstance(bid, (int, float)) and isinstance(ask, (int, float)):
        bid_val = float(bid)
        ask_val = float(ask)
        if bid_val >= 0 and ask_val >= 0 and ask_val >= bid_val:
            mid_value = (bid_val + ask_val) / 2.0
    for candidate in (mid_value, last, fallback, bid, ask):
        if isinstance(candidate, (int, float)):
            value = float(candidate)
            if math.isfinite(value) and value > 0:
                return value
    return None


def _compute_spread_pct(bid: float | None, ask: float | None, price: float | None) -> float | None:
    if not isinstance(bid, (int, float)) or not isinstance(ask, (int, float)):
        return None
    bid_val = float(bid)
    ask_val = float(ask)
    if bid_val <= 0 or ask_val <= 0 or ask_val < bid_val:
        return None
    basis = price if isinstance(price, (int, float)) and price > 0 else (ask_val + bid_val) / 2.0
    if basis <= 0:
        return None
    return (ask_val - bid_val) / basis * 100.0


def _contract_label(symbol: str, expiry: str | None, strike: Any, option_type: str | None) -> str:
    strike_text = _format_strike(strike)
    suffix = option_type[:1].upper() if option_type else ""
    components = [symbol.upper()]
    if expiry:
        components.append(expiry)
    components.append(f"{strike_text}{suffix}")
    return " ".join(components)


def _percentile(values: np.ndarray, target: float) -> float | None:
    if values.size == 0 or target is None or not math.isfinite(target):
        return None
    sorted_vals = np.sort(values)
    if sorted_vals.size == 0:
        return None
    rank = np.searchsorted(sorted_vals, target, side="right")
    percentile = (rank / sorted_vals.size) * 100.0
    return round(float(percentile), 2)


_MULTI_CONTEXT_CACHE_TTL = 30.0
_MULTI_CONTEXT_CACHE: Dict[Tuple[str, str, int], Tuple[float, Dict[str, Any]]] = {}

# Idea snapshot store (in-memory)
_IDEA_STORE: Dict[str, List[Dict[str, Any]]] = {}
_IDEA_LOCK = asyncio.Lock()
_STREAM_SUBSCRIBERS: Dict[str, List[asyncio.Queue]] = {}
_STREAM_LOCK = asyncio.Lock()
_IV_METRICS_CACHE_TTL = 120.0
_IV_METRICS_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}


async def _compute_iv_metrics(symbol: str) -> Dict[str, Any]:
    key = symbol.upper()
    now = time.monotonic()
    cached = _IV_METRICS_CACHE.get(key)
    if cached and now - cached[0] < _IV_METRICS_CACHE_TTL:
        return dict(cached[1])

    metrics: Dict[str, Any] = {
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "iv_atm": None,
        "iv_rank": None,
        "iv_percentile": None,
        "hv_20": None,
        "hv_60": None,
        "hv_120": None,
        "hv_20_percentile": None,
        "iv_to_hv_ratio": None,
        "skew_25d": None,
    }

    daily_history = await _load_remote_ohlcv(symbol, "D")
    hv_series = None
    if daily_history is not None and not daily_history.empty:
        daily = daily_history.sort_index()
        closes = daily["close"].astype(float)
        returns = closes.pct_change().dropna()
        if not returns.empty:
            def _hv(window: int) -> float | None:
                if len(returns) < window:
                    return None
                vol = returns.tail(window).std(ddof=0)
                if vol is None or not math.isfinite(vol):
                    return None
                return float(vol * math.sqrt(252.0) * 100.0)

            hv20 = _hv(20)
            hv60 = _hv(60)
            hv120 = _hv(120)
            metrics.update({
                "hv_20": hv20,
                "hv_60": hv60,
                "hv_120": hv120,
            })

            rolling = returns.rolling(20).std(ddof=0) * math.sqrt(252.0) * 100.0
            hv_series = rolling.dropna().to_numpy(dtype=float)
            if hv20 is not None and hv_series.size:
                metrics["hv_20_percentile"] = _percentile(hv_series, hv20)

    atm_iv = None
    try:
        chain = await fetch_option_chain_cached(symbol)
    except Exception:
        chain = pd.DataFrame()

    if isinstance(chain, pd.DataFrame) and not chain.empty:
        chain = chain.dropna(subset=["strike"])
        if not chain.empty:
            price_ref = None
            if daily_history is not None and not daily_history.empty:
                try:
                    price_ref = float(daily_history["close"].iloc[-1])
                except Exception:
                    price_ref = None
            if price_ref is None:
                try:
                    price_ref = float(chain.get("underlying_price").dropna().iloc[-1])
                except Exception:
                    price_ref = None
            candidates = chain.copy()
            if price_ref is not None:
                candidates["strike_diff"] = (candidates["strike"] - price_ref).abs()
            else:
                candidates["strike_diff"] = 0.0
            delta_series = pd.to_numeric(candidates.get("delta"), errors="coerce")
            if delta_series is None:
                delta_series = pd.Series(dtype=float)
            candidates["abs_delta"] = delta_series.abs()
            candidates = candidates.dropna(subset=["abs_delta", "dte"])
            candidates = candidates[(candidates["dte"].astype(float) >= 15) & (candidates["dte"].astype(float) <= 60)]
            if not candidates.empty:
                candidates = candidates.sort_values(by=["strike_diff", "abs_delta"])
                for _, row in candidates.iterrows():
                    iv_val = row.get("iv") or row.get("implied_volatility")
                    if isinstance(iv_val, (int, float)) and math.isfinite(iv_val) and iv_val > 0:
                        atm_iv = float(iv_val) * 100.0
                        break
    if atm_iv is not None:
        metrics["iv_atm"] = round(atm_iv, 2)
        if hv_series is not None and hv_series.size:
            hv_min = float(np.nanmin(hv_series))
            hv_max = float(np.nanmax(hv_series))
            if hv_max > hv_min:
                metrics["iv_rank"] = round(_norm(atm_iv, hv_min, hv_max) * 100.0, 2)
                percentile = _percentile(hv_series, atm_iv)
                if percentile is not None:
                    metrics["iv_percentile"] = round(percentile / 100.0, 4)
            hv20_val = metrics.get("hv_20")
            if hv20_val:
                metrics["iv_to_hv_ratio"] = round(atm_iv / hv20_val, 4)

    _IV_METRICS_CACHE[key] = (now, dict(metrics))
    return metrics


async def _build_interval_context(symbol: str, interval: str, lookback: int) -> Dict[str, Any]:
    cache_key = (symbol.upper(), interval, int(lookback))
    now = time.monotonic()
    cached = _MULTI_CONTEXT_CACHE.get(cache_key)
    if cached and now - cached[0] < _MULTI_CONTEXT_CACHE_TTL:
        payload = dict(cached[1])
        payload["cached"] = True
        return payload

    frame = get_candles(symbol, interval, lookback=lookback)
    if frame.empty:
        raise HTTPException(status_code=502, detail=f"No market data available for {symbol.upper()} ({interval}).")

    df = frame.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    history = df.set_index("time")
    key_levels = _extract_key_levels(history)
    snapshot = _build_market_snapshot(history, key_levels)

    bars: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        ts = pd.Timestamp(row["time"])
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        bars.append(
            {
                "time": ts.isoformat(),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": _safe_number(row.get("volume")) or 0.0,
            }
        )

    ema9_series = ema(history["close"], 9) if len(history) >= 9 else pd.Series(dtype=float)
    ema20_series = ema(history["close"], 20) if len(history) >= 20 else pd.Series(dtype=float)
    ema50_series = ema(history["close"], 50) if len(history) >= 50 else pd.Series(dtype=float)
    vwap_series = vwap(history["close"], history["volume"])
    atr_series = atr(history["high"], history["low"], history["close"], 14)
    adx_series = adx(history["high"], history["low"], history["close"], 14)

    indicators = {
        "ema9": _series_points(ema9_series),
        "ema20": _series_points(ema20_series),
        "ema50": _series_points(ema50_series),
        "vwap": _series_points(vwap_series),
        "atr14": _series_points(atr_series),
        "adx14": _series_points(adx_series),
    }

    payload = {
        "interval": interval,
        "lookback": lookback,
        "bars": bars,
        "key_levels": key_levels,
        "snapshot": snapshot,
        "indicators": indicators,
        "cached": False,
    }

    _MULTI_CONTEXT_CACHE[cache_key] = (now, dict(payload))
    return payload


tv_api = APIRouter(prefix="/tv-api", tags=["tv"])


@tv_api.get("/config")
async def tv_config() -> Dict[str, Any]:
    return {
        "supports_search": True,
        "supports_group_request": False,
        "supports_marks": False,
        "supports_timescale_marks": False,
        "supports_time": True,
        "supported_resolutions": TV_SUPPORTED_RESOLUTIONS,
        "exchanges": [{"value": "", "name": "TradingCoach", "desc": "Trading Coach"}],
        "symbols_types": [{"name": "All", "value": "all"}],
    }


@tv_api.get("/symbols")
async def tv_symbol(symbol: str = Query(..., alias="symbol")) -> Dict[str, Any]:
    settings = get_settings()
    timeframe = "1"
    history = await _load_remote_ohlcv(symbol, timeframe)
    last_price = None
    if history is not None and not history.empty:
        last_price = float(history["close"].iloc[-1])

    return {
        "name": symbol.upper(),
        "ticker": symbol.upper(),
        "description": symbol.upper(),
        "type": "stock",
        "session": "0930-1600",
        "timezone": "America/New_York",
        "exchange": "CUSTOM",
        "minmov": 1,
        "pricescale": _price_scale_for(last_price),
        "has_intraday": True,
        "has_no_volume": False,
        "has_weekly_and_monthly": True,
        "supported_resolutions": TV_SUPPORTED_RESOLUTIONS,
        "volume_precision": 0,
        "data_status": "streaming" if settings.polygon_api_key else "endofday",
    }


@tv_api.get("/bars")
async def tv_bars(
    symbol: str = Query(...),
    resolution: str = Query(...),
    from_: Optional[int] = Query(None, alias="from"),
    to: Optional[int] = Query(None),
    range_: Optional[str] = Query(None, alias="range"),
) -> Dict[str, Any]:
    timeframe = _resolution_to_timeframe(resolution)
    logger.info(
        "tv-api/bars request symbol=%s resolution=%s -> timeframe=%s window=%s-%s range=%s",
        symbol,
        resolution,
        timeframe,
        from_,
        to,
        range_,
    )
    if timeframe is None:
        raise HTTPException(status_code=400, detail=f"Unsupported resolution {resolution}")

    history = await _load_remote_ohlcv(symbol, timeframe)
    src_tf = timeframe
    # Resiliency: if intraday is unavailable for this symbol, try a coarser TF
    if history is None or history.empty:
        alt_tf = None
        if timeframe not in {"D", "1D"}:
            alt_tf = "15"  # 15-minute fallback
        if alt_tf:
            history = await _load_remote_ohlcv(symbol, alt_tf)
            src_tf = alt_tf
        if history is None or history.empty:
            # Last resort: show daily so the chart isn't blank
            history = await _load_remote_ohlcv(symbol, "D")
            src_tf = "D"
        if history is None or history.empty:
            logger.warning(
                "tv-api/bars no data for symbol=%s resolution=%s tried_tf=%s",
                symbol,
                resolution,
                timeframe,
            )
            return {"s": "no_data"}

    history = history.sort_index()
    # Compute window: allow unix seconds or milliseconds; allow missing values with range fallback
    now_sec = int(pd.Timestamp.utcnow().timestamp())

    def _to_seconds(val: Optional[int]) -> Optional[int]:
        if val is None:
            return None
        # Heuristic: treat 13-digit as ms
        if val > 10_000_000_000:
            return int(val / 1000)
        return int(val)

    start_s = _to_seconds(from_)
    end_s = _to_seconds(to)

    def _range_to_span(res: str, token: Optional[str]) -> int:
        # Accept e.g., 5D, 3D, 1W, 1M; default 5D
        if not token:
            token = "5D"
        t = token.strip().upper()
        try:
            if t.endswith("D"):
                days = int(t[:-1] or "1")
                return days * 24 * 60 * 60
            if t.endswith("W"):
                weeks = int(t[:-1] or "1")
                return weeks * 7 * 24 * 60 * 60
            if t.endswith("M"):
                months = int(t[:-1] or "1")
                return months * 30 * 24 * 60 * 60
        except Exception:
            pass
        # Fallback to ~600 bars worth of time
        minutes = _resolution_to_minutes(resolution) if resolution else 5
        return max(600 * minutes * 60, 24 * 60 * 60)

    if end_s is None:
        end_s = now_sec
    if start_s is None:
        span = _range_to_span(resolution, range_)
        start_s = end_s - span

    start_ts = pd.to_datetime(start_s, unit="s", utc=True)
    end_ts = pd.to_datetime(end_s, unit="s", utc=True)
    window = history.loc[(history.index >= start_ts) & (history.index <= end_ts)]

    if window.empty:
        logger.info(
            "tv-api/bars empty window for symbol=%s src_tf=%s; returning tail",
            symbol,
            src_tf,
        )
        window = history.tail(min(len(history), 600))
        if window.empty:
            earlier = history[history.index < start_ts]
            if earlier.empty:
                logger.warning(
                    "tv-api/bars no earlier data for symbol=%s src_tf=%s",
                    symbol,
                    src_tf,
                )
                return {"s": "no_data"}
            next_time = int(earlier.index[-1].timestamp())
            return {"s": "no_data", "nextTime": next_time}

    volume_values = [float(val) for val in window["volume"].tolist()] if "volume" in window.columns else [0.0] * len(window)

    payload = {
        "s": "ok",
        "t": [int(ts.timestamp()) for ts in window.index],
        "o": [round(float(val), 6) for val in window["open"].tolist()],
        "h": [round(float(val), 6) for val in window["high"].tolist()],
        "l": [round(float(val), 6) for val in window["low"].tolist()],
        "c": [round(float(val), 6) for val in window["close"].tolist()],
        "v": volume_values,
    }
    try:
        logger.info(
            "tv-api/bars OK symbol=%s src_tf=%s bars=%d window=%s-%s",
            symbol,
            src_tf,
            len(payload["t"]),
            start_ts,
            end_ts,
        )
    except Exception:
        pass
    return payload


# ---------------------------------------------------------------------------
# GPT router
# ---------------------------------------------------------------------------

gpt = APIRouter(prefix="/gpt", tags=["gpt"])


@gpt.get("/health", summary="Lightweight readiness probe")
async def gpt_health(_: AuthedUser = Depends(require_api_key)) -> Dict[str, Any]:
    settings = get_settings()

    async def _check_polygon() -> Dict[str, Any]:
        if not settings.polygon_api_key:
            return {"status": "missing"}
        try:
            sample = await _fetch_polygon_ohlcv("SPY", "5")
            if sample is None or sample.empty:
                return {"status": "unavailable"}
            latest = sample.index[-1]
            if latest.tzinfo is None:
                latest = latest.tz_localize("UTC")
            age_minutes = (pd.Timestamp.utcnow() - latest).total_seconds() / 60.0
            return {"status": "ok", "latest_bar_utc": latest.isoformat(), "age_minutes": round(age_minutes, 2)}
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Polygon health check failed: %s", exc)
            return {"status": "error", "error": str(exc)}

    async def _check_tradier() -> Dict[str, Any]:
        if not settings.tradier_token:
            return {"status": "missing"}
        try:
            chain = await fetch_option_chain("SPY")
        except TradierNotConfiguredError:
            return {"status": "missing"}
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Tradier health check failed: %s", exc)
            return {"status": "error", "error": str(exc)}
        if chain is None or chain.empty:
            return {"status": "unavailable"}
        sample = chain.iloc[0].to_dict()
        return {
            "status": "ok",
            "symbol": sample.get("symbol"),
            "expiration": sample.get("expiration_date"),
        }

    polygon_status, tradier_status = await asyncio.gather(_check_polygon(), _check_tradier())

    return {
        "status": "ok",
        "services": {
            "polygon": polygon_status,
            "tradier": tradier_status,
        },
    }


@gpt.post("/scan", summary="Rank trade setups across a list of tickers")
async def gpt_scan(
    universe: ScanUniverse,
    request: Request,
    user: AuthedUser = Depends(require_api_key),
) -> List[Dict[str, Any]]:
    if not universe.tickers:
        raise HTTPException(status_code=400, detail="No tickers provided")

    style_filter = _normalize_style(universe.style)
    data_timeframe = {"scalp": "1", "intraday": "5", "swing": "60", "leap": "D"}.get(style_filter, "5")

    settings = get_settings()
    market_data = await _collect_market_data(universe.tickers, timeframe=data_timeframe)
    if not market_data:
        raise HTTPException(status_code=502, detail="No market data available for the requested tickers.")
    signals = await scan_market(universe.tickers, market_data)

    unique_symbols = sorted({signal.symbol for signal in signals})

    polygon_enabled = bool(settings.polygon_api_key)
    tradier_enabled = bool(settings.tradier_token)

    benchmark_symbol = "SPY"
    benchmark_history: pd.DataFrame | None = market_data.get(benchmark_symbol)
    if benchmark_history is None:
        try:
            benchmark_history = await _load_remote_ohlcv(benchmark_symbol, data_timeframe)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Benchmark data fetch failed for %s: %s", benchmark_symbol, exc)
            benchmark_history = None

    polygon_chains: Dict[str, pd.DataFrame] = {}
    if unique_symbols and polygon_enabled:
        try:
            tasks = [fetch_polygon_option_chain(symbol) for symbol in unique_symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for symbol, result in zip(unique_symbols, results):
                if isinstance(result, Exception):
                    logger.warning("Polygon option chain fetch failed for %s: %s", symbol, result)
                    polygon_chains[symbol] = pd.DataFrame()
                else:
                    polygon_chains[symbol] = result if isinstance(result, pd.DataFrame) else pd.DataFrame()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Polygon option chain request error: %s", exc)
            polygon_chains.clear()

    tradier_suggestions: Dict[str, Dict[str, Any] | None] = {}
    if unique_symbols and tradier_enabled:
        try:
            tasks = [select_tradier_contract(symbol) for symbol in unique_symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for symbol, result in zip(unique_symbols, results):
                if isinstance(result, Exception):
                    logger.warning("Tradier contract lookup failed for %s: %s", symbol, result)
                    tradier_suggestions[symbol] = None
                else:
                    tradier_suggestions[symbol] = result
        except Exception as exc:  # pragma: no cover - safety net
            logger.warning("Tradier integration error: %s", exc)

    payload: List[Dict[str, Any]] = []
    options_cache: Dict[tuple[str, str], Dict[str, Any] | None] = {}
    for signal in signals:
        style = _style_for_strategy(signal.strategy_id)
        if style_filter and style_filter != style:
            continue
        history = market_data[signal.symbol]
        latest_row = history.iloc[-1]
        entry_price = float(latest_row["close"])
        key_levels = _extract_key_levels(history)
        # Strategy direction inference hint (AI will make the final decision)
        direction_hint = signal.features.get("direction_bias")
        if direction_hint not in {"long", "short"}:
            direction_hint = _direction_for_strategy(signal.strategy_id)

        snapshot = _build_market_snapshot(history, key_levels)
        snapshot.setdefault("trend", {})["direction_hint"] = direction_hint
        snapshot.setdefault("price", {})["entry_reference"] = entry_price

        indicators = _indicators_for_strategy(signal.strategy_id)
        ema_spans = sorted(
            {
                int(token[3:])
                for token in indicators
                if token.upper().startswith("EMA") and token[3:].isdigit()
            }
        )
        if not ema_spans:
            ema_spans = [9, 21]

        base_url = str(request.base_url).rstrip("/")
        interval = _timeframe_for_style(style)
        title = f"{signal.symbol.upper()} {signal.strategy_id}"
        chart_query: Dict[str, Any] = {
            "symbol": signal.symbol.upper(),
            "interval": interval,
            "ema": ",".join(str(span) for span in ema_spans),
            "view": _view_for_style(style),
            "title": title,
            "vwap": "1",
            "theme": "dark",
        }
        plan_payload: Dict[str, Any] | None = None
        enhancements: Dict[str, Any] | None = None
        chain = polygon_chains.get(signal.symbol)
        try:
            enhancements = compute_context_overlays(
                history,
                symbol=signal.symbol,
                interval=interval,
                benchmark_history=benchmark_history,
                options_chain=chain,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("compute_context_overlays failed for %s: %s", signal.symbol, exc)
            enhancements = {}

        if signal.plan is not None:
            plan_payload = signal.plan.as_dict()
            chart_query["entry"] = f"{signal.plan.entry:.2f}"
            chart_query["stop"] = f"{signal.plan.stop:.2f}"
            chart_query["tp"] = ",".join(f"{target:.2f}" for target in signal.plan.targets)
            chart_query.setdefault("direction", signal.plan.direction)
            if signal.plan.atr and "atr" not in chart_query:
                chart_query["atr"] = f"{float(signal.plan.atr):.4f}"
            # Always include a notes field for chart URLs, even if brief
            notes_text = (signal.plan.notes or "").strip()
            if not notes_text:
                # fallback: concise phrase using strategy + direction
                notes_text = f"{signal.strategy_id} ({direction_hint or signal.plan.direction})"
            chart_query["notes"] = notes_text[:140]
        overlay_params = _encode_overlay_params(enhancements or {})
        for key, value in overlay_params.items():
            chart_query[key] = value
        level_tokens = _extract_levels_for_chart(key_levels)
        if level_tokens:
            chart_query["levels"] = ",".join(level_tokens)
        chart_query["strategy"] = signal.strategy_id
        if direction_hint:
            chart_query["direction"] = direction_hint
        atr_hint = snapshot.get("indicators", {}).get("atr14")
        if isinstance(atr_hint, (int, float)) and math.isfinite(atr_hint):
            chart_query["atr"] = f"{float(atr_hint):.4f}"
        chart_query = {key: str(value) for key, value in chart_query.items() if value is not None}
        feature_payload = _serialize_features(signal.features)
        feature_payload.setdefault("atr", snapshot.get("indicators", {}).get("atr14"))
        feature_payload.setdefault("adx", snapshot.get("indicators", {}).get("adx14"))
        if signal.plan is not None:
            plan_dict = signal.plan.as_dict()
            for key, value in plan_dict.items():
                feature_payload[f"plan_{key}"] = value

        polygon_bundle: Dict[str, Any] | None = None
        if polygon_chains:
            cache_key = (signal.symbol, signal.strategy_id)
            polygon_bundle = options_cache.get(cache_key)
            if polygon_bundle is None:
                chain = polygon_chains.get(signal.symbol)
                rules = signal.options_rules if isinstance(signal.options_rules, dict) else None
                polygon_bundle = summarize_polygon_chain(chain, rules=rules, top_n=3) if chain is not None else None
                options_cache[cache_key] = polygon_bundle

        best_contract = None
        if polygon_bundle and polygon_bundle.get("best"):
            best_contract = polygon_bundle.get("best")
        else:
            best_contract = tradier_suggestions.get(signal.symbol)

        payload.append(
            {
                "symbol": signal.symbol,
                "style": style,
                "strategy_id": signal.strategy_id,
                "description": signal.description,
                "score": signal.score,
                "contract_suggestion": best_contract,
                "direction_hint": direction_hint,
                "key_levels": key_levels,
                "market_snapshot": snapshot,
                "charts": {
                    "params": chart_query,
                },
                "features": feature_payload,
                **({"plan": plan_payload} if plan_payload else {}),
                "warnings": plan_payload.get("warnings") if plan_payload else [],
                "data": {
                    "bars": f"{base_url}/gpt/context/{signal.symbol}?interval={interval}&lookback=300"
                },
                "context_overlays": enhancements,
                **({"options": polygon_bundle} if polygon_bundle else {}),
            }
        )

    logger.info("scan universe=%s user=%s results=%d", universe.tickers, user.user_id, len(payload))
    return payload


@gpt.post("/plan", summary="Return a single trade plan for a symbol", response_model=PlanResponse)
async def gpt_plan(
    request_payload: PlanRequest,
    request: Request,
    user: AuthedUser = Depends(require_api_key),
) -> PlanResponse:
    """Compatibility endpoint that returns the top plan for a single symbol.

    Internally reuses /gpt/scan to keep plan logic centralized.
    """
    symbol = (request_payload.symbol or "").strip().upper()
    if not symbol:
        raise HTTPException(status_code=400, detail="Symbol is required")
    universe = ScanUniverse(tickers=[symbol], style=request_payload.style)
    results = await gpt_scan(universe, request, user)
    if not results:
        raise HTTPException(status_code=404, detail=f"No valid plan for {symbol}")
    first = next((item for item in results if (item.get("symbol") or "").upper() == symbol), results[0])

    plan_id = uuid.uuid4().hex[:10]
    version = 1
    base_url = str(request.base_url)
    idea_url = _build_idea_url(base_url, plan_id, version)

    # Build calc_notes + htf from available payload
    plan = first.get("plan") or {}
    charts = (first.get("charts") or {}).get("params") or {}
    snapshot = first.get("market_snapshot") or {}
    indicators = (snapshot.get("indicators") or {})
    volatility = (snapshot.get("volatility") or {})
    rr_inputs = None
    try:
        entry = float(plan.get("entry")) if plan.get("entry") is not None else float(charts.get("entry"))
        stop = float(plan.get("stop")) if plan.get("stop") is not None else float(charts.get("stop"))
        tp_csv = charts.get("tp") or ",".join(str(x) for x in (plan.get("targets") or []))
        tp1 = float(str(tp_csv).split(",")[0]) if tp_csv else None
        if tp1 is not None:
            rr_inputs = {"entry": entry, "stop": stop, "tp1": tp1}
    except Exception:
        rr_inputs = None
    calc_notes = {
        "atr14": indicators.get("atr14"),
        "stop_multiple": None,
        "em_cap_applied": bool((volatility.get("expected_move_horizon") if isinstance(volatility, dict) else None)),
        **({"rr_inputs": rr_inputs} if rr_inputs else {}),
    }
    # Infer snapped_targets by comparing target prices to named levels (key_levels + overlays)
    snapped_names: List[str] = []
    try:
        targets_list = plan.get("targets") or []
        if not targets_list and charts.get("tp"):
            targets_list = [float(x.strip()) for x in str(charts.get("tp")).split(",") if x.strip()]
        atr_val = float(indicators.get("atr14") or 0.0)
        window = max(atr_val * 0.30, 0.0)
        levels_dict = first.get("key_levels") or {}
        overlays = first.get("context_overlays") or {}
        named: List[Tuple[str, float]] = []
        # Key levels (map some to canonical short labels)
        alias = {
            "opening_range_high": "ORH",
            "opening_range_low": "ORL",
            "prev_high": "prev_high",
            "prev_low": "prev_low",
            "prev_close": "prev_close",
            "session_high": "session_high",
            "session_low": "session_low",
            "gap_fill": "gap_fill",
        }
        for k, v in (levels_dict or {}).items():
            if isinstance(v, (int, float)):
                try:
                    named.append((alias.get(k, k), float(v)))
                except Exception:
                    continue
        # Volume profile (VAH/VAL/POC/VWAP)
        vp = overlays.get("volume_profile") or {}
        for lab, key in [("VAH", "vah"), ("VAL", "val"), ("POC", "poc"), ("VWAP", "vwap")]:
            val = vp.get(key)
            if isinstance(val, (int, float)):
                try:
                    named.append((lab, float(val)))
                except Exception:
                    pass
        # Anchored VWAPs
        av = overlays.get("avwap") or {}
        av_map = {
            "from_open": "AVWAP(open)",
            "from_prev_close": "AVWAP(prev_close)",
            "from_session_low": "AVWAP(session_low)",
            "from_session_high": "AVWAP(session_high)",
        }
        for k, lab in av_map.items():
            val = av.get(k)
            if isinstance(val, (int, float)):
                try:
                    named.append((lab, float(val)))
                except Exception:
                    pass
        for tp in targets_list[:2]:
            try:
                tp_f = float(tp)
            except Exception:
                continue
            nearest = None
            best_name = None
            for name, val in named:
                if nearest is None or abs(val - tp_f) < abs(nearest - tp_f):
                    nearest = val
                    best_name = name
            if nearest is not None and abs(nearest - tp_f) <= window and best_name:
                snapped_names.append(best_name)
    except Exception:
        snapped_names = []

    htf = {
        "bias": ((snapshot.get("trend") or {}).get("ema_stack") or "unknown"),
        "snapped_targets": snapped_names,
    }
    data_quality = {
        "series_present": True,
        "iv_present": True,
        "earnings_present": True,
    }
    plan_warnings: List[str] = first.get("warnings") or plan.get("warnings") or []

    # Build chart URL via internal validator (best effort)
    chart_url_value: Optional[str] = None
    chart_params_payload = (first.get("charts") or {}).get("params")
    if isinstance(chart_params_payload, dict) and chart_params_payload.get("symbol") and chart_params_payload.get("interval"):
        try:
            chart_model = ChartParams(**chart_params_payload)
            chart_links = await gpt_chart_url(chart_model, request)
            chart_url_value = chart_links.interactive
        except Exception:
            chart_url_value = None

    price_close = snapshot.get("price", {}).get("close")
    decimals_value = 2
    if isinstance(price_close, (int, float)):
        try:
            scale = _price_scale_for(float(price_close))
            if scale > 0:
                decimals_value = int(round(math.log10(scale)))
        except Exception:
            decimals_value = 2

    plan_core = _extract_plan_core(first, plan_id, version, decimals_value)
    plan_core["idea_url"] = idea_url
    if plan_warnings:
        plan_core["warnings"] = plan_warnings
    summary_snapshot = _build_snapshot_summary(first)
    idea_snapshot = {
        "plan": plan_core,
        "summary": summary_snapshot,
        "volatility_regime": summary_snapshot.get("volatility_regime"),
        "htf": htf,
        "data_quality": data_quality,
        "chart_url": chart_url_value,
        "options": first.get("options"),
        "why_this_works": [],
        "invalidation": [],
        "risk_note": None,
    }
    await _store_idea_snapshot(plan_id, idea_snapshot)

    # Debug info: include any structural TP1 notes from features
    debug_payload = {}
    try:
        feats = first.get("features") or {}
        tp1_dbg = feats.get("tp1_struct_debug")
        if tp1_dbg:
            debug_payload["tp1"] = tp1_dbg
    except Exception:
        debug_payload = {}

    return PlanResponse(
        plan_id=plan_id,
        version=version,
        idea_url=idea_url,
        warnings=plan_warnings or None,
        symbol=first.get("symbol"),
        style=first.get("style"),
        strategy_id=first.get("strategy_id"),
        description=first.get("description"),
        score=first.get("score"),
        plan=first.get("plan"),
        charts=first.get("charts"),
        key_levels=first.get("key_levels"),
        market_snapshot=first.get("market_snapshot"),
        features=first.get("features"),
        options=first.get("options"),
        calc_notes=calc_notes,
        htf=htf,
        decimals=decimals_value,
        data_quality=data_quality,
        debug=debug_payload or None,
    )


@app.post("/internal/idea/store", include_in_schema=False, tags=["internal"])
async def internal_idea_store(payload: IdeaStoreRequest, request: Request) -> IdeaStoreResponse:
    plan_block = payload.plan or {}
    plan_id = plan_block.get("plan_id")
    version = plan_block.get("version")
    if not plan_id or version is None:
        raise HTTPException(status_code=400, detail="plan.plan_id and plan.version are required")
    base_url = str(request.base_url)
    idea_url = _build_idea_url(base_url, plan_id, int(version))
    snapshot = payload.model_dump()
    snapshot.setdefault("chart_url", None)
    await _store_idea_snapshot(plan_id, snapshot)
    return IdeaStoreResponse(plan_id=plan_id, idea_url=idea_url)


@app.get("/idea/{plan_id}")
async def get_latest_idea(plan_id: str) -> Dict[str, Any]:
    return await _get_idea_snapshot(plan_id)


@app.get("/idea/{plan_id}/{version}")
async def get_idea_version(plan_id: str, version: int) -> Dict[str, Any]:
    return await _get_idea_snapshot(plan_id, version=version)


@app.get("/stream/market")
async def stream_market(symbol: str = Query(..., min_length=1)) -> StreamingResponse:
    async def event_generator():
        async for chunk in _stream_generator(symbol.upper()):
            yield chunk

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/internal/stream/push", include_in_schema=False, tags=["internal"])
async def internal_stream_push(payload: StreamPushRequest) -> Dict[str, str]:
    symbol = (payload.symbol or "").upper()
    if not symbol:
        raise HTTPException(status_code=400, detail="Symbol is required")
    await _publish_stream_event(symbol, payload.event or {})
    return {"status": "ok"}


@app.get("/simulate")
async def simulate_trade(
    symbol: str = Query(..., min_length=1),
    minutes: int = Query(30, ge=5, le=300),
    entry: float = Query(...),
    stop: float = Query(...),
    tp1: float = Query(...),
    tp2: float | None = Query(None),
    direction: str = Query(..., regex="^(long|short)$"),
) -> StreamingResponse:
    params = {
        "minutes": minutes,
        "entry": entry,
        "stop": stop,
        "tp1": tp1,
        "tp2": tp2,
        "direction": direction,
    }

    async def playback():
        async for chunk in _simulate_generator(symbol.upper(), params):
            yield chunk

    return StreamingResponse(playback(), media_type="text/event-stream")


@gpt.post("/multi-context", summary="Return multi-interval context with vol metrics")
async def gpt_multi_context(
    request_payload: MultiContextRequest,
    _: AuthedUser = Depends(require_api_key),
) -> MultiContextResponse:
    symbol = (request_payload.symbol or "").strip().upper()
    if not symbol:
        raise HTTPException(status_code=400, detail="Symbol is required")
    if not request_payload.intervals:
        raise HTTPException(status_code=400, detail="At least one interval is required")

    contexts: List[Dict[str, Any]] = []
    seen: set[str] = set()
    lookback = int(request_payload.lookback or 300)
    for token in request_payload.intervals:
        raw = (token or "").strip()
        if not raw:
            continue
        try:
            normalized = normalize_interval(raw)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if normalized in seen:
            continue
        seen.add(normalized)
        try:
            context = await _build_interval_context(symbol, normalized, lookback)
        except HTTPException as exc:
            if exc.status_code == 502:
                raise HTTPException(status_code=404, detail=f"No data for {symbol} ({normalized})") from exc
            raise
        context["interval"] = normalized
        context["requested"] = raw
        if not request_payload.include_series:
            # Trim heavy series when gating is requested
            context.pop("bars", None)
            indicators = context.get("indicators")
            if isinstance(indicators, dict):
                context["indicators"] = {k: v[-1] if isinstance(v, list) and v else v for k, v in indicators.items()}
        contexts.append(context)

    if not contexts:
        raise HTTPException(status_code=400, detail="No valid intervals provided")

    iv_metrics = await _compute_iv_metrics(symbol)
    volatility_regime = {
        "iv_rank": iv_metrics.get("iv_rank"),
        "iv_percentile": iv_metrics.get("iv_percentile"),
        "iv_atm": iv_metrics.get("iv_atm"),
        "hv_20": iv_metrics.get("hv_20"),
        "hv_60": iv_metrics.get("hv_60"),
        "hv_120": iv_metrics.get("hv_120"),
        "hv_20_percentile": iv_metrics.get("hv_20_percentile"),
        "iv_to_hv_ratio": iv_metrics.get("iv_to_hv_ratio"),
        "timestamp": iv_metrics.get("timestamp"),
        "skew_25d": iv_metrics.get("skew_25d"),
    }
    enrichment = await _fetch_context_enrichment(symbol)
    sentiment = (enrichment or {}).get("sentiment")
    events = (enrichment or {}).get("events")
    earnings = (enrichment or {}).get("earnings")

    # Build summary block
    frames_used = [c.get("interval") for c in contexts]
    trend_notes: Dict[str, str] = {}
    votes: List[int] = []
    for c in contexts:
        snap = c.get("snapshot") or {}
        trend = (snap.get("trend") or {}).get("ema_stack")
        label = "flat"
        if trend == "bullish":
            label = "up"
            votes.append(1)
        elif trend == "bearish":
            label = "down"
            votes.append(-1)
        trend_notes[str(c.get("interval"))] = label
    confluence_score = None
    if votes:
        same_dir = abs(sum(1 for v in votes if v > 0) - sum(1 for v in votes if v < 0))
        confluence_score = round(max(0.0, min(1.0, same_dir / max(1, len(votes)))), 2)

    # Vol regime label
    regime_label = None
    try:
        iv_rank = volatility_regime.get("iv_rank")
        if isinstance(iv_rank, (int, float)):
            if iv_rank >= 90:
                regime_label = "extreme"
            elif iv_rank >= 75:
                regime_label = "elevated"
            elif iv_rank <= 25:
                regime_label = "low"
            else:
                regime_label = "normal"
    except Exception:
        pass
    vol_summary = dict(volatility_regime)
    if regime_label:
        vol_summary["regime_label"] = regime_label

    # Expected move horizon: use first snapshot that has it
    expected_move_horizon = None
    for c in contexts:
        snap = c.get("snapshot") or {}
        vol = snap.get("volatility") or {}
        em = vol.get("expected_move_horizon")
        if isinstance(em, (int, float)):
            expected_move_horizon = float(em)
            break

    # Nearby levels (compact markers)
    marker_keys = ["POC", "VAH", "VAL", "prev_high", "prev_low", "prev_close", "opening_range_high", "opening_range_low", "session_high", "session_low"]
    nearby_levels: List[str] = []
    for c in contexts:
        lv = (c.get("snapshot") or {}).get("levels") or {}
        for k in marker_keys:
            if k in lv and k not in nearby_levels:
                nearby_levels.append(k)
        if len(nearby_levels) >= 6:
            break
    summary = {
        "frames_used": frames_used,
        "confluence_score": confluence_score,
        "trend_notes": trend_notes,
        "volatility_regime": vol_summary,
        "expected_move_horizon": expected_move_horizon,
        "nearby_levels": nearby_levels,
    }

    # Decimals based on first frame's close
    decimals = 2
    try:
        first_close = None
        for c in contexts:
            snap = c.get("snapshot") or {}
            price = (snap.get("price") or {}).get("close")
            if isinstance(price, (int, float)):
                first_close = float(price)
                break
        if first_close is not None:
            scale = _price_scale_for(first_close)
            # pricescale = 10**decimals
            import math as _math
            decimals = int(round(_math.log10(scale))) if scale > 0 else 2
    except Exception:
        decimals = 2

    data_quality = {
        "series_present": bool(request_payload.include_series),
        "iv_present": any(volatility_regime.get(k) is not None for k in ("iv_rank", "iv_atm")),
        "earnings_present": earnings is not None,
    }

    return MultiContextResponse(
        symbol=symbol,
        contexts=contexts,
        volatility_regime=volatility_regime,
        sentiment=sentiment,
        events=events,
        earnings=earnings,
        summary=summary,
        decimals=decimals,
        data_quality=data_quality,
    )


def _infer_contract_side(side: str | None, bias: str | None) -> str | None:
    token = (side or "").strip().lower()
    if token.startswith("c"):
        return "call"
    if token.startswith("p"):
        return "put"
    bias_token = (bias or "").strip().lower()
    if bias_token.startswith("short") or bias_token in {"bearish", "put"}:
        return "put"
    if bias_token.startswith("long") or bias_token in {"bullish", "call"}:
        return "call"
    return None


def _prepare_contract_filters(payload: ContractsRequest, style: str) -> Dict[str, float | int]:
    config: Dict[str, float | int] = _style_default_bounds(style)
    if payload.min_dte is not None:
        config["min_dte"] = int(payload.min_dte)
    if payload.max_dte is not None:
        config["max_dte"] = int(payload.max_dte)
    if payload.min_delta is not None:
        config["min_delta"] = float(payload.min_delta)
    if payload.max_delta is not None:
        config["max_delta"] = float(payload.max_delta)
    if payload.max_spread_pct is not None:
        config["max_spread_pct"] = float(payload.max_spread_pct)
    if payload.min_oi is not None:
        config["min_oi"] = int(payload.min_oi)
    return config


def _screen_contracts(
    chain: pd.DataFrame,
    quotes: Dict[str, Dict[str, Any]],
    *,
    symbol: str,
    style: str,
    side: str | None,
    filters: Dict[str, float | int],
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    min_dte = int(filters.get("min_dte", 0))
    max_dte = int(filters.get("max_dte", 366))
    min_delta = float(filters.get("min_delta", 0.0))
    max_delta = float(filters.get("max_delta", 1.0))
    max_spread = float(filters.get("max_spread_pct", 100.0))
    min_oi = int(filters.get("min_oi", 0))

    for _, row in chain.iterrows():
        option_symbol = row.get("symbol")
        if not option_symbol:
            continue
        quote = quotes.get(option_symbol)
        row_type = (row.get("option_type") or "").strip().lower()
        if side and row_type != side:
            continue

        bid = quote.get("bid") if quote else row.get("bid")
        ask = quote.get("ask") if quote else row.get("ask")
        last = quote.get("last") if quote else None
        price = _compute_price(bid, ask, last, row.get("mid"))
        if price is None:
            continue

        spread_pct = _compute_spread_pct(bid, ask, price)
        if spread_pct is None:
            spread_pct = float(row.get("spread_pct") or 999.0) * 100.0 if row.get("spread_pct") is not None else 999.0
        if spread_pct > max_spread:
            continue

        expiration = quote.get("expiration_date") if quote else row.get("expiration_date")
        dte = row.get("dte")
        if dte is None and expiration:
            try:
                exp_ts = pd.Timestamp(expiration)
                dte = max((exp_ts.date() - pd.Timestamp.utcnow().date()).days, 0)
            except Exception:  # pragma: no cover - defensive
                dte = None
        if dte is None:
            continue
        dte_int = int(dte)
        if dte_int < min_dte or dte_int > max_dte:
            continue

        delta = quote.get("delta") if quote else row.get("delta")
        if delta is None or not math.isfinite(delta):
            continue
        abs_delta = abs(float(delta))
        if abs_delta < min_delta or abs_delta > max_delta:
            continue

        oi = quote.get("open_interest") if quote else row.get("open_interest")
        if oi is None:
            oi = row.get("open_interest") or row.get("oi")
        oi_val = float(oi or 0)
        if oi_val < min_oi:
            continue

        volume = quote.get("volume") if quote else row.get("volume")
        gamma = quote.get("gamma") if quote else row.get("gamma")
        theta = quote.get("theta") if quote else row.get("theta")
        vega = quote.get("vega") if quote else row.get("vega")
        iv = quote.get("iv") if quote else row.get("iv")

        tradeability = _tradeability_score(
            spread_pct=spread_pct,
            delta=abs_delta,
            style=style,
            oi=oi_val,
            iv_rank=None,
            theta=float(theta) if isinstance(theta, (int, float)) else None,
        )

        contract = {
            "label": _contract_label(symbol, expiration, row.get("strike"), row_type),
            "symbol": option_symbol,
            "expiry": expiration,
            "dte": dte_int,
            "strike": row.get("strike"),
            "type": row_type.upper() if row_type else None,
            "price": round(price, 2),
            "bid": float(bid) if isinstance(bid, (int, float)) else None,
            "ask": float(ask) if isinstance(ask, (int, float)) else None,
            "spread_pct": round(float(spread_pct), 2) if spread_pct is not None else None,
            "volume": int(volume) if isinstance(volume, (int, float)) else None,
            "oi": int(oi_val),
            "delta": float(delta),
            "gamma": float(gamma) if isinstance(gamma, (int, float)) else None,
            "theta": float(theta) if isinstance(theta, (int, float)) else None,
            "vega": float(vega) if isinstance(vega, (int, float)) else None,
            "iv": float(iv) if isinstance(iv, (int, float)) else None,
            "iv_rank": None,
            "tradeability": tradeability,
        }
        candidates.append(contract)

    candidates.sort(key=lambda item: item["tradeability"], reverse=True)
    return candidates


def _enrich_contract_with_plan(contract: Dict[str, Any], plan_anchor: Any, risk_budget: float | None) -> Dict[str, Any]:
    enriched = dict(contract)
    price = float(enriched.get("price") or 0.0)
    contract_cost = price * 100.0 if price > 0 else None
    risk_budget = float(risk_budget) if risk_budget is not None else 100.0

    risk_per_contract = round(contract_cost, 2) if contract_cost is not None else None
    if risk_per_contract and risk_per_contract > 0:
        contracts_possible = max(1, int(risk_budget // risk_per_contract)) if risk_budget > 0 else 1
    else:
        contracts_possible = 1

    pnl_block: Dict[str, Any] = {
        "per_contract_cost": risk_per_contract,
        "at_stop": None,
        "at_tp1": None,
        "at_tp2": None,
        "rr_to_tp1": None,
    }

    pl_projection: Dict[str, Any] = {
        "risk_per_contract": risk_per_contract,
        "risk_budget": float(risk_budget) if risk_budget is not None else None,
        "contracts_possible": contracts_possible,
        "max_profit_est": None,
        "max_loss_est": None,
    }

    plan = plan_anchor or {}
    try:
        underlying_entry = float(plan.get("underlying_entry") or plan.get("entry"))
        stop_level = plan.get("stop")
        tps = plan.get("targets") or plan.get("tps") or plan.get("tp") or []
        if isinstance(tps, (int, float)):
            tps = [float(tps)]
        tps = [float(val) for val in tps if isinstance(val, (int, float))]
        stop_level = float(stop_level) if stop_level is not None else None
    except (TypeError, ValueError):
        underlying_entry = None
        stop_level = None
        tps = []

    if underlying_entry is not None and (stop_level is not None or tps):
        delta_val = float(enriched.get("delta") or 0.0)
        gamma_val = float(enriched.get("gamma") or 0.0)
        theta_val = float(enriched.get("theta") or 0.0)
        vega_val = float(enriched.get("vega") or 0.0)
        iv_shift = float(plan.get("iv_shift_bps") or 0.0) / 10000.0
        slippage = abs(float(plan.get("slippage_bps") or 0.0)) / 10000.0
        horizon_minutes = float(plan.get("horizon_minutes") or 30.0)
        trading_hours = float(plan.get("trading_hours_per_day") or 6.5)
        if trading_hours <= 0:
            trading_hours = 6.5

        def _scenario(option_price: float, delta_s: float) -> float:
            d_option = (
                delta_val * delta_s
                + 0.5 * gamma_val * (delta_s ** 2)
                + vega_val * iv_shift
                - abs(theta_val) * (horizon_minutes / (60.0 * trading_hours))
            )
            raw_price = option_price + d_option
            raw_price = max(raw_price, 0.0)
            if raw_price >= option_price:
                return max(raw_price * (1.0 - slippage), 0.0)
            return max(raw_price * (1.0 + slippage), 0.0)

        stop_delta = None
        stop_price_option = None
        if stop_level is not None:
            stop_delta = float(stop_level) - float(underlying_entry)
            stop_price_option = _scenario(price, stop_delta)
            pnl_stop = (stop_price_option - price) * 100.0
            pnl_block["at_stop"] = round(pnl_stop, 2)

        tp_prices: List[float] = []
        for target in tps:
            delta_s = float(target) - float(underlying_entry)
            tp_prices.append(_scenario(price, delta_s))

        if tp_prices:
            pnl_tp1 = (tp_prices[0] - price) * 100.0
            pnl_block["at_tp1"] = round(pnl_tp1, 2)
            if stop_level is not None and pnl_block["at_stop"] is not None and pnl_block["at_stop"] < 0:
                risk = abs(pnl_block["at_stop"])
                if risk > 0:
                    pnl_block["rr_to_tp1"] = round(pnl_tp1 / risk, 2)
            if len(tp_prices) > 1:
                pnl_tp2 = (tp_prices[1] - price) * 100.0
                pnl_block["at_tp2"] = round(pnl_tp2, 2)

            max_profit = max((tp - price) * 100.0 for tp in tp_prices)
            pl_projection["max_profit_est"] = round(max_profit * contracts_possible, 2)
        if pnl_block["at_stop"] is not None:
            loss = pnl_block["at_stop"] * contracts_possible
            pl_projection["max_loss_est"] = round(abs(loss), 2)

    if pl_projection["max_loss_est"] is None and pl_projection["risk_budget"]:
        pl_projection["max_loss_est"] = round(float(pl_projection["risk_budget"]), 2)

    enriched["pnl"] = pnl_block
    enriched["pl_projection"] = pl_projection
    if risk_per_contract is not None:
        enriched.setdefault("cost_basis", {})["per_contract"] = risk_per_contract
    return enriched

@gpt.post("/contracts", summary="Return ranked option contracts for a symbol")
async def gpt_contracts(
    request_payload: ContractsRequest,
    _: AuthedUser = Depends(require_api_key),
) -> Dict[str, Any]:
    symbol = request_payload.symbol.upper().strip()
    if not symbol:
        raise HTTPException(status_code=400, detail="Symbol is required")

    try:
        chain = await fetch_option_chain_cached(symbol, request_payload.expiry)
    except TradierNotConfiguredError as exc:
        raise HTTPException(status_code=503, detail="Tradier integration is not configured") from exc
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Tradier chain fetch failed for %s", symbol)
        raise HTTPException(status_code=502, detail=f"Option chain unavailable for {symbol}") from exc

    if chain.empty:
        raise HTTPException(status_code=502, detail=f"Option chain unavailable for {symbol}")

    # collect quotes for symbols present in the chain
    option_symbols = [str(sym) for sym in chain["symbol"].dropna().tolist()]
    quotes = await fetch_option_quotes(option_symbols)

    style = _normalize_contract_style(request_payload.style)
    filters = _prepare_contract_filters(request_payload, style)
    side = _infer_contract_side(request_payload.side, request_payload.bias)
    risk_amount = request_payload.risk_amount or request_payload.max_price or 100.0

    candidates = _screen_contracts(chain, quotes, symbol=symbol, style=style, side=side, filters=filters)

    relaxed = False
    if not candidates:
        relaxed = True
        filters_delta = filters.copy()
        filters_delta["min_delta"] = _clamp(float(filters_delta.get("min_delta", 0.0)) - 0.05, 0.0, 1.0)
        filters_delta["max_delta"] = _clamp(float(filters_delta.get("max_delta", 1.0)) + 0.05, 0.0, 1.0)
        candidates = _screen_contracts(chain, quotes, symbol=symbol, style=style, side=side, filters=filters_delta)
        if not candidates:
            filters_dte = filters_delta.copy()
            filters_dte["min_dte"] = max(0, int(filters_dte.get("min_dte", 0)) - 2)
            filters_dte["max_dte"] = int(filters_dte.get("max_dte", 365)) + 2
            candidates = _screen_contracts(chain, quotes, symbol=symbol, style=style, side=side, filters=filters_dte)
            filters = filters_dte
        else:
            filters = filters_delta
    else:
        filters = filters

    plan_anchor = getattr(request_payload, "plan_anchor", None)
    best = [_enrich_contract_with_plan(contract, plan_anchor, risk_amount) for contract in candidates[:3]]
    alternatives = [_enrich_contract_with_plan(contract, plan_anchor, risk_amount) for contract in candidates[3:10]]

    # Compact table view for UI rendering
    table_rows: List[Dict[str, Any]] = []
    for row in best[:6]:
        try:
            label = row.get("label") or row.get("symbol") or ""
            # Preserve a compact, ordered shape: label, dte, strike, price, bid, ask, delta, theta, iv, spread_pct, oi, liquidity_score
            price_val = row.get("price") or row.get("mid") or row.get("mark")
            table_rows.append({
                "label": label,
                "dte": row.get("dte"),
                "strike": row.get("strike"),
                "price": price_val,
                "bid": row.get("bid"),
                "ask": row.get("ask"),
                "delta": row.get("delta"),
                "theta": row.get("theta"),
                "iv": row.get("implied_volatility") or row.get("iv"),
                "spread_pct": row.get("spread_pct"),
                "oi": row.get("open_interest") or row.get("oi"),
                "liquidity_score": row.get("tradeability") or row.get("liquidity_score"),
            })
        except Exception:
            continue

    return {
        "symbol": symbol,
        "side": side,
        "style": style,
        "risk_amount": risk_amount,
        "filters": filters,
        "relaxed_filters": relaxed,
        "best": best,
        "alternatives": alternatives,
        "table": table_rows,
    }


@gpt.post("/chart-url", summary="Build a canonical chart URL from params", response_model=ChartLinks)
async def gpt_chart_url(payload: ChartParams, request: Request) -> ChartLinks:
    """Validate and return a canonical charts/html URL.

    Validation rules:
    - Required fields: symbol, interval, direction, entry, stop, tp
    - Monotonic order by direction
    - R:R gate (1.5 for index intraday; else 1.2)
    - Min TP distance if ATR provided (0.3×ATR intraday; 0.6×ATR swing), unless confluence label at TP exists
    - Whitelist interval and view tokens
    - Percent-encode free-text fields (levels, notes, strategy)
    """

    # Collect raw data, preserving extras
    data: Dict[str, Any] = dict(payload.model_extra or {})
    raw_symbol = str(payload.symbol or "").strip()
    raw_interval = str(payload.interval or "").strip()
    direction = (str(data.get("direction") or "").strip().lower())
    entry = data.get("entry")
    stop = data.get("stop")
    tp_csv = data.get("tp")

    # 1) Required fields
    def _missing(field: str):
        raise HTTPException(status_code=422, detail={"error": f"missing field {field}"})

    if not raw_symbol:
        _missing("symbol")
    if not raw_interval:
        _missing("interval")
    if not direction:
        _missing("direction")
    if entry is None:
        _missing("entry")
    if stop is None:
        _missing("stop")
    if not tp_csv:
        _missing("tp")

    try:
        entry_f = float(entry)
        stop_f = float(stop)
        tp1_f = float(str(tp_csv).split(",")[0].strip())
    except Exception:
        raise HTTPException(status_code=422, detail={"error": "entry/stop/tp must be numeric"})

    # 6) Whitelist interval + view
    # Use charts_api.normalize_interval for canonical tokens: {'1m','5m','15m','1h','d'}
    try:
        interval_norm = normalize_interval(raw_interval)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail={"error": str(exc)})

    allowed_intervals = {"1m", "5m", "15m", "1h", "d"}
    if interval_norm not in allowed_intervals:
        raise HTTPException(status_code=422, detail={"error": f"interval '{raw_interval}' not allowed"})

    view = str(data.get("view") or "6M").strip()
    allowed_views = {"1d", "5d", "1M", "3M", "6M", "1Y"}
    if view not in allowed_views:
        # default to 6M if out of set
        view = "6M"
    data["view"] = view

    # 2) Monotonic order
    if direction == "long":
        if not (stop_f < entry_f < tp1_f):
            raise HTTPException(status_code=422, detail={"error": "order invalid for long (stop < entry < TP1)"})
    elif direction == "short":
        if not (stop_f > entry_f > tp1_f):
            raise HTTPException(status_code=422, detail={"error": "order invalid for short (stop > entry > TP1)"})
    else:
        raise HTTPException(status_code=422, detail={"error": "direction must be 'long' or 'short'"})

    # 3) R:R gate
    def _rr(e: float, s: float, t: float, d: str) -> float:
        risk = (e - s) if d == "long" else (s - e)
        reward = (t - e) if d == "long" else (e - t)
        if risk <= 0:
            return 0.0
        return reward / risk

    is_index = raw_symbol.upper() in {"SPY", "QQQ", "IWM"}
    is_intraday = interval_norm in {"1m", "5m", "15m"}
    min_rr = 1.5 if is_index and is_intraday else 1.2
    rr_val = _rr(entry_f, stop_f, tp1_f, direction)
    if rr_val < min_rr:
        raise HTTPException(status_code=422, detail={"error": f"R:R {rr_val:.2f} < {min_rr:.1f}"})

    # 4) Min TP distance if ATR provided
    atr_val = data.get("atr14") or data.get("atr")
    if atr_val is not None:
        try:
            atr_f = float(atr_val)
        except Exception:
            atr_f = None
        if atr_f and atr_f > 0:
            k = 0.3 if interval_norm in {"1m", "5m", "15m", "1h"} else 0.6
            min_tp = entry_f + k * atr_f if direction == "long" else entry_f - k * atr_f
            ok = (tp1_f >= min_tp) if direction == "long" else (tp1_f <= min_tp)
            if not ok:
                # allow if confluence label exists at TP price in `levels` as price|label;...
                levels = str(data.get("levels") or "")
                has_confluence = False
                if levels:
                    for chunk in levels.split(";"):
                        parts = [p.strip() for p in chunk.split("|")]
                        if not parts:
                            continue
                        try:
                            price = float(parts[0])
                        except Exception:
                            continue
                        if len(parts) >= 2 and abs(price - tp1_f) <= max(1e-4, 0.01):
                            has_confluence = True
                            break
                if not has_confluence:
                    raise HTTPException(status_code=422, detail={"error": "TP1 too close; fails ATR gate"})

    # Build base URL
    # If `BASE_URL`/`CHART_BASE_URL` is configured, use it verbatim.
    # Otherwise default to the local charts HTML renderer.
    settings = get_settings()
    configured_base = (settings.chart_base_url or "").strip()
    if configured_base:
        base = configured_base.rstrip("/")
    else:
        base = f"{str(request.base_url).rstrip('/')}/charts/html"

    # Assemble query with normalized fields
    data["symbol"] = _normalize_chart_symbol(raw_symbol)
    # For charts/html we prefer canonical interval tokens (1m/5m/15m/1h/d)
    data["interval"] = interval_norm
    data["direction"] = direction
    data["entry"] = f"{entry_f:.2f}"
    data["stop"] = f"{stop_f:.2f}"
    data["tp"] = str(tp_csv)

    # Whitelist keys and encode
    query: Dict[str, str] = {}
    for key, value in data.items():
        if key not in ALLOWED_CHART_KEYS or value is None:
            continue
        if isinstance(value, (list, tuple)):
            value = ",".join(str(item) for item in value if item is not None)
        query[key] = str(value)

    # Percent-encode strategy/levels/notes explicitly
    if "strategy" in query:
        query["strategy"] = quote(query["strategy"], safe="|;:,.+-_() ")
    if "levels" in data and data.get("levels"):
        query["levels"] = quote(str(data["levels"]), safe="|;:,.+-_() ")
    if "notes" in data and data.get("notes"):
        query["notes"] = quote(str(data["notes"])[:140], safe="|;:,.+-_() ")

    encoded = urlencode(query, doseq=False, safe=",", quote_via=quote)
    url = f"{base}?{encoded}" if encoded else base
    return ChartLinks(interactive=url)


@gpt.get("/context/{symbol}", summary="Return recent market context for a ticker")
async def gpt_context(
    symbol: str,
    interval: str = Query("1m"),
    lookback: int = Query(300, ge=50, le=1000),
    user: AuthedUser = Depends(require_api_key),
) -> Dict[str, Any]:
    try:
        interval_normalized = normalize_interval(interval)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    frame = get_candles(symbol, interval_normalized, lookback=lookback)
    if frame.empty:
        raise HTTPException(status_code=502, detail=f"No market data available for {symbol.upper()} ({interval_normalized}).")

    df = frame.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    history = df.set_index("time")
    key_levels = _extract_key_levels(history)
    snapshot = _build_market_snapshot(history, key_levels)

    bars = []
    for _, row in df.iterrows():
        ts = pd.Timestamp(row["time"])
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        bars.append(
            {
                "time": ts.isoformat(),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": _safe_number(row.get("volume")) or 0.0,
            }
        )

    ema9_series = ema(history["close"], 9) if len(history) >= 9 else pd.Series(dtype=float)
    ema20_series = ema(history["close"], 20) if len(history) >= 20 else pd.Series(dtype=float)
    ema50_series = ema(history["close"], 50) if len(history) >= 50 else pd.Series(dtype=float)
    vwap_series = vwap(history["close"], history["volume"])
    atr_series = atr(history["high"], history["low"], history["close"], 14)
    adx_series = adx(history["high"], history["low"], history["close"], 14)

    indicators = {
        "ema9": _series_points(ema9_series),
        "ema20": _series_points(ema20_series),
        "ema50": _series_points(ema50_series),
        "vwap": _series_points(vwap_series),
        "atr14": _series_points(atr_series),
        "adx14": _series_points(adx_series),
    }

    benchmark_history: pd.DataFrame | None = None
    if symbol.upper() != "SPY":
        try:
            bench_frame = get_candles("SPY", interval_normalized, lookback=lookback)
            bench_frame = bench_frame.copy()
            bench_frame["time"] = pd.to_datetime(bench_frame["time"], utc=True)
            benchmark_history = bench_frame.set_index("time")
        except HTTPException:
            benchmark_history = None

    chain_df: pd.DataFrame | None = None
    polygon_bundle: Dict[str, Any] | None = None
    settings = get_settings()
    if settings.polygon_api_key:
        chain_df = await fetch_polygon_option_chain(symbol)
        if chain_df is not None and not chain_df.empty:
            polygon_bundle = summarize_polygon_chain(chain_df, rules=None, top_n=3)

    enhancements = compute_context_overlays(
        history,
        symbol=symbol.upper(),
        interval=interval_normalized,
        benchmark_history=benchmark_history,
        options_chain=chain_df,
    )

    response: Dict[str, Any] = {
        "symbol": symbol.upper(),
        "interval": interval_normalized,
        "lookback": lookback,
        "bars": bars,
        "indicators": indicators,
        "key_levels": key_levels,
        "snapshot": snapshot,
    }
    response.update(enhancements)
    response["context_overlays"] = enhancements
    if polygon_bundle:
        response["options"] = polygon_bundle
    level_tokens = _extract_levels_for_chart(key_levels)

    chart_params = {
        "symbol": symbol.upper(),
        "interval": interval_normalized,
        "ema": "9,20,50",
        "view": "fit",
        "title": f"{symbol.upper()} {interval_normalized}",
        "vwap": "1",
        "theme": "dark",
    }
    if level_tokens:
        chart_params["levels"] = ",".join(level_tokens)
    response["charts"] = {"params": {key: str(value) for key, value in chart_params.items()}}
    return response


@gpt.get("/widgets/{kind}", summary="Generate lightweight dashboard widgets")
async def gpt_widget(kind: str, symbol: str | None = None, user: AuthedUser = Depends(require_api_key)) -> Dict[str, Any]:
    if kind == "ticker_wedge" and symbol:
        return {
            "type": "ticker_wedge",
            "symbol": symbol.upper(),
            "pattern": "rising_wedge",
            "confidence": 0.72,
            "levels": {"support": 98.4, "resistance": 102.6},
        }
    raise HTTPException(status_code=404, detail="Unknown widget kind or missing params")


# Register GPT endpoints with the application
app.include_router(tv_api)
app.include_router(gpt)
app.include_router(charts_router)
app.include_router(gpt_sentiment_router)


# ---------------------------------------------------------------------------
# Platform health endpoints
# ---------------------------------------------------------------------------


@app.get("/healthz", summary="Readiness probe used by Railway")
async def healthz() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/", summary="Service metadata")
async def root() -> Dict[str, Any]:
    return {
        "name": "trading-coach-gpt-backend",
        "description": "Backend endpoints intended for a custom GPT Action.",
        "routes": {
            "scan": "/gpt/scan",
            "context": "/gpt/context/{symbol}",
            "widgets": "/gpt/widgets/{kind}",
            "charts_html": "/charts/html",
            "charts_png": "/charts/png",
            "health": "/healthz",
        },
    }
