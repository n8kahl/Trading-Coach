"""Trading Coach backend tailored for GPT Actions integrations.

The service now focuses on a lean surface area that lets a custom GPT pull
ranked setups (with richer level-aware targets) and render interactive charts
driven by the same OHLCV data. Legacy endpoints for watchlists, notes, and
trade-following have been removed to keep the API aligned with the coaching
workflow.
"""

from __future__ import annotations

import asyncio
import math
import logging
from typing import Any, Dict, List, Optional

import httpx
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydantic import ConfigDict
from urllib.parse import urlencode, quote

from .config import get_settings
from .calculations import atr, ema, bollinger_bands, keltner_channels, adx, vwap
from .charts_api import router as charts_router, get_candles, normalize_interval
from .scanner import scan_market
from .tradier import TradierNotConfiguredError, fetch_option_chain, select_tradier_contract
from .polygon_options import fetch_polygon_option_chain, summarize_polygon_chain
from .context_overlays import compute_context_overlays


logger = logging.getLogger(__name__)

ALLOWED_CHART_KEYS = {
    "symbol",
    "interval",
    "view",
    "ema",
    "vwap",
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
}


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


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

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
    # Prefer Polygon
    poly = await _fetch_polygon_ohlcv(symbol, timeframe)
    if poly is not None and not poly.empty and not _is_stale_frame(poly, timeframe):
        return poly
    if poly is not None and not poly.empty:
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
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {"interval": interval, "range": range_span, "includePrePost": "false"}

    timeout = httpx.Timeout(6.0, connect=3.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("Yahoo Finance fetch failed for %s: %s", symbol, exc)
            return None

    payload = response.json()
    try:
        chart = payload["chart"]
        if chart.get("error"):
            raise ValueError(chart["error"])
        result = chart["result"][0]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        logger.warning("Unexpected Yahoo Finance payload for %s: %s", symbol, exc)
        return None

    timestamps = result.get("timestamp")
    if not timestamps:
        logger.warning("Yahoo Finance returned no timestamps for %s", symbol)
        return None

    quote = result["indicators"]["quote"][0]
    o = quote.get("open")
    h = quote.get("high")
    l = quote.get("low")
    c = quote.get("close")
    v = quote.get("volume")
    if not all([o, h, l, c, v]):
        logger.warning("Incomplete OHLCV data for %s", symbol)
        return None

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
    return frame


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
    mapping = {"scalp": "1m", "intraday": "5m", "swing": "1h", "leap": "d"}
    return mapping.get(normalized, "5m")


def _view_for_style(style: str | None) -> str:
    normalized = _normalize_style(style) or ""
    mapping = {"scalp": "30m", "intraday": "1d", "swing": "5d", "leap": "fit"}
    return mapping.get(normalized, "fit")




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
        }
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

        chain = polygon_chains.get(signal.symbol)
        enhancements = compute_context_overlays(
            history,
            symbol=signal.symbol,
            interval=interval,
            benchmark_history=benchmark_history,
            options_chain=chain,
        )

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
                "data": {
                    "bars": f"{base_url}/gpt/context/{signal.symbol}?interval={interval}&lookback=300"
                },
                "context_overlays": enhancements,
                **({"options": polygon_bundle} if polygon_bundle else {}),
            }
        )

    logger.info("scan universe=%s user=%s results=%d", universe.tickers, user.user_id, len(payload))
    return payload


@gpt.post("/chart-url", summary="Build a canonical chart URL from params", response_model=ChartLinks)
async def gpt_chart_url(payload: ChartParams, request: Request) -> ChartLinks:
    settings = get_settings()

    base_url = (settings.chart_base_url or str(request.base_url)).rstrip("/")
    if not base_url:
        raise HTTPException(status_code=500, detail="Base URL is not configured")

    data: Dict[str, Any] = dict(payload.model_extra or {})
    data["symbol"] = payload.symbol
    data["interval"] = payload.interval

    query: Dict[str, str] = {}
    for key, value in data.items():
        if key not in ALLOWED_CHART_KEYS or value is None:
            continue
        if isinstance(value, (list, tuple)):
            value = ",".join(str(item) for item in value if item is not None)
        query[key] = str(value)

    if "symbol" not in query or "interval" not in query:
        raise HTTPException(status_code=400, detail="Required: symbol, interval")

    encoded = urlencode(query, doseq=False, safe=",", quote_via=quote)
    url = f"{base_url}/charts/html?{encoded}" if encoded else f"{base_url}/charts/html"
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
    chart_params = {
        "symbol": symbol.upper(),
        "interval": interval_normalized,
        "ema": "9,20,50",
        "view": "fit",
        "title": f"{symbol.upper()} {interval_normalized}",
    }
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
app.include_router(gpt)
app.include_router(charts_router)


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
