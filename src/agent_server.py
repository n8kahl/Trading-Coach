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
from typing import Any, Dict, List

import httpx
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import get_settings
from .calculations import atr
from .charts_api import build_chart_url, router as charts_router
from .scanner import scan_market
from .tradier import TradierNotConfiguredError, select_tradier_contract


logger = logging.getLogger(__name__)

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


def _plan_trade_levels(
    history: pd.DataFrame,
    entry: float,
    direction: str,
    atr_hint: float | None,
    key_levels: Dict[str, float],
) -> tuple[float, float, float, float, float, float]:
    """Compute stop/target that respect ATR and structural levels."""
    atr_series = atr(history["high"], history["low"], history["close"], period=14)
    atr_value = float(atr_hint or 0.0)
    if not atr_series.empty and not np.isnan(atr_series.iloc[-1]):
        atr_value = float(atr_series.iloc[-1])
    if atr_value <= 0:
        atr_value = max(entry * 0.01, 0.25)

    session_high = key_levels.get("session_high")
    session_low = key_levels.get("session_low")
    opening_range_high = key_levels.get("opening_range_high")
    opening_range_low = key_levels.get("opening_range_low")
    prev_close = key_levels.get("prev_close")
    prev_high = key_levels.get("prev_high")
    prev_low = key_levels.get("prev_low")
    gap_fill = key_levels.get("gap_fill")

    if direction == "long":
        stop_candidates = [
            level for level in [opening_range_low, prev_close, prev_low, session_low] if level and level < entry
        ]
        stop = max(stop_candidates) if stop_candidates else entry - atr_value * 1.1

        target_candidates = [
            level for level in [gap_fill, opening_range_high, session_high, prev_high] if level and level > entry
        ]
        target = min(target_candidates) if target_candidates else entry + atr_value * 1.8
    else:
        stop_candidates = [
            level for level in [opening_range_high, prev_close, prev_high, session_high] if level and level > entry
        ]
        stop = min(stop_candidates) if stop_candidates else entry + atr_value * 1.1

        target_candidates = [
            level for level in [gap_fill, opening_range_low, session_low, prev_low] if level and level < entry
        ]
        target = max(target_candidates) if target_candidates else entry - atr_value * 1.8

    if direction == "long" and stop >= entry:
        stop = entry - atr_value
    if direction == "short" and stop <= entry:
        stop = entry + atr_value

    risk = abs(entry - stop)
    reward = abs(target - entry)
    risk_reward = reward / risk if risk else 0.0

    if direction == "long":
        target_secondary = target + max(atr_value * 1.2, risk if risk else atr_value)
        if target_secondary <= target:
            target_secondary = target + atr_value
    else:
        target_secondary = target - max(atr_value * 1.2, risk if risk else atr_value)
        if target_secondary >= target:
            target_secondary = target - atr_value

    return entry, stop, target, target_secondary, atr_value, risk_reward
# Strategy utilities ---------------------------------------------------------

def _direction_for_strategy(strategy_id: str) -> str:
    sid = strategy_id.lower()
    if "short" in sid or "put" in sid:
        return "short"
    return "long"


def _indicators_for_strategy(strategy_id: str) -> List[str]:
    sid = strategy_id.lower()
    if "vwap" in sid:
        return ["VWAP", "EMA20"]
    if "orb" in sid:
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
            snapshot = await select_tradier_contract("SPY")
            if snapshot is None:
                return {"status": "unavailable"}
            return {"status": "ok", "symbol": snapshot.get("symbol"), "expiration": snapshot.get("expiration")}
        except TradierNotConfiguredError:
            return {"status": "missing"}
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Tradier health check failed: %s", exc)
            return {"status": "error", "error": str(exc)}

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

    market_data = await _collect_market_data(universe.tickers, timeframe=data_timeframe)
    if not market_data:
        raise HTTPException(status_code=502, detail="No market data available for the requested tickers.")
    signals = await scan_market(universe.tickers, market_data)

    contract_suggestions: Dict[str, Dict[str, Any] | None] = {}
    unique_symbols = sorted({signal.symbol for signal in signals})
    if unique_symbols:
        settings = get_settings()
        if not settings.tradier_token:
            logger.info("Tradier token not configured; skipping contract suggestions.")
        else:
            try:
                tasks = [select_tradier_contract(symbol) for symbol in unique_symbols]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for symbol, result in zip(unique_symbols, results):
                    if isinstance(result, Exception):
                        logger.warning("Tradier contract lookup failed for %s: %s", symbol, result)
                        contract_suggestions[symbol] = None
                    else:
                        contract_suggestions[symbol] = result
            except Exception as exc:  # pragma: no cover - safety net
                logger.warning("Tradier integration error: %s", exc)

    payload: List[Dict[str, Any]] = []
    for signal in signals:
        style = _style_for_strategy(signal.strategy_id)
        if style_filter and style_filter != style:
            continue
        history = market_data[signal.symbol]
        latest_row = history.iloc[-1]
        entry_price = float(latest_row["close"])
        key_levels = _extract_key_levels(history)
        direction = _direction_for_strategy(signal.strategy_id)
        entry, stop, target, target_secondary, atr_value, risk_reward = _plan_trade_levels(
            history, entry_price, direction, signal.features.get("atr"), key_levels
        )
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
        interactive_url = build_chart_url(
            base_url,
            "html",
            signal.symbol.upper(),
            entry=entry,
            stop=stop,
            tps=[target, target_secondary],
            emas=ema_spans,
            interval=interval,
            title=title,
            renderer_params={
                "direction": direction,
                "strategy": signal.strategy_id,
                "atr": f"{atr_value:.2f}",
                "risk_reward": f"{risk_reward:.2f}",
            },
            view=_view_for_style(style),
        )
        payload.append(
            {
                "symbol": signal.symbol,
                "style": style,
                "strategy_id": signal.strategy_id,
                "description": signal.description,
                "score": signal.score,
                "contract_suggestion": contract_suggestions.get(signal.symbol),
                "direction": direction,
        "levels": {
            "entry": round(entry, 2),
            "stop": round(stop, 2),
            "target": round(target, 2),
            "target_secondary": round(target_secondary, 2),
            "risk_reward": round(risk_reward, 2),
        },
                "key_levels": key_levels,
                "charts": {
                    "interactive": interactive_url,
                },
                "features": {**signal.features, "atr": atr_value, "key_levels": key_levels},
            }
        )

    logger.info("scan universe=%s user=%s results=%d", universe.tickers, user.user_id, len(payload))
    return payload


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
            "widgets": "/gpt/widgets/{kind}",
            "charts_html": "/charts/html",
            "charts_png": "/charts/png",
            "health": "/healthz",
        },
    }
