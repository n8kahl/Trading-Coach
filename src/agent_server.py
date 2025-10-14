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
import os
import time
import uuid
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Literal
import copy

import httpx
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, AliasChoices
from pydantic import ConfigDict
from urllib.parse import urlencode, quote, urlsplit, urlunsplit, parse_qsl
from fastapi.responses import StreamingResponse, PlainTextResponse

from .config import get_settings
from .calculations import atr, ema, bollinger_bands, keltner_channels, adx, vwap
from .charts_api import router as charts_router, get_candles, normalize_interval
from .gpt_sentiment import router as gpt_sentiment_router
from .scanner import (
    scan_market,
    _apply_tp_logic,
    _base_targets_for_style,
    _normalize_trade_style,
    _runner_config,
)
from .strategy_library import (
    normalize_style_input,
    public_style,
    strategy_public_category,
)
from .tradier import (
    TradierNotConfiguredError,
    fetch_option_chain,
    fetch_option_chain_cached,
    fetch_option_quotes,
    select_tradier_contract,
)
from .polygon_options import (
    fetch_polygon_option_chain,
    fetch_polygon_option_chain_asof,
    summarize_polygon_chain,
)
from .context_overlays import compute_context_overlays
from .db import (
    ensure_schema as ensure_db_schema,
    fetch_idea_snapshot as db_fetch_idea_snapshot,
    store_idea_snapshot as db_store_idea_snapshot,
)
from .data_sources import fetch_polygon_ohlcv, last_price_asof
from .symbol_streamer import SymbolStreamCoordinator
from .live_plan_engine import LivePlanEngine
from .statistics import get_style_stats
from zoneinfo import ZoneInfo

from .market_clock import MarketClock
from .app.services import session_now, parse_session_as_of, build_chart_url
from .app.engine import TargetEngineResult, build_structured_plan
from .app.engine.context import build_context as build_context_block
from .app.engine.evolve import evolve_plan
from .app.engine.events import apply_event_gating
from .app.engine.options_select import best_contract_example
from .app.engine.options_select import build_example_leg, score_contract
from .app.engine.risk import risk_model_payload
from .app.engine.scoring import overall_confidence, quality_grade, score_components
from .app.routers.validators import canonical_chart_url

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

_FUTURES_PROXY_MAP: Dict[str, str] = {
    "es_proxy": "SPY",
    "nq_proxy": "QQQ",
    "ym_proxy": "DIA",
    "rty_proxy": "IWM",
    "vix": "CBOE:VIX",
}

_FUTURES_CACHE: Dict[str, Any] = {"data": None, "ts": 0.0}

_MARKET_CLOCK = MarketClock()


def _session_block() -> Dict[str, Any]:
    """Return a session dictionary safe for inclusion in responses."""
    try:
        session = session_now()
        return session.to_dict()
    except Exception:  # pragma: no cover - defensive logging
        logger.exception("session_now failed; defaulting to closed session banner")
        return {
            "status": "closed",
            "as_of": "",
            "next_open": "",
            "tz": "America/New_York",
            "banner": "Session status unavailable",
        }


def _session_asof_timestamp(session: Dict[str, Any]) -> Optional[pd.Timestamp]:
    dt = parse_session_as_of(session)
    if dt is None:
        return None
    ts = pd.Timestamp(dt)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _coalesce_events(block: Any) -> List[Dict[str, Any]]:
    if block is None:
        return []
    if isinstance(block, list):
        return [dict(item) for item in block if isinstance(item, dict)]
    if isinstance(block, dict):
        collected: List[Dict[str, Any]] = []
        for key in ("items", "events", "upcoming", "data"):
            value = block.get(key)
            if isinstance(value, list):
                collected.extend([dict(item) for item in value if isinstance(item, dict)])
        if not collected:
            collected.append({k: v for k, v in block.items() if k in {"severity", "minutes_to_event", "label", "type"}})
        return [item for item in collected if item]
    return []


def _plan_stream_key(symbol: str, plan_id: str) -> str:
    return f"{symbol.upper()}::{plan_id}"


def _transform_plan_event(symbol: str, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    plan_id = event.get("plan_id")
    if not plan_id:
        return None
    payload: Dict[str, Any] = {
        "symbol": symbol.upper(),
        "plan_id": plan_id,
    }
    event_type = event.get("t") or "note"
    if event_type == "price":
        payload.update({
            "type": "price",
            "price": event.get("price"),
            "ts": event.get("ts"),
        })
        return payload
    if event_type == "plan_update":
        payload.update({
            "type": "update",
            "setup": event.get("setup"),
            "ts": event.get("ts"),
        })
        return payload
    if event_type == "hit":
        payload.update({
            "type": "hit",
            "hit": event.get("level"),
            "price": event.get("price"),
            "ts": event.get("ts"),
        })
        return payload
    if event_type in {"stop", "invalidate", "reverse"}:
        payload.update({
            "type": event_type,
            "ts": event.get("ts"),
            "price": event.get("price"),
            "reason": event.get("reason"),
        })
        return payload
    if event_type == "plan_delta":
        changes = event.get("changes") or {}
        payload.update({
            "type": "price",
            "changes": changes,
            "raw": event,
        })
        breach = str(changes.get("breach") or "").lower()
        status = str(changes.get("status") or "").lower()
        if breach in {"stop_hit", "tp1_hit", "tp2_hit"}:
            payload["type"] = "hit"
            payload["hit"] = breach
        elif status == "invalidated":
            payload["type"] = "hit"
            payload["hit"] = "invalid"
        if "last_price" in changes:
            payload["price"] = changes["last_price"]
        if "note" in changes:
            payload.setdefault("note", changes.get("note"))
    elif event_type == "plan_full":
        payload.update({
            "type": "replan",
            "plan": event.get("payload"),
        })
    else:
        payload.update({
            "type": "note",
            "event": event,
        })
    return payload


async def _publish_plan_event(symbol: str, event: Dict[str, Any]) -> None:
    plan_id = event.get("plan_id")
    if not plan_id:
        return
    key = _plan_stream_key(symbol, plan_id)
    async with _PLAN_STREAM_LOCK:
        queues = list(_PLAN_STREAM_SUBSCRIBERS.get(key, []))
    if not queues:
        return
    payload = _transform_plan_event(symbol, event) or event
    try:
        message = json.dumps({"symbol": symbol.upper(), "plan_id": plan_id, "event": payload})
    except TypeError:
        return
    for queue in queues:
        try:
            queue.put_nowait(message)
        except asyncio.QueueFull:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            queue.put_nowait(message)


def _market_snapshot_payload() -> Tuple[Dict[str, Any], Dict[str, Any], datetime, bool]:
    snapshot = _MARKET_CLOCK.snapshot()
    as_of_dt = _MARKET_CLOCK.last_rth_close()
    frozen = snapshot.status != "open"
    market_payload = {
        "status": snapshot.status,
        "session": snapshot.session,
        "now_et": snapshot.now_et.isoformat(),
        "next_open_et": snapshot.next_open_et.isoformat() if snapshot.next_open_et else None,
        "next_close_et": snapshot.next_close_et.isoformat() if snapshot.next_close_et else None,
    }
    data_payload = {
        "as_of_ts": int(as_of_dt.timestamp() * 1000),
        "frozen": frozen,
        "ok": True,
    }
    return market_payload, data_payload, as_of_dt, not frozen


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
    session: Dict[str, Any] | None = None


app = FastAPI(
    title="Trading Coach GPT Backend",
    description="Backend utilities for a custom GPT that offers trading guidance.",
    version="3.0.0",
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

APP_STATIC_DIR = STATIC_ROOT / "app"
if APP_STATIC_DIR.exists():
    app.mount("/app", StaticFiles(directory=str(APP_STATIC_DIR), html=True), name="app")


@app.on_event("startup")
async def _startup_tasks() -> None:
    global _IDEA_PERSISTENCE_ENABLED, _SYMBOL_STREAM_COORDINATOR
    persisted = await ensure_db_schema()
    if persisted:
        _IDEA_PERSISTENCE_ENABLED = True
        logger.info("Persistent idea snapshot storage enabled")
    else:
        _IDEA_PERSISTENCE_ENABLED = False
        logger.info("Idea snapshots will be cached in-memory (database unavailable or not configured)")
    _SYMBOL_STREAM_COORDINATOR = SymbolStreamCoordinator(_symbol_stream_emit)
    logger.info("Live symbol streamer initialized")


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
        description="Optional style filter: 'scalp', 'intraday', 'swing', or 'leaps'.",
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
    plan_id: str | None = None


class PlanResponse(BaseModel):
    plan_id: str | None = None
    version: int | None = None
    trade_detail: str | None = None
    idea_url: str | None = None
    warnings: List[str] | None = None
    planning_context: str | None = None
    symbol: str
    style: str | None = None
    bias: str | None = None
    setup: str | None = None
    entry: float | None = None
    stop: float | None = None
    targets: List[float] | None = None
    target_meta: List[Dict[str, Any]] | None = None
    rr_to_t1: float | None = None
    confidence: float | None = None
    confidence_factors: List[str] | None = None
    notes: str | None = None
    relevant_levels: Dict[str, Any] | None = None
    expected_move_basis: str | None = None
    sentiment: Dict[str, Any] | None = None
    events: Dict[str, Any] | None = None
    earnings: Dict[str, Any] | None = None
    charts_params: Dict[str, Any] | None = None
    chart_url: str | None = None
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
    runner: Dict[str, Any] | None = None
    updated_from_version: int | None = None
    update_reason: str | None = None
    market: Dict[str, Any] | None = None
    data: Dict[str, Any] | None = None
    session: Dict[str, Any] | None = None
    target_profile: Dict[str, Any] | None = None
    structured_plan: Dict[str, Any] | None = None


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
    trade_detail: str
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
    snapshots: List[Dict[str, Any]]
    volatility_regime: Dict[str, Any]
    sentiment: Dict[str, Any] | None = None
    events: Dict[str, Any] | None = None
    earnings: Dict[str, Any] | None = None
    summary: Dict[str, Any] | None = None
    decimals: int | None = None
    data_quality: Dict[str, Any] | None = None
    contexts: List[Dict[str, Any]] | None = None  # backward compatibility
    session: Dict[str, Any] | None = None


class SessionMeta(BaseModel):
    status: Literal["open", "closed"]
    as_of: str
    next_open: Optional[str] = None
    tz: Optional[str] = None
    banner: Optional[str] = None


class EntryMeta(BaseModel):
    type: Literal["break", "retest", "limit"] = "limit"
    level: float


class OptionExample(BaseModel):
    type: Literal["call", "put"]
    expiry: str
    delta: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    spread_pct: Optional[float] = None
    spread_stability: Optional[float] = None
    oi: Optional[int] = None
    volume: Optional[int] = None
    iv_percentile: Optional[float] = None
    composite_score: Optional[float] = None
    tradeability: Optional[int] = None


class OptionsBlockMeta(BaseModel):
    style_horizon_applied: Optional[str] = None
    dte_window: Optional[str] = None
    example: Optional[OptionExample] = None
    note: Optional[str] = None


class SetupMeta(BaseModel):
    plan_id: str
    symbol: str
    style: Literal["scalp", "intraday", "swing", "leaps"]
    direction: Literal["long", "short"]
    version: Optional[int] = None
    entry: EntryMeta
    invalid: Optional[float] = None
    stop: float
    targets: List[float]
    probabilities: Optional[Dict[str, float]] = None
    probability_components: Optional[Dict[str, float]] = None
    trade_quality_score: Optional[str] = None
    runner: Optional[Dict[str, Any]] = None
    confluence: Optional[List[str]] = None
    key_levels: Optional[Dict[str, Any]] = None
    mtf_analysis: Optional[Dict[str, Any]] = None
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: Optional[str] = None
    options: Optional[OptionsBlockMeta] = None
    context: Optional[Dict[str, Any]] = None
    em_used: Optional[float] = None
    atr_used: Optional[float] = None
    style_horizon_applied: Optional[str] = None
    risk_model: Optional[Dict[str, Any]] = None
    chart_url: str
    as_of: str


class ExecResponse(BaseModel):
    ok: bool = True
    session: SessionMeta
    count: int
    setups: List[SetupMeta]


class ExecRequest(BaseModel):
    symbols: Optional[List[str]] = None
    style: Optional[Literal["scalp", "intraday", "swing", "leap", "leaps"]] = None
    limit: Optional[int] = 3
    ui_mode: Optional[Literal["api", "chat"]] = "api"
    include_series: Optional[Literal["none", "compact", "full"]] = "none"
    user_prefs: Optional[Dict[str, Any]] = None


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
    return strategy_public_category(strategy_id)


def _normalize_style(style: str | None) -> str | None:
    if style is None:
        return None
    token = style.strip().lower()
    if not token:
        return None
    if token in {"power_hour", "powerhour", "power-hour", "power hour"}:
        token = "scalp"
    return normalize_style_input(token)


def _canonical_style_token(style: str | None) -> str:
    token = (_normalize_trade_style(style) or "intraday").lower()
    return "leaps" if token == "leap" else token


def _append_query_params(url: str, extra: Dict[str, str]) -> str:
    try:
        parsed = urlsplit(url)
    except Exception:
        return url
    existing = dict(parse_qsl(parsed.query, keep_blank_values=True))
    existing.update({k: v for k, v in extra.items() if v is not None})
    new_query = urlencode(existing, doseq=True)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, new_query, parsed.fragment))


def _market_phase_chicago(now: Optional[datetime] = None) -> str:
    tz = ZoneInfo("America/Chicago")
    dt_now = (now or datetime.now(timezone.utc)).astimezone(tz)
    if dt_now.weekday() >= 5:
        return "closed"
    minutes = dt_now.hour * 60 + dt_now.minute
    reg_open, reg_close = 8 * 60 + 30, 15 * 60  # 08:30–15:00 CT
    pre_open = 3 * 60
    after_close = 19 * 60
    if pre_open <= minutes < reg_open:
        return "premarket"
    if reg_open <= minutes < reg_close:
        return "regular"
    if reg_close <= minutes < after_close:
        return "afterhours"
    return "closed"


async def _fetch_futures_quote(client: httpx.AsyncClient, symbol: str, api_key: str) -> Dict[str, Any]:
    try:
        resp = await client.get(
            "https://finnhub.io/api/v1/quote",
            params={"symbol": symbol, "token": api_key},
            timeout=8.0,
        )
        resp.raise_for_status()
        payload = resp.json() or {}
        last = payload.get("c")
        prev_close = payload.get("pc")
        pct = None
        if isinstance(last, (int, float)) and isinstance(prev_close, (int, float)) and prev_close not in (0, None):
            try:
                pct = (float(last) / float(prev_close)) - 1.0
            except Exception:
                pct = None
        return {
            "symbol": symbol,
            "last": last,
            "percent": pct,
            "time_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "stale": False,
        }
    except Exception:
        return {
            "symbol": symbol,
            "last": None,
            "percent": None,
            "time_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "stale": True,
        }


def _build_trade_detail_url(request: Request, plan_id: str, version: int) -> str:
    headers = request.headers
    scheme = None
    host = None

    forwarded_proto = headers.get("x-forwarded-proto")
    if forwarded_proto:
        scheme = forwarded_proto.split(",")[0].strip()

    forwarded_host = headers.get("x-forwarded-host")
    if forwarded_host:
        host = forwarded_host.split(",")[0].strip()

    forwarded_header = headers.get("forwarded")
    if forwarded_header:
        first_token = forwarded_header.split(",", 1)[0]
        for part in first_token.split(";"):
            name, _, value = part.partition("=")
            if not value:
                continue
            name = name.strip().lower()
            value = value.strip().strip('"')
            if name == "proto" and not scheme:
                scheme = value
            elif name == "host" and not host:
                host = value

    if not scheme:
        scheme = request.url.scheme or "https"

    if not host:
        host = headers.get("host") or request.url.netloc

    # Prefer pretty permalink under /idea/{plan_id} (content-negotiated)
    root = f"{scheme}://{host}" if host else str(request.base_url).rstrip("/")
    path = f"/idea/{plan_id}/{int(version)}"
    base = f"{root.rstrip('/')}{path}"
    logger.info(
        "trade_detail components resolved",
        extra={
            "plan_id": plan_id,
            "version": version,
            "scheme": scheme,
            "host": host,
            "path": path,
            "xfwd_proto": headers.get("x-forwarded-proto"),
            "xfwd_host": headers.get("x-forwarded-host"),
            "forwarded": headers.get("forwarded"),
            "resolved_url": base,
        },
    )
    return base


def _generate_plan_slug(symbol: str, style: Optional[str], direction: Optional[str], snapshot: Dict[str, Any]) -> str:
    """Generate a deterministic, human-readable plan_id slug.

    Format: {symbol-lower}-{style-lower}-{direction-lower}-{YYYY-MM}
    Fallbacks: unknown values become 'unknown'.
    """
    import datetime as _dt

    sym = (symbol or "").strip().lower() or "unknown"
    sty = (style or "").strip().lower() or "unknown"
    drn = (direction or "").strip().lower() or "unknown"
    ts = (snapshot or {}).get("timestamp_utc")
    try:
        when = _dt.datetime.fromisoformat(str(ts).replace("Z", "+00:00")) if ts else _dt.datetime.utcnow()
    except Exception:
        when = _dt.datetime.utcnow()
    stamp = when.strftime("%Y-%m")
    return f"{sym}-{sty}-{drn}-{stamp}"


def _parse_plan_slug(plan_id: str) -> Optional[Dict[str, str]]:
    token = (plan_id or '').strip().lower()
    if not token:
        return None
    parts = [chunk for chunk in token.split('-') if chunk]
    if len(parts) < 3:
        return None
    start_idx = 0
    if parts[0] in {"offline", "online", "live"} and len(parts) >= 4:
        start_idx = 1
    if len(parts) - start_idx < 2:
        return None
    symbol = parts[start_idx].upper()
    style = parts[start_idx + 1].lower() if len(parts) - start_idx >= 2 else "unknown"
    direction_token = parts[start_idx + 2].lower() if len(parts) - start_idx >= 3 else "unknown"
    if direction_token.isdigit():
        direction_token = "unknown"
    return {
        'symbol': symbol,
        'style': style,
        'direction': direction_token,
    }


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
        "target_meta": plan_block.get("target_meta"),
        "rr_to_t1": plan_block.get("risk_reward"),
        "confidence": plan_block.get("confidence"),
        "decimals": decimals,
        "charts_params": charts.get("params") if isinstance(charts, dict) else None,
        "runner": plan_block.get("runner"),
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


def _extract_snapshot_version(snapshot: Dict[str, Any]) -> Optional[int]:
    plan = snapshot.get("plan") or {}
    version = plan.get("version")
    try:
        return int(version)
    except (TypeError, ValueError):
        return None


async def _cache_snapshot(plan_id: str, snapshot: Dict[str, Any]) -> None:
    version = _extract_snapshot_version(snapshot)
    async with _IDEA_LOCK:
        versions = list(_IDEA_STORE.get(plan_id, []))
        if version is not None:
            versions = [snap for snap in versions if _extract_snapshot_version(snap) != version]
        versions.append(snapshot)
        versions.sort(key=lambda snap: _extract_snapshot_version(snap) or 0)
        if len(versions) > _MAX_IDEA_CACHE_VERSIONS:
            versions = versions[-_MAX_IDEA_CACHE_VERSIONS:]
        _IDEA_STORE[plan_id] = versions


async def _get_cached_snapshot(plan_id: str, version: Optional[int]) -> Optional[Dict[str, Any]]:
    async with _IDEA_LOCK:
        versions = _IDEA_STORE.get(plan_id)
        if not versions:
            return None
        if version is None:
            return versions[-1]
        for snap in versions:
            if _extract_snapshot_version(snap) == version:
                return snap
    return None


async def _latest_snapshot_version(plan_id: str) -> Optional[int]:
    async with _IDEA_LOCK:
        versions = _IDEA_STORE.get(plan_id)
        if versions:
            latest = _extract_snapshot_version(versions[-1])
            if latest is not None:
                return latest
    if not _IDEA_PERSISTENCE_ENABLED:
        return None
    snapshot = await db_fetch_idea_snapshot(plan_id)
    if snapshot:
        await _cache_snapshot(plan_id, snapshot)
        return _extract_snapshot_version(snapshot)
    return None


async def _next_plan_version(plan_id: str) -> int:
    latest = await _latest_snapshot_version(plan_id)
    return latest + 1 if latest is not None else 1


async def _store_idea_snapshot(plan_id: str, snapshot: Dict[str, Any]) -> None:
    await _cache_snapshot(plan_id, snapshot)
    persisted = False
    version = _extract_snapshot_version(snapshot)
    if _IDEA_PERSISTENCE_ENABLED:
        if version is not None:
            persisted = await db_store_idea_snapshot(plan_id, version, snapshot)
            if not persisted:
                logger.warning(
                    "idea snapshot persistence failed; continuing with in-memory cache",
                    extra={"plan_id": plan_id, "version": version},
                )
        else:
            logger.warning(
                "idea snapshot missing version; skipping persistence",
                extra={"plan_id": plan_id},
            )
    logger.info(
        "idea snapshot stored",
        extra={
            "plan_id": plan_id,
            "versions": len((_IDEA_STORE.get(plan_id) or [])),
            "snapshot_keys": list(snapshot.keys()),
            "persisted": persisted,
        },
    )
    plan_block = snapshot.get("plan") or {}
    symbol_for_event = (plan_block.get("symbol") or "").upper()
    plan_full_event = None
    try:
        plan_state_event = await _LIVE_PLAN_ENGINE.register_snapshot(snapshot)
    except Exception:
        logger.exception("live plan engine snapshot registration failed", extra={"plan_id": plan_id})
        plan_state_event = None
    if symbol_for_event:
        await _ensure_symbol_stream(symbol_for_event)
        plan_full_event = {
            "t": "plan_full",
            "plan_id": plan_block.get("plan_id"),
            "payload": snapshot,
            "reason": (plan_state_event or {}).get("reason") if isinstance(plan_state_event, dict) else "snapshot_stored",
        }
        await _publish_stream_event(symbol_for_event, plan_full_event)
    if plan_state_event and symbol_for_event:
        await _publish_stream_event(symbol_for_event, plan_state_event)


async def _publish_stream_event(symbol: str, event: Dict[str, Any]) -> None:
    async with _STREAM_LOCK:
        queues = list(_STREAM_SUBSCRIBERS.get(symbol, []))
    try:
        payload = json.dumps({"symbol": symbol, "event": event})
    except TypeError:
        payload = json.dumps({"symbol": symbol, "event": "unserializable"})
    for queue in queues:
        try:
            queue.put_nowait(payload)
        except asyncio.QueueFull:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            queue.put_nowait(payload)
    await _publish_plan_event(symbol, event)


def _output_style_token(style: Optional[str]) -> str:
    token = (_normalize_trade_style(style) or "intraday").lower()
    if token == "leap":
        return "leaps"
    return token


def _clamp_confidence(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, numeric))


def _build_options_block_meta(options_payload: Any, structured_plan: Dict[str, Any]) -> Optional[OptionsBlockMeta]:
    if not isinstance(options_payload, dict):
        return None
    style_applied = options_payload.get("style_horizon_applied") or structured_plan.get("style_horizon_applied")
    filters = options_payload.get("filters") if isinstance(options_payload.get("filters"), dict) else {}
    dte_window = None
    min_dte = filters.get("min_dte") if isinstance(filters, dict) else None
    max_dte = filters.get("max_dte") if isinstance(filters, dict) else None
    if isinstance(min_dte, (int, float)) and isinstance(max_dte, (int, float)):
        dte_window = f"{int(min_dte)}–{int(max_dte)}d"

    example_payload = options_payload.get("example") or options_payload.get("example_leg")
    option_example: Optional[OptionExample] = None
    if isinstance(example_payload, dict):
        ex = dict(example_payload)
        if "expiry" not in ex and "expiration" in ex:
            ex["expiry"] = ex.pop("expiration")
        if "spread_stability" not in ex:
            components = options_payload.get("liquidity_components")
            if isinstance(components, dict):
                ex["spread_stability"] = components.get("spread_stability")
        if "composite_score" not in ex and "score" in ex:
            ex["composite_score"] = ex.get("score")
        if "tradeability" in ex and isinstance(ex["tradeability"], float):
            ex["tradeability"] = int(round(ex["tradeability"]))
        option_example = OptionExample.model_validate(ex)

    return OptionsBlockMeta(
        style_horizon_applied=style_applied,
        dte_window=dte_window,
        example=option_example,
        note=options_payload.get("note"),
    )


def _build_setup_from_plan_response(plan_resp: PlanResponse, session_info: Dict[str, Any]) -> SetupMeta:
    structured = plan_resp.structured_plan or {}
    entry_block = structured.get("entry") or {}
    entry_level = plan_resp.entry if plan_resp.entry is not None else entry_block.get("level")
    if entry_level is None:
        entry_level = (plan_resp.plan or {}).get("entry") if plan_resp.plan else 0.0
    entry_meta = EntryMeta(type=entry_block.get("type", "limit"), level=float(entry_level))

    direction = plan_resp.bias or (plan_resp.plan or {}).get("direction") or structured.get("direction") or "long"
    style_token = _output_style_token(plan_resp.style or structured.get("style") or (plan_resp.plan or {}).get("style"))
    direction_token = str(direction).lower()

    targets = plan_resp.targets or structured.get("targets") or []
    probabilities = None
    profile_dict = plan_resp.target_profile or structured.get("probabilities")
    if isinstance(profile_dict, dict):
        probabilities = {k: float(v) for k, v in profile_dict.items() if isinstance(v, (int, float))}

    options_block = _build_options_block_meta(plan_resp.options, structured)

    stop_value = float(plan_resp.stop) if plan_resp.stop is not None else float(structured.get("stop") or 0.0)
    targets_clean = [float(t) for t in targets]

    chart_url = plan_resp.chart_url or structured.get("chart_url") or plan_resp.trade_detail
    as_of = structured.get("as_of") or session_info.get("as_of")
    as_of_value = as_of or session_info.get("as_of") or ""

    confluence = structured.get("confluence") if isinstance(structured.get("confluence"), list) else None

    plan_identifier = plan_resp.plan_id or (plan_resp.plan or {}).get("plan_id")
    if not plan_identifier:
        plan_identifier = f"{plan_resp.symbol}-{uuid.uuid4().hex[:8]}"
    plan_version = str(plan_resp.version) if plan_resp.version is not None else None

    entry_dict = entry_meta.model_dump()
    if not canonical_chart_url(chart_url):
        chart_url = build_chart_url(
            CHART_BASE,
            symbol=plan_resp.symbol,
            plan_id=plan_identifier,
            as_of=as_of_value,
            entry=entry_dict,
            stop=stop_value,
            targets=targets_clean,
            plan_version=plan_version,
        )

    htf_block = plan_resp.htf if isinstance(plan_resp.htf, dict) else {}
    key_levels = plan_resp.key_levels if isinstance(plan_resp.key_levels, dict) else None
    mtf_analysis = plan_resp.mtf_analysis if isinstance(getattr(plan_resp, "mtf_analysis", None), dict) else None
    if mtf_analysis is None and htf_block:
        mtf_analysis = {"1h": htf_block}

    context_block = build_context_block(plan_resp.symbol, as_of_value)
    confluence_count = len(confluence or [])
    component_inputs = {
        "trend_alignment": 0.6 if str(htf_block.get("bias") or "").lower().startswith(direction_token[:1]) else 0.5,
        "liquidity_structure": max(0.0, min(1.0, 0.5 + confluence_count * 0.04)),
        "momentum_signal": 0.55 if (plan_resp.features or {}).get("momentum") else 0.5,
        "volatility_regime": 0.5,
    }
    components = score_components(component_inputs)
    component_confidence = overall_confidence(components)
    base_confidence = _clamp_confidence(plan_resp.confidence)
    confidence_value = max(0.0, min(1.0, (base_confidence + component_confidence) / 2.0))
    trade_quality = quality_grade(confidence_value)

    def _extract_float(value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    atr_used_value = _extract_float(plan_resp.calc_notes.get("atr14")) if plan_resp.calc_notes else _extract_float(structured.get("atr_used"))
    probabilities_payload = probabilities or {}
    risk_payload = risk_model_payload(
        entry=float(entry_meta.level),
        stop=stop_value,
        targets=targets_clean,
        probabilities=probabilities_payload,
        direction=direction_token,
        atr_used=atr_used_value,
        style=style_token,
    )
    return SetupMeta(
        plan_id=plan_identifier,
        symbol=plan_resp.symbol,
        style=style_token,  # type: ignore[arg-type]
        direction=direction_token,  # type: ignore[arg-type]
        entry=entry_meta,
        invalid=None,
        stop=stop_value,
        targets=targets_clean,
        probabilities=probabilities,
        probability_components=components,
        trade_quality_score=trade_quality,
        runner=plan_resp.runner,
        confluence=confluence,
        key_levels=key_levels,
        mtf_analysis=mtf_analysis,
        confidence=confidence_value,
        rationale=plan_resp.notes or structured.get("rationale"),
        options=options_block,
        context=context_block,
        em_used=structured.get("em_used") or plan_resp.calc_notes.get("expected_move") if plan_resp.calc_notes else None,
        atr_used=atr_used_value,
        style_horizon_applied=structured.get("style_horizon_applied"),
        risk_model=risk_payload,
        chart_url=chart_url,
        as_of=as_of_value,
        version=int(plan_resp.version) if plan_resp.version is not None else None,
    )


async def _ingest_stream_event(symbol: str, event: Dict[str, Any]) -> None:
    """Push a raw market event through the live engine before fan-out."""

    derived_events = await _LIVE_PLAN_ENGINE.handle_market_event(symbol, event)
    await _publish_stream_event(symbol, event)
    for derived in derived_events:
        await _publish_stream_event(symbol, derived)
    if event.get("t") == "tick":
        symbol_key = symbol.upper()
        async with _ACTIVE_SETUPS_LOCK:
            active = [dict(value) for value in _ACTIVE_SETUPS.values() if value.get("symbol", "").upper() == symbol_key]
        updates: List[Dict[str, Any]] = []
        for setup in active:
            evolved = evolve_plan(setup, event, event.get("ts"))
            if evolved:
                updates.append(evolved)
        if updates:
            async with _ACTIVE_SETUPS_LOCK:
                for payload in updates:
                    _ACTIVE_SETUPS[payload["plan_id"]] = dict(payload)
            for payload in updates:
                ts = event.get("ts") or event.get("timestamp")
                price_event = {
                    "plan_id": payload["plan_id"],
                    "t": "price",
                    "price": event.get("p"),
                    "ts": ts,
                }
                await _publish_plan_event(symbol, price_event)
                update_event = {
                    "plan_id": payload["plan_id"],
                    "t": "plan_update",
                    "setup": payload,
                    "ts": ts,
                }
                await _publish_plan_event(symbol, update_event)
                hit_block = payload.get("event")
                if isinstance(hit_block, dict) and hit_block.get("type") == "hit":
                    for level in hit_block.get("levels", []):
                        hit_event = {
                            "plan_id": payload["plan_id"],
                            "t": "hit",
                            "level": f"tp{level}",
                            "price": event.get("p"),
                            "ts": ts,
                        }
                        await _publish_plan_event(symbol, hit_event)


async def _get_idea_snapshot(plan_id: str, version: Optional[int] = None) -> Dict[str, Any]:
    cached = await _get_cached_snapshot(plan_id, version)
    if cached:
        return cached
    if _IDEA_PERSISTENCE_ENABLED:
        snapshot = await db_fetch_idea_snapshot(plan_id, version=version)
        if snapshot:
            await _cache_snapshot(plan_id, snapshot)
            return snapshot
    if version is None:
        raise HTTPException(status_code=404, detail="Plan not found")
    raise HTTPException(status_code=404, detail="Plan version not found")


async def _regenerate_snapshot_from_slug(plan_id: str, version: Optional[int], request: Request, slug_meta: Dict[str, str]) -> Optional[Dict[str, Any]]:
    """Regenerate a snapshot for slug-style plan IDs when not cached."""

    try:
        plan_request = PlanRequest(symbol=slug_meta['symbol'], style=slug_meta.get('style'))
    except Exception:
        return None

    # Call gpt_plan directly with a synthetic user context
    user = AuthedUser(user_id="slug-regenerator")
    response = await gpt_plan(plan_request, request, user)

    # Fetch the snapshot that gpt_plan just stored
    base_snapshot = None
    try:
        base_snapshot = await _get_idea_snapshot(response.plan_id, version=response.version)
    except HTTPException:
        base_snapshot = None

    if base_snapshot is None:
        return None

    # If the generated plan already matches the slug and version, just return it
    if response.plan_id == plan_id:
        return base_snapshot

    cloned = copy.deepcopy(base_snapshot)
    plan_block = dict(cloned.get("plan") or {})
    plan_block["plan_id"] = plan_id
    plan_block["version"] = response.version
    redirect_url = _build_trade_detail_url(request, plan_id, response.version)
    plan_block["trade_detail"] = redirect_url
    plan_block["idea_url"] = redirect_url
    cloned["plan"] = plan_block
    cloned["trade_detail"] = redirect_url
    cloned["idea_url"] = redirect_url

    await _store_idea_snapshot(plan_id, cloned)
    return cloned


async def _stream_generator(symbol: str) -> Any:
    queue: asyncio.Queue[str] = asyncio.Queue(maxsize=100)
    async with _STREAM_LOCK:
        _STREAM_SUBSCRIBERS.setdefault(symbol, []).append(queue)
    await _ensure_symbol_stream(symbol)
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


async def _build_watch_plan(symbol: str, style: Optional[str], request: Request) -> PlanResponse | None:
    logger.debug("watch plan builder deprecated for %s", symbol)
    return None


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
        timestamp = row.get("time")
        if isinstance(timestamp, pd.Timestamp):
            if timestamp.tzinfo is None:
                timestamp = timestamp.tz_localize("UTC")
            else:
                timestamp = timestamp.tz_convert("UTC")
            timestamp = timestamp.isoformat()
        event = {
            "type": "bar",
            "state": state,
            "price": price,
            "time": timestamp if timestamp is not None else None,
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
    """Fetch recent OHLCV data using Polygon only."""
    candidates = _data_symbol_candidates(symbol)

    # Try Polygon first for each candidate symbol
    fresh_polygon: pd.DataFrame | None = None
    stale_polygon: pd.DataFrame | None = None
    for candidate in candidates:
        poly = await fetch_polygon_ohlcv(candidate, timeframe)
        if poly is None or poly.empty:
            continue
        if not _is_stale_frame(poly, timeframe):
            fresh_polygon = poly
            break
        stale_polygon = poly if stale_polygon is None else stale_polygon

    if fresh_polygon is not None:
        return fresh_polygon

    if stale_polygon is not None:
        logger.warning("Polygon data is stale for %s; returning last known data.", symbol)
        return stale_polygon

    logger.warning("No Polygon data available for %s", symbol)
    return None


async def _collect_market_data(
    tickers: List[str],
    timeframe: str = "5",
    *,
    as_of: datetime | None = None,
) -> Dict[str, pd.DataFrame]:
    """Fetch OHLCV for a list of tickers from Polygon."""
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
        if as_of is not None:
            cutoff = pd.Timestamp(as_of).tz_convert("UTC")
            frame = frame.loc[frame.index <= cutoff]
            if frame.empty:
                logger.warning("No market data available for %s up to %s", ticker, cutoff)
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
        rule = f"{minutes}min"
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


def _fallback_scan_payload(
    tickers: List[str],
    market_data: Dict[str, pd.DataFrame],
    *,
    style_token: str,
    session_info: Dict[str, Any],
    market_meta: Dict[str, Any],
    data_meta: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Construct deterministic fallback setups when scan_market yields nothing."""

    fallback: List[Dict[str, Any]] = []
    primary_style = style_token or "intraday"
    interval_map = {
        "scalp": "5",
        "intraday": "15",
        "swing": "60",
        "leaps": "1D",
    }
    interval_token = interval_map.get(primary_style, "15")

    for symbol in tickers:
        history = market_data.get(symbol)
        if history is None or history.empty:
            continue
        df = history.sort_index()
        last = df.iloc[-1]
        close_price = float(last["close"])
        high = df["high"]
        low = df["low"]
        close = df["close"]
        atr_series = atr(high, low, close, period=14)
        atr_value = float(atr_series.iloc[-1]) if not atr_series.empty else float("nan")
        if not math.isfinite(atr_value) or atr_value <= 0:
            atr_value = max(close_price * 0.01, 0.5)
        ema20_series = ema(close, 20) if len(close) >= 20 else None
        direction = "long"
        if ema20_series is not None and not ema20_series.empty:
            ema20_val = float(ema20_series.iloc[-1])
            if math.isfinite(ema20_val) and close_price < ema20_val:
                direction = "short"
        stop = close_price - atr_value if direction == "long" else close_price + atr_value
        targets: List[float] = []
        for mult in (1.0, 2.0, 3.0):
            level = close_price + atr_value * mult if direction == "long" else close_price - atr_value * mult
            targets.append(round(level, 4))
        key_levels = _extract_key_levels(df)
        snapshot = _build_market_snapshot(df, key_levels)
        plan_id = f"{symbol}-{primary_style}-fallback"
        plan_payload = {
            "plan_id": plan_id,
            "symbol": symbol,
            "style": primary_style,
            "direction": direction,
            "entry": round(close_price, 4),
            "stop": round(stop, 4),
            "targets": targets,
            "confidence": 0.65,
            "risk_reward": 2.0,
            "notes": "Fallback ATR ladder generated automatically.",
            "warnings": ["Fallback plan generated because no live strategy qualified."],
        }
        chart_entry_payload = {"type": "limit", "level": plan_payload["entry"]}
        chart_url = build_chart_url(
            CHART_BASE,
            symbol=symbol,
            plan_id=plan_id,
            as_of=session_info.get("as_of") or "",
            entry=chart_entry_payload,
            stop=plan_payload["stop"],
            targets=targets,
            plan_version="0",
        )
        charts_params = {
            "symbol": symbol,
            "interval": normalize_interval(interval_token),
            "direction": direction,
            "entry": f"{plan_payload['entry']:.2f}",
            "stop": f"{plan_payload['stop']:.2f}",
            "tp": ",".join(f"{t:.2f}" for t in targets),
        }
        charts_payload = {"params": charts_params, "interactive": chart_url}
        fallback.append(
            {
                "symbol": symbol,
                "style": primary_style,
                "strategy_id": "fallback_atr",
                "description": "Fallback ATR ladder (auto-generated)",
                "score": min(0.6, max(0.1, atr_value / max(close_price, 1e-6))),
                "contract_suggestion": None,
                "direction_hint": direction,
                "key_levels": key_levels,
                "market_snapshot": snapshot,
                "charts": charts_payload,
                "features": {"atr14": atr_value},
                "plan": plan_payload,
                "warnings": ["fallback_setup"],
                "data": dict(data_meta),
                "market": dict(market_meta),
                "options": None,
                "session": session_info,
            }
        )

    fallback.sort(key=lambda item: item.get("score", 0.0), reverse=True)
    return fallback[:3]


def _polygon_snapshot_list(kind: str, api_key: str) -> List[Dict[str, Any]]:
    url = f"https://api.polygon.io/v2/snapshot/locale/us/markets/stocks/{kind}"
    params = {"apiKey": api_key}
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            payload = resp.json()
    except httpx.HTTPError:
        return []
    return payload.get("results") or payload.get("tickers") or []


def _autofill_symbols(style: Optional[str]) -> Tuple[List[str], Dict[str, Any]]:
    """Return an auto-generated universe for day-trading scans."""

    metadata: Dict[str, Any] = {"source": "static"}
    settings = get_settings()
    api_key = settings.polygon_api_key
    if not api_key:
        metadata["large_cap"] = _DEFAULT_TOP_SYMBOLS[:100]
        metadata["mid_cap"] = _DEFAULT_TOP_SYMBOLS[100:200]
        return list(_DEFAULT_TOP_SYMBOLS), metadata

    combined: List[Dict[str, Any]] = []
    for endpoint in ("most_actives", "gainers", "losers"):
        combined.extend(_polygon_snapshot_list(endpoint, api_key))

    seen: set[str] = set()
    large: List[str] = []
    mid: List[str] = []
    for item in combined:
        ticker = item.get("ticker")
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        last_trade = item.get("lastTrade") or {}
        price = last_trade.get("p") or last_trade.get("price")
        day = item.get("day") or {}
        volume = day.get("v") or day.get("volume")
        if price is None or volume is None:
            continue
        try:
            price_val = float(price)
            volume_val = float(volume)
        except (TypeError, ValueError):
            continue
        if price_val >= 75 or volume_val >= 15_000_000:
            if len(large) < 150:
                large.append(ticker)
        elif 5 <= price_val <= 75 and volume_val >= 1_500_000:
            if len(mid) < 150:
                mid.append(ticker)

    if not large and not mid:
        metadata["large_cap"] = _DEFAULT_TOP_SYMBOLS[:100]
        metadata["mid_cap"] = _DEFAULT_TOP_SYMBOLS[100:200]
        return list(_DEFAULT_TOP_SYMBOLS), metadata

    style_token = (_normalize_trade_style(style) or "intraday").lower()
    selection: List[str] = []
    large_pool = large[:120]
    mid_pool = [sym for sym in mid if sym not in large_pool][:120]
    if style_token in {"scalp", "intraday"}:
        selection.extend(large_pool[:12])
        selection.extend([symbol for symbol in mid_pool if symbol not in selection][:8])
    elif style_token in {"swing", "leaps"}:
        selection.extend(large_pool[:15])
        selection.extend([symbol for symbol in mid_pool if symbol not in selection][:5])
    else:
        selection.extend(large_pool[:10])
        selection.extend([symbol for symbol in mid_pool if symbol not in selection][:10])

    if not selection:
        selection = (large_pool or mid_pool or list(_DEFAULT_TOP_SYMBOLS))[:20]

    metadata["source"] = "polygon"
    metadata["large_cap"] = large_pool[:200]
    metadata["mid_cap"] = mid_pool[:200]
    return selection, metadata


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
    mapping = {"scalp": "1", "intraday": "5", "swing": "60", "leaps": "1D", "leap": "1D"}
    return mapping.get(normalized, "5")


def _view_for_style(style: str | None) -> str:
    normalized = _normalize_style(style) or ""
    mapping = {"scalp": "1d", "intraday": "5d", "swing": "3M", "leaps": "1Y", "leap": "1Y"}
    return mapping.get(normalized, "fit")


def _range_for_style(style: str | None) -> str:
    normalized = _normalize_style(style) or ""
    mapping = {
        "scalp": "5d",
        "intraday": "15d",
        "swing": "6m",
        "leaps": "1y",
    }
    return mapping.get(normalized, "30d")


def _humanize_strategy(strategy_id: str | None) -> str:
    if not strategy_id:
        return "Setup"
    tokens = re.split(r"[_\s]+", strategy_id.strip()) if isinstance(strategy_id, str) else []
    cleaned = [token.capitalize() for token in tokens if token]
    return " ".join(cleaned) if cleaned else "Setup"


def _format_chart_title(symbol: str, bias: str | None, strategy_id: str | None) -> str:
    symbol_token = symbol.upper() if symbol else "PLAN"
    bias_token = (bias or "").strip().lower()
    if bias_token == "long":
        bias_label = "Long Bias"
    elif bias_token == "short":
        bias_label = "Short Bias"
    else:
        bias_label = None
    strategy_label = _humanize_strategy(strategy_id)
    if bias_label:
        return f"{symbol_token} · {bias_label} ({strategy_label})"
    return f"{symbol_token} · {strategy_label}"


def _format_chart_note(
    symbol: str,
    style: str | None,
    entry: float | None,
    stop: float | None,
    targets: List[float] | None,
) -> str:
    parts: List[str] = []
    if symbol:
        parts.append(symbol.upper())
    if style:
        parts.append(style.title())
    if entry is not None:
        parts.append(f"Entry {entry:.2f}")
    if stop is not None:
        parts.append(f"Stop {stop:.2f}")
    if targets:
        formatted = "/".join(f"{float(tp):.2f}" for tp in targets[:2] if isinstance(tp, (int, float)))
        if formatted:
            parts.append(f"Targets {formatted}")
    summary = " | ".join(parts)
    return summary[:140]


def _build_tv_chart_url(request: Request, params: Dict[str, Any]) -> str:
    # Respect reverse-proxy headers to avoid mixed-content (http iframe on https page)
    headers = request.headers
    scheme = None
    host = None
    xf_proto = headers.get("x-forwarded-proto")
    if xf_proto:
        scheme = xf_proto.split(",")[0].strip()
    xf_host = headers.get("x-forwarded-host")
    if xf_host:
        host = xf_host.split(",")[0].strip()
    fwd = headers.get("forwarded")
    if fwd:
        first = fwd.split(",", 1)[0]
        for part in first.split(";"):
            name, _, value = part.partition("=")
            if not value:
                continue
            name = name.strip().lower()
            value = value.strip().strip('"')
            if name == "proto" and not scheme:
                scheme = value
            elif name == "host" and not host:
                host = value
    if not scheme:
        scheme = request.url.scheme or "https"
    if not host:
        host = headers.get("host") or request.url.netloc
    base = f"{scheme}://{host}/tv"
    query: Dict[str, str] = {}
    for key, value in params.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            value = ",".join(str(item) for item in value if item is not None)
        query[key] = str(value)
    if not query:
        return base
    return f"{base}?{urlencode(query, safe=',|:;+-_() ')}"


TV_SUPPORTED_RESOLUTIONS = ["1", "3", "5", "10", "15", "30", "60", "120", "240", "1D", "1W"]


def _resolution_to_timeframe(resolution: str) -> str | None:
    token = (resolution or "").strip().upper()
    if not token:
        return None
    if token == "10":
        return "5"
    if token.endswith("M") and token[:-1].isdigit():
        return token[:-1]
    if token.endswith("H") and token[:-1].isdigit():
        try:
            return str(int(token[:-1]) * 60)
        except Exception:
            return None
    if token.endswith("D"):
        return "D"
    if token.endswith("W"):
        return "D"
    if token.isdigit():
        return token
    return None


def _resolution_to_minutes(resolution: str) -> int:
    token = (resolution or "").strip().upper()
    if token.endswith("D"):
        days = int("".join(ch for ch in token if ch.isdigit()) or "1")
        return days * 24 * 60
    if token.endswith("W"):
        weeks = int("".join(ch for ch in token if ch.isdigit()) or "1")
        return weeks * 7 * 24 * 60
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
        ("session_high", "Session High"),
        ("session_low", "Session Low"),
        ("opening_range_high", "OR High"),
        ("opening_range_low", "OR Low"),
        ("prev_high", "Prev High"),
        ("prev_low", "Prev Low"),
        ("prev_close", "Prev Close"),
        ("gap_fill", "Gap Fill"),
    ]
    included: set[str] = set()
    levels: List[str] = []

    def _append(value: Any, label: str) -> None:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return
        if not math.isfinite(numeric):
            return
        token = f"{numeric:.2f}|{label}"
        if token not in levels:
            levels.append(token)

    for key, label in order:
        if key in key_levels:
            _append(key_levels.get(key), label)
            included.add(key)

    for key, value in key_levels.items():
        if key in included:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(numeric):
            continue
        label = key.replace("_", " ").title()
        _append(numeric, label)

    return levels


def _plan_meta_payload(
    *,
    symbol: str,
    style: str | None,
    plan: Dict[str, Any],
    runner: Dict[str, Any] | None,
    expected_move: float | None = None,
    horizon_minutes: float | None = None,
    extra: Dict[str, Any] | None = None,
) -> str:
    risk_reward = plan.get("risk_reward")
    if risk_reward is None:
        risk_reward = plan.get("rr_to_t1")

    payload: Dict[str, Any] = {
        "symbol": symbol.upper(),
        "style": style,
        "bias": plan.get("direction"),
        "confidence": plan.get("confidence"),
        "risk_reward": risk_reward,
        "notes": plan.get("notes"),
        "warnings": plan.get("warnings") or [],
        "entry": plan.get("entry"),
        "stop": plan.get("stop"),
        "targets": plan.get("targets") or [],
        "target_meta": plan.get("target_meta") or [],
        "runner": runner,
        "strategy": plan.get("setup") or plan.get("strategy"),
        "atr": plan.get("atr"),
        "expected_move": expected_move,
        "horizon_minutes": horizon_minutes,
    }
    if extra:
        payload.update({k: v for k, v in extra.items() if v is not None})
    return json.dumps(payload, separators=(",", ":"), default=_json_safe_default)


def _json_safe_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return float(value)
    return str(value)


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

# Idea snapshot store (in-memory cache with optional database persistence)
_IDEA_STORE: Dict[str, List[Dict[str, Any]]] = {}
_IDEA_LOCK = asyncio.Lock()
_MAX_IDEA_CACHE_VERSIONS = 20
_IDEA_PERSISTENCE_ENABLED = False
_STREAM_SUBSCRIBERS: Dict[str, List[asyncio.Queue]] = {}
_STREAM_LOCK = asyncio.Lock()
_PLAN_STREAM_SUBSCRIBERS: Dict[str, List[asyncio.Queue]] = {}
_PLAN_STREAM_LOCK = asyncio.Lock()
_IV_METRICS_CACHE_TTL = 120.0
_IV_METRICS_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_LIVE_PLAN_ENGINE = LivePlanEngine()
_SYMBOL_STREAM_COORDINATOR: Optional[SymbolStreamCoordinator] = None
_ACTIVE_SETUPS: Dict[str, Dict[str, Any]] = {}
_ACTIVE_SETUPS_LOCK = asyncio.Lock()
_DEFAULT_TOP_SYMBOLS: List[str] = [
    "SPY",
    "QQQ",
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "META",
    "TSLA",
    "IWM",
    "DIA",
]


async def _symbol_stream_emit(symbol: str, event: Dict[str, Any]) -> None:
    await _ingest_stream_event(symbol, event)


async def _ensure_symbol_stream(symbol: str) -> None:
    coordinator = _SYMBOL_STREAM_COORDINATOR
    symbol_key = (symbol or "").upper()
    if not coordinator or not symbol_key:
        return
    try:
        await coordinator.ensure_symbol(symbol_key)
    except Exception:
        logger.exception("failed to ensure symbol streamer", extra={"symbol": symbol_key})


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

    session_info = _session_block()
    underlying_price_asof = None
    if session_info.get("status") != "open":
        as_of_dt = parse_session_as_of(session_info)
        if as_of_dt is not None:
            underlying_price_asof = await last_price_asof(symbol, as_of_dt)
    as_of_ts = _session_asof_timestamp(session_info)
    if session_info.get("status") != "open" and as_of_ts is not None:
        limited = frame[frame["time"] <= as_of_ts]
        if not limited.empty:
            frame = limited

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
        "session": session_info,
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
    aggregate_resolution = (resolution or "").strip().upper()

    def _resample_frame(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
        try:
            resampled = (
                frame[["open", "high", "low", "close", "volume"]]
                .resample(rule, label="right", closed="right")
                .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
            )
            resampled = resampled.dropna(subset=["open", "high", "low", "close"])
            if resampled.empty:
                return frame
            return resampled
        except Exception:
            return frame

    if aggregate_resolution == "10" and src_tf in {"5", "3", "1"}:
        history = _resample_frame(history, "10min")
    elif aggregate_resolution == "1W":
        history = _resample_frame(history, "W")

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
            sample = await fetch_polygon_ohlcv("SPY", "5")
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

    payload = {
        "status": "ok",
        "services": {
            "polygon": polygon_status,
            "tradier": tradier_status,
        },
    }
    payload["session"] = _session_block()
    return payload


@gpt.get("/futures-snapshot", summary="Overnight/offsessions market tape (ETF proxies via Finnhub)")
async def gpt_futures_snapshot(_: AuthedUser = Depends(require_api_key)) -> Dict[str, Any]:
    now_ts = time.time()
    cached = _FUTURES_CACHE.get("data")
    ts = float(_FUTURES_CACHE.get("ts") or 0)
    if cached and (now_ts - ts < 180):
        payload = {k: (dict(v) if isinstance(v, dict) else v) for k, v in cached.items()}
        payload["stale_seconds"] = int(now_ts - ts)
        payload["session"] = _session_block()
        return payload

    settings = get_settings()
    api_key = (settings.finnhub_api_key or "").strip() if hasattr(settings, "finnhub_api_key") else ""
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail={"code": "UNAVAILABLE", "message": "FINNHUB_API_KEY missing"},
        )

    quotes: Dict[str, Any] = {}
    async with httpx.AsyncClient(timeout=8.0) as client:
        for key, symbol in _FUTURES_PROXY_MAP.items():
            quotes[key] = await _fetch_futures_quote(client, symbol, api_key)

    payload: Dict[str, Any] = dict(quotes)
    payload["market_phase"] = _market_phase_chicago()
    payload["stale_seconds"] = 0
    payload["session"] = _session_block()
    _FUTURES_CACHE["data"] = copy.deepcopy(payload)
    _FUTURES_CACHE["ts"] = now_ts
    return payload


@gpt.post("/scan", summary="Rank trade setups across a list of tickers")
async def gpt_scan(
    universe: ScanUniverse,
    request: Request,
    user: AuthedUser = Depends(require_api_key),
    auto_universe: bool = False,
    auto_meta: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    if not universe.tickers:
        raise HTTPException(status_code=400, detail="No tickers provided")

    market_meta, data_meta, as_of_dt, is_open = _market_snapshot_payload()
    session_info = _session_block()
    session_as_of = parse_session_as_of(session_info)
    if session_as_of is not None:
        as_of_dt = session_as_of
    as_of_ts = None
    if as_of_dt is not None:
        as_of_ts = pd.Timestamp(as_of_dt)
        if as_of_ts.tzinfo is None:
            as_of_ts = as_of_ts.tz_localize("UTC")
        else:
            as_of_ts = as_of_ts.tz_convert("UTC")

    style_filter = _normalize_style(universe.style)
    data_timeframe = {"scalp": "1", "intraday": "5", "swing": "60", "leaps": "D", "leap": "D"}.get(style_filter, "5")

    settings = get_settings()
    market_data = await _collect_market_data(
        universe.tickers,
        timeframe=data_timeframe,
        as_of=None if is_open else (as_of_ts.to_pydatetime() if as_of_ts is not None else None),
    )
    if not market_data:
        raise HTTPException(status_code=502, detail="No market data available for the requested tickers.")
    signals = await scan_market(universe.tickers, market_data)

    if not signals:
        logger.info(
            "No qualifying signals detected; generating fallback setups",
            extra={"tickers": universe.tickers, "style": style_filter},
        )
        fallback_payload = _fallback_scan_payload(
            universe.tickers,
            market_data,
            style_token=style_filter or "intraday",
            session_info=session_info,
            market_meta=market_meta,
            data_meta=data_meta,
        )
        if fallback_payload:
            return fallback_payload

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
    if benchmark_history is not None and as_of_ts is not None and not is_open:
        cutoff = as_of_ts
        benchmark_history = benchmark_history.loc[benchmark_history.index <= cutoff]
        if benchmark_history.empty:
            benchmark_history = None

    symbol_freshness: Dict[str, float] = {}
    data_meta.setdefault("ok", True)
    data_meta["auto_universe"] = auto_universe
    if auto_universe and auto_meta:
        data_meta["auto_universe_source"] = auto_meta.get("source")
        data_meta["auto_universe_large"] = auto_meta.get("large_cap")
        data_meta["auto_universe_mid"] = auto_meta.get("mid_cap")
    if is_open:
        now_utc = pd.Timestamp.utcnow()
        for symbol_key, frame in market_data.items():
            last_ts = frame.index[-1]
            age_ms = max((now_utc - last_ts).total_seconds() * 1000.0, 0.0)
            symbol_freshness[symbol_key] = age_ms
        if symbol_freshness:
            stale_symbols = [sym for sym, age in symbol_freshness.items() if age > 2000]
            if stale_symbols:
                refreshed = False
                for sym in stale_symbols:
                    try:
                        refreshed_frame = await _load_remote_ohlcv(sym, data_timeframe)
                    except Exception as exc:
                        logger.warning("Refresh fetch failed for %s: %s", sym, exc)
                        continue
                    if refreshed_frame is None or refreshed_frame.empty:
                        continue
                    market_data[sym] = refreshed_frame
                    last_ts = refreshed_frame.index[-1]
                    symbol_freshness[sym] = max((now_utc - last_ts).total_seconds() * 1000.0, 0.0)
                    refreshed = True
                if refreshed:
                    logger.info("Refreshed %d symbols due to stale feed", len(stale_symbols))
            if symbol_freshness:
                max_age = max(symbol_freshness.values())
                data_meta["data_freshness_ms"] = int(max_age)
                if max_age > 2000:
                    logger.warning("Detected stale market data during RTH (max age %.0f ms)", max_age)
                    data_meta["ok"] = False
                    data_meta["error"] = "stale_feed"
                else:
                    data_meta.pop("error", None)
            else:
                data_meta["data_freshness_ms"] = None
        else:
            data_meta["data_freshness_ms"] = None
            data_meta.pop("error", None)
    else:
        data_meta.pop("data_freshness_ms", None)
        data_meta["ok"] = True
        data_meta.pop("error", None)

    polygon_chains: Dict[str, pd.DataFrame] = {}
    if unique_symbols and polygon_enabled:
        try:
            if not is_open and as_of_ts is not None:
                tasks = [fetch_polygon_option_chain_asof(symbol, as_of_ts.to_pydatetime()) for symbol in unique_symbols]
            else:
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
        if not is_open and as_of_dt is not None:
            cutoff = pd.Timestamp(as_of_dt).tz_convert("UTC")
            history = history.loc[history.index <= cutoff]
            if history.empty:
                logger.warning("No market data available for %s at %s", signal.symbol, cutoff)
                continue
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
        chart_query: Dict[str, Any] = {
            "symbol": signal.symbol.upper(),
            "interval": interval,
            "ema": ",".join(str(span) for span in ema_spans),
            "view": _view_for_style(style),
            "vwap": "1",
            "theme": "dark",
        }
        if session_info.get("as_of"):
            chart_query["as_of"] = session_info["as_of"]
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

        plan_entry = None
        plan_stop = None
        plan_targets: List[float] = []
        plan_direction = direction_hint
        version = 1
        plan_id = _generate_plan_slug(signal.symbol, style, plan_direction, snapshot)
        if signal.plan is not None:
            plan_payload = signal.plan.as_dict()
            plan_entry = float(signal.plan.entry)
            plan_stop = float(signal.plan.stop)
            plan_targets = [float(target) for target in signal.plan.targets]
            plan_direction = signal.plan.direction or plan_direction
            version = int(plan_payload.get("version") or version)
            slug_hint = plan_payload.get("plan_id") if isinstance(plan_payload, dict) else None
            if slug_hint:
                plan_id = str(slug_hint)
            chart_query["entry"] = f"{signal.plan.entry:.2f}"
            chart_query["stop"] = f"{signal.plan.stop:.2f}"
            chart_query["tp"] = ",".join(f"{target:.2f}" for target in signal.plan.targets)
            chart_query.setdefault("direction", signal.plan.direction)
            if signal.plan.atr and "atr" not in chart_query:
                chart_query["atr"] = f"{float(signal.plan.atr):.4f}"
        chart_query["range"] = _range_for_style(style)
        bias_for_chart = plan_direction or direction_hint
        chart_query["title"] = _format_chart_title(signal.symbol, bias_for_chart, signal.strategy_id)
        chart_note = None
        if signal.plan is not None and signal.plan.notes:
            chart_note = str(signal.plan.notes).strip()
        if not chart_note:
            chart_note = _format_chart_note(signal.symbol, style, plan_entry, plan_stop, plan_targets)
        if chart_note:
            chart_query["notes"] = chart_note[:140]
        overlay_params = _encode_overlay_params(enhancements or {})
        for key, value in overlay_params.items():
            chart_query[key] = value
        level_tokens = _extract_levels_for_chart(key_levels)
        if level_tokens:
            chart_query["levels"] = ",".join(level_tokens)
        chart_query["strategy"] = signal.strategy_id
        chart_query["plan_id"] = plan_id
        chart_query["plan_version"] = version
        if bias_for_chart:
            chart_query["direction"] = bias_for_chart
        if signal.plan is not None:
            plan_meta_plan = {
                "direction": plan_direction,
                "entry": plan_entry,
                "stop": plan_stop,
                "targets": plan_targets,
                "target_meta": plan_payload.get("target_meta") if isinstance(plan_payload, dict) else [],
                "confidence": float(signal.plan.confidence) if signal.plan.confidence is not None else None,
                "risk_reward": (
                    float(signal.plan.risk_reward)
                    if signal.plan.risk_reward is not None
                    else (
                        float(plan_payload.get("rr_to_t1"))
                        if isinstance(plan_payload, dict) and plan_payload.get("rr_to_t1") is not None
                        else None
                    )
                ),
                "notes": plan_payload.get("notes") if isinstance(plan_payload, dict) else None,
                "warnings": plan_payload.get("warnings") if isinstance(plan_payload, dict) else [],
                "setup": signal.strategy_id,
                "atr": float(signal.plan.atr) if signal.plan.atr is not None else None,
            }
            runner_meta = plan_payload.get("runner") if isinstance(plan_payload, dict) else None
            expected_move_meta = plan_payload.get("expected_move") if isinstance(plan_payload, dict) else None
            chart_query["plan_meta"] = _plan_meta_payload(
                symbol=signal.symbol,
                style=style,
                plan=plan_meta_plan,
                runner=runner_meta,
                expected_move=expected_move_meta,
                horizon_minutes=None,
                extra={
                    "style_display": public_style(style),
                    "strategy_label": signal.strategy_id,
                    "key_levels": key_levels,
                },
            )
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
            if plan_dict.get("target_meta"):
                try:
                    chart_query["tp_meta"] = json.dumps(plan_dict["target_meta"])
                except Exception:
                    chart_query["tp_meta"] = json.dumps([])
            if plan_dict.get("runner"):
                chart_query["runner"] = json.dumps(plan_dict["runner"])

        chart_links = None
        required_chart_keys = {"direction", "entry", "stop", "tp"}
        if required_chart_keys.issubset(chart_query.keys()):
            try:
                chart_links = await gpt_chart_url(ChartParams(**chart_query), request)
            except HTTPException as exc:
                logger.debug("chart link generation failed for %s: %s", signal.symbol, exc)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("chart link generation error for %s: %s", signal.symbol, exc)

        charts_payload: Dict[str, Any] = {"params": chart_query}
        if chart_links:
            charts_payload["interactive"] = chart_links.interactive
        elif required_chart_keys.issubset(chart_query.keys()):
            fallback_chart_url = _build_tv_chart_url(request, chart_query)
            charts_payload["interactive"] = fallback_chart_url
            logger.debug(
                "chart link fallback used",
                extra={"symbol": signal.symbol, "strategy_id": signal.strategy_id, "url": fallback_chart_url},
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
                "charts": charts_payload,
                "features": feature_payload,
                **({"plan": plan_payload} if plan_payload else {}),
                "warnings": plan_payload.get("warnings") if plan_payload else [],
                "data": {
                    **data_meta,
                    "bars": f"{base_url}/gpt/context/{signal.symbol}?interval={interval}&lookback=300",
                    "symbol_freshness_ms": (
                        int(symbol_freshness.get(signal.symbol, 0.0)) if symbol_freshness else None
                    ),
                },
                "market": dict(market_meta),
                "context_overlays": enhancements,
                **({"options": polygon_bundle} if polygon_bundle else {}),
                "session": session_info,
            }
        )

    logger.info("scan universe=%s user=%s results=%d", universe.tickers, user.user_id, len(payload))
    for item in payload:
        if isinstance(item, dict):
            item.setdefault("session", session_info)
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
    forced_plan_id = (request_payload.plan_id or "").strip()
    logger.info(
        "gpt_plan received",
        extra={
            "symbol": symbol,
            "style": request_payload.style,
            "user_id": getattr(user, "user_id", None),
        },
    )
    universe = ScanUniverse(tickers=[symbol], style=request_payload.style)
    results = await gpt_scan(universe, request, user)
    if not results:
        raise HTTPException(status_code=404, detail=f"No valid plan for {symbol}")
    first = next((item for item in results if (item.get("symbol") or "").upper() == symbol), results[0])
    session_info = first.get("session") if isinstance(first.get("session"), dict) else _session_block()
    logger.info(
        "gpt_plan raw result received",
        extra={
            "symbol": symbol,
            "style": first.get("style"),
            "contains_plan": bool(first.get("plan")),
            "planning_context": first.get("planning_context"),
            "available_keys": sorted(first.keys()),
        },
    )

    snapshot = first.get("market_snapshot") or {}
    indicators = (snapshot.get("indicators") or {})
    volatility = (snapshot.get("volatility") or {})
    # Build calc_notes + htf from available payload
    raw_plan = first.get("plan") or {}
    plan: Dict[str, Any] = dict(raw_plan)
    raw_plan_id = str(plan.get("plan_id") or "").strip()
    raw_version = plan.get("version")
    try:
        version = int(raw_version) if raw_version is not None else 1
    except (TypeError, ValueError):
        version = 1
    # Determine direction for slugging
    direction_hint = plan.get("direction") or (snapshot.get("trend") or {}).get("direction_hint")
    plan_id = raw_plan_id or forced_plan_id or _generate_plan_slug(symbol, first.get("style"), direction_hint, snapshot)
    # If version not provided, bump based on snapshot store to ensure unique URLs
    if forced_plan_id:
        plan_id = forced_plan_id
    plan["plan_id"] = plan_id
    plan["version"] = version
    updated_from_version = version - 1 if version > 1 else None
    update_reason = "manual_refresh" if forced_plan_id else None
    plan.setdefault("symbol", symbol)
    plan.setdefault("style", first.get("style"))
    plan.setdefault("direction", plan.get("direction") or (snapshot.get("trend") or {}).get("direction_hint"))
    first["plan"] = plan
    logger.info(
        "plan identity normalized",
        extra={
            "symbol": symbol,
            "requested_style": request_payload.style,
            "plan_id": plan_id,
            "version": version,
            "source_plan_keys": list(raw_plan.keys()),
        },
    )
    charts_container = first.get("charts") or {}
    charts = charts_container.get("params") if isinstance(charts_container, dict) else None
    chart_params_payload: Dict[str, Any] = charts if isinstance(charts, dict) else {}
    chart_url_value: Optional[str] = charts_container.get("interactive") if isinstance(charts_container, dict) else None

    def _coerce_float(value: Any) -> Optional[float]:
        try:
            if value is None or (isinstance(value, str) and not value.strip()):
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    entry_val = _coerce_float(plan.get("entry")) or _coerce_float(chart_params_payload.get("entry"))
    stop_val = _coerce_float(plan.get("stop")) or _coerce_float(chart_params_payload.get("stop"))
    target_tokens: List[Any] = list(plan.get("targets") or [])
    if not target_tokens and chart_params_payload.get("tp"):
        target_tokens = [
            _coerce_float(token.strip())
            for token in str(chart_params_payload.get("tp")).split(",")
            if token and str(token).strip()
        ]
    targets_list = [float(tp) for tp in target_tokens if tp is not None]
    tp1_value = targets_list[0] if targets_list else None

    rr_inputs = None
    if entry_val is not None and stop_val is not None and tp1_value is not None:
        rr_inputs = {"entry": entry_val, "stop": stop_val, "tp1": tp1_value}

    if chart_params_payload:
        bias_token = plan.get("direction") or chart_params_payload.get("direction") or (snapshot.get("trend") or {}).get("direction_hint")
        chart_params_payload.setdefault(
            "title",
            _format_chart_title(symbol, bias_token, first.get("strategy_id")),
        )
        chart_params_payload.setdefault("plan_id", plan_id)
        chart_params_payload.setdefault("plan_version", version)
        chart_params_payload.setdefault("strategy", first.get("strategy_id") or plan.get("setup"))
        chart_params_payload.setdefault("symbol", symbol)
        chart_params_payload.setdefault("range", _range_for_style(first.get("style")))
        if session_info.get("as_of"):
            chart_params_payload.setdefault("as_of", session_info["as_of"])
        if entry_val is not None or stop_val is not None or targets_list:
            default_note = _format_chart_note(symbol, first.get("style"), entry_val, stop_val, targets_list)
            if default_note and not chart_params_payload.get("notes"):
                chart_params_payload["notes"] = default_note
        try:
            chart_model = ChartParams(**chart_params_payload)
            if not chart_url_value or not isinstance(chart_url_value, str):
                chart_links = await gpt_chart_url(chart_model, request)
                chart_url_value = chart_links.interactive
        except HTTPException as exc:
            logger.debug("plan chart link validation failed for %s: %s", symbol, exc)
            chart_url_value = None
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("plan chart link error for %s: %s", symbol, exc)
            chart_url_value = None

    charts_payload: Dict[str, Any] = {}
    minimal_params: Dict[str, Any] | None = None
    if chart_params_payload:
        charts_payload["params"] = chart_params_payload
    if chart_url_value:
        charts_payload["interactive"] = chart_url_value
    elif chart_params_payload and {"direction", "entry", "stop", "tp"}.issubset(chart_params_payload.keys()):
        fallback_chart_url = _build_tv_chart_url(request, chart_params_payload)
        charts_payload["interactive"] = fallback_chart_url
        chart_url_value = fallback_chart_url
        logger.debug(
            "plan chart fallback used",
            extra={"symbol": symbol, "plan_id": plan_id, "url": fallback_chart_url},
        )

    if not chart_url_value:
        minimal_params = {
            "symbol": symbol,
            "interval": normalize_interval(plan.get("interval") or chart_params_payload.get("interval") or "15"),
            "plan_id": plan_id,
            "plan_version": str(version),
        }
        if entry_val is not None:
            minimal_params["entry"] = f"{entry_val:.2f}"
        if stop_val is not None:
            minimal_params["stop"] = f"{stop_val:.2f}"
        if targets_list:
            minimal_params["tp"] = ",".join(f"{target:.2f}" for target in targets_list)
        if session_info.get("as_of"):
            minimal_params["as_of"] = session_info["as_of"]
        chart_url_value = _build_tv_chart_url(request, minimal_params)
        charts_payload.setdefault("params", minimal_params)
        charts_payload["interactive"] = chart_url_value

    atr_val = _safe_number(indicators.get("atr14"))

    profile_dict = plan.get("target_profile") or {}
    try:
        target_profile_obj = TargetEngineResult(
            entry=float(profile_dict.get("entry", entry_output or plan.get("entry") or 0.0)),
            stop=float(profile_dict.get("stop", stop_output or plan.get("stop") or 0.0)),
            targets=[float(t) for t in profile_dict.get("targets", plan.get("targets") or targets_list or [])],
            probabilities={k: float(v) for k, v in (profile_dict.get("probabilities") or {}).items()},
            em_used=profile_dict.get("em_used"),
            atr_used=profile_dict.get("atr_used", indicators.get("atr14")),
            snap_trace=[dict(item) for item in profile_dict.get("snap_trace", []) if isinstance(item, dict)],
            meta=[dict(item) for item in profile_dict.get("meta", []) if isinstance(item, dict)],
            warnings=[str(w) for w in profile_dict.get("warnings", plan_warnings or [])],
            runner=plan.get("runner"),
            bias=plan.get("direction"),
            style=plan.get("style"),
            expected_move=profile_dict.get("expected_move"),
        )
    except Exception:
        target_profile_obj = TargetEngineResult(
            entry=float(plan.get("entry") or entry_val or 0.0),
            stop=float(plan.get("stop") or stop_val or 0.0),
            targets=[float(t) for t in (plan.get("targets") or targets_list or [])],
            probabilities={},
            em_used=None,
            atr_used=atr_val,
            snap_trace=[],
            meta=[],
            warnings=[str(w) for w in (plan.get("warnings") or [])],
            runner=plan.get("runner"),
            bias=plan.get("direction"),
            style=plan.get("style"),
            expected_move=None,
        )
    plan["target_profile"] = target_profile_obj.to_dict()

    entry_type_token = None
    if isinstance(chart_params_payload, dict):
        entry_type_token = chart_params_payload.get("entry_type")
    entry_type_token = plan.get("entry_type") or entry_type_token or "limit"
    if isinstance(entry_type_token, str):
        entry_type_token = entry_type_token.lower()
    chart_entry_payload = {
        "type": entry_type_token,
        "level": target_profile_obj.entry,
    }
    chart_targets_payload = target_profile_obj.targets or targets_list
    chart_stop_value = target_profile_obj.stop
    chart_url_value = build_chart_url(
        CHART_BASE,
        symbol=symbol,
        plan_id=plan_id,
        as_of=session_info.get("as_of") or "",
        entry=chart_entry_payload,
        stop=chart_stop_value,
        targets=chart_targets_payload,
        plan_version=str(version),
    )
    charts_payload["interactive"] = chart_url_value

    trade_detail_url = chart_url_value
    plan["trade_detail"] = trade_detail_url
    plan["idea_url"] = trade_detail_url

    confluence_tags = []
    for meta_entry in target_profile_obj.meta:
        tag = meta_entry.get("snap_tag") or meta_entry.get("tag")
        if tag:
            token = str(tag)
            if token not in confluence_tags:
                confluence_tags.append(token)

    structured_plan = build_structured_plan(
        plan_id=plan_id,
        symbol=symbol,
        style=plan.get("style"),
        direction=plan.get("direction"),
        profile=target_profile_obj,
        confidence=plan.get("confidence"),
        rationale=plan.get("notes"),
        options_block=first.get("options"),
        chart_url=chart_url_value,
        session=session_info,
        confluence=confluence_tags,
    )
    _validate_chart_url(structured_plan.get("chart_url"))
    events_block = first.get("events")
    gating_decision = apply_event_gating(_coalesce_events(events_block))
    if gating_decision.action == "suppress":
        structured_plan["status"] = "suppressed"
        plan_warnings.append(
            f"Plan suppressed due to upcoming event ({gating_decision.reason})"
        )
        logger.info(
            "plan gating suppress",
            extra={
                "symbol": symbol,
                "plan_id": plan_id,
                "events": gating_decision.triggered,
                "reason": gating_decision.reason,
            },
        )
    elif gating_decision.action == "defined_risk":
        structured_plan["defined_risk_only"] = True
        plan_warnings.append(
            f"Defined-risk only: upcoming event ({gating_decision.reason})"
        )
        logger.info(
            "plan gating defined-risk",
            extra={
                "symbol": symbol,
                "plan_id": plan_id,
                "events": gating_decision.triggered,
                "reason": gating_decision.reason,
            },
        )

    plan["structured_plan"] = structured_plan
    first["structured_plan"] = structured_plan
    first["target_profile"] = plan["target_profile"]
    if isinstance(first.get("features"), dict):
        first["features"]["target_profile"] = plan["target_profile"]
        first["features"]["structured_plan"] = structured_plan
        first["features"]["event_gating"] = gating_decision.to_dict()

    calc_notes: Dict[str, Any] = {}
    if atr_val is not None:
        calc_notes["atr14"] = atr_val
    stop_multiple = None
    if atr_val and atr_val > 0 and entry_val is not None and stop_val is not None:
        try:
            stop_multiple = abs(entry_val - stop_val) / atr_val
        except ZeroDivisionError:
            stop_multiple = None
    if stop_multiple is not None:
        calc_notes["stop_multiple"] = round(float(stop_multiple), 3)
    if rr_inputs:
        calc_notes["rr_inputs"] = rr_inputs
    # Infer snapped_targets by comparing target prices to named levels (key_levels + overlays)
    snapped_names: List[str] = []
    try:
        targets_for_snap = list(targets_list)
        atr_for_window = float(indicators.get("atr14") or 0.0)
        window = max(atr_for_window * 0.30, 0.0)
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
        for tp in targets_for_snap[:2]:
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
    raw_warnings = first.get("warnings") or plan.get("warnings") or []
    if isinstance(raw_warnings, list):
        plan_warnings: List[str] = list(raw_warnings)
    elif raw_warnings:
        plan_warnings = [str(raw_warnings)]
    else:
        plan_warnings = []
    plan_warnings = [w for w in plan_warnings if "watch plan" not in str(w).lower()]
    planning_context_value: str | None = None
    market_meta_context = first.get("market") or first.get("meta")
    data_meta_context = first.get("data") or first.get("meta")
    if isinstance(market_meta_context, dict) and market_meta_context.get("status") != "open":
        planning_context_value = "frozen"

    style_token = plan.get("style") or first.get("style") or request_payload.style
    style_public = public_style(style_token) or "intraday"
    side_hint = _infer_contract_side(plan.get("side"), plan.get("direction") or direction_hint)
    options_payload = first.get("options")
    if side_hint:
        plan_anchor: Dict[str, Any] = {}
        if entry_val is not None:
            plan_anchor["underlying_entry"] = entry_val
        if stop_val is not None:
            plan_anchor["stop"] = stop_val
        if targets_list:
            plan_anchor["targets"] = targets_list[:2]
        plan_anchor["horizon_minutes"] = 60 if style_public in {"swing", "leaps"} else 30
        contract_request = ContractsRequest(
            symbol=symbol,
            side=side_hint,
            style=style_public,
            selection_mode="analyze",
            plan_anchor=plan_anchor or None,
        )
        try:
            options_payload = await gpt_contracts(contract_request, request, user)
            first["options"] = options_payload
        except HTTPException as exc:
            logger.info("contract lookup skipped for %s: %s", symbol, exc)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("contract lookup error for %s: %s", symbol, exc)

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
    if plan_core.get("setup") in {"watch_plan", "offline"}:
        inferred_setup = first.get("strategy_id") or plan.get("setup")
        plan_core["setup"] = inferred_setup
        plan["setup"] = inferred_setup
    else:
        plan_core.setdefault("setup", first.get("strategy_id"))
        plan.setdefault("setup", plan_core.get("setup"))
    plan_core["trade_detail"] = trade_detail_url
    plan_core["idea_url"] = trade_detail_url
    if updated_from_version:
        plan_core["updated_from_version"] = updated_from_version
    if update_reason:
        plan_core["update_reason"] = update_reason
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
    logger.info(
        "plan response built",
        extra={
            "symbol": symbol,
            "style": first.get("style"),
            "plan_id": plan_id,
            "trade_detail": trade_detail_url,
            "version": version,
            "chart_url_present": bool(chart_url_value),
            "targets": targets_list[:2],
        },
    )

    # Debug info: include any structural TP1 notes from features
    debug_payload = {}
    try:
        feats = first.get("features") or {}
        tp1_dbg = feats.get("tp1_struct_debug")
        if tp1_dbg:
            debug_payload["tp1"] = tp1_dbg
    except Exception:
        debug_payload = {}

    entry_output = plan.get("entry") if plan.get("entry") is not None else entry_val
    stop_output = plan.get("stop") if plan.get("stop") is not None else stop_val
    targets_output = plan.get("targets") or (targets_list if targets_list else None)
    rr_output = plan.get("risk_reward")
    confidence_output = plan.get("confidence")
    notes_output = (plan.get("notes") or "").strip() if plan else ""
    if plan and "watch plan" in notes_output.lower():
        plan["notes"] = None
        notes_output = ""
    elif "watch plan" in notes_output.lower():
        notes_output = ""
    bias_output = plan.get("direction") or ((snapshot.get("trend") or {}).get("direction_hint"))
    relevant_levels = first.get("key_levels") or {}
    expected_move_basis = None
    if isinstance(volatility.get("expected_move_horizon"), (int, float)):
        expected_move_basis = "expected_move_horizon"
    sentiment_block = first.get("sentiment")
    events_block = first.get("events")
    earnings_block = first.get("earnings")
    confidence_factors = None
    feature_block = first.get("features") or {}
    for key in ("plan_confidence_factors", "plan_confidence_reasons", "confidence_reasons"):
        raw = feature_block.get(key)
        if isinstance(raw, (list, tuple)):
            confidence_factors = [str(item) for item in raw if item]
            break
    calc_notes_output = calc_notes or None
    if calc_notes_output is not None and not calc_notes_output:
        calc_notes_output = None
    charts_field = charts_payload or None
    charts_params_output = chart_params_payload or None
    chart_url_output = chart_url_value or None
    market_meta = market_meta_context if isinstance(market_meta_context, dict) else None
    data_meta = data_meta_context if isinstance(data_meta_context, dict) else None
    if market_meta is None or data_meta is None:
        fallback_market, fallback_data, _, _ = _market_snapshot_payload()
        if market_meta is None:
            market_meta = fallback_market
        if data_meta is None:
            data_meta = fallback_data
        else:
            data_meta.setdefault("as_of_ts", fallback_data["as_of_ts"])
            data_meta.setdefault("frozen", fallback_data["frozen"])
            data_meta.setdefault("ok", fallback_data.get("ok", True))
    if planning_context_value is None:
        planning_context_value = "live"
    if planning_context_value is None:
        planning_context_value = "live"

    logger.info(
        "plan response ready",
        extra={
            "symbol": symbol,
            "plan_id": plan_id,
            "version": version,
            "planning_context": planning_context_value,
            "trade_detail": trade_detail_url,
            "idea_url": trade_detail_url,
        },
    )

    return PlanResponse(
        plan_id=plan_id,
        version=version,
        trade_detail=trade_detail_url,
        idea_url=trade_detail_url,
        warnings=plan_warnings or None,
        planning_context="frozen" if market_meta.get("status") != "open" else "live",
        symbol=first.get("symbol"),
        style=first.get("style"),
        bias=bias_output,
        setup=first.get("strategy_id"),
        entry=entry_output,
        stop=stop_output,
        targets=targets_output,
        target_meta=plan.get("target_meta") if plan else None,
        rr_to_t1=rr_output,
        confidence=confidence_output,
        confidence_factors=confidence_factors,
        notes=notes_output,
        relevant_levels=relevant_levels or None,
        expected_move_basis=expected_move_basis,
        sentiment=sentiment_block,
        events=events_block,
        earnings=earnings_block,
        charts_params=charts_params_output,
        chart_url=chart_url_output,
        strategy_id=first.get("strategy_id"),
        description=first.get("description"),
        score=first.get("score"),
        plan=first.get("plan"),
        charts=charts_field,
        key_levels=first.get("key_levels"),
        market_snapshot=first.get("market_snapshot"),
        features=first.get("features"),
        options=first.get("options"),
        calc_notes=calc_notes_output,
        htf=htf,
        decimals=decimals_value,
        data_quality=data_quality,
        debug=debug_payload or None,
        runner=plan.get("runner") if plan else None,
        updated_from_version=updated_from_version,
        update_reason=update_reason,
        market=market_meta,
        data=data_meta,
        session=session_info,
        target_profile=plan.get("target_profile") if plan else None,
        structured_plan=plan.get("structured_plan") if plan else None,
    )

@app.post("/internal/idea/store", include_in_schema=False, tags=["internal"])
async def internal_idea_store(payload: IdeaStoreRequest, request: Request) -> IdeaStoreResponse:
    plan_block = dict(payload.plan or {})
    plan_id = plan_block.get("plan_id")
    version = plan_block.get("version")
    if not plan_id or version is None:
        raise HTTPException(status_code=400, detail="plan.plan_id and plan.version are required")
    trade_detail_url = _build_trade_detail_url(request, plan_id, int(version))
    snapshot = payload.model_dump()
    plan_payload = snapshot.get("plan") or {}
    legacy_detail = plan_payload.pop("idea_url", None)
    if "trade_detail" not in plan_payload:
        plan_payload["trade_detail"] = legacy_detail or trade_detail_url
    snapshot["plan"] = plan_payload
    snapshot.setdefault("chart_url", None)
    await _store_idea_snapshot(plan_id, snapshot)
    return IdeaStoreResponse(plan_id=plan_id, trade_detail=trade_detail_url, idea_url=trade_detail_url)


async def _ensure_snapshot(plan_id: str, version: Optional[int], request: Request) -> Dict[str, Any]:
    try:
        snapshot = await _get_idea_snapshot(plan_id, version=version)
    except HTTPException as exc:
        if exc.status_code != 404:
            raise
        slug_meta = _parse_plan_slug(plan_id)
        if not slug_meta:
            raise
        snapshot = await _regenerate_snapshot_from_slug(plan_id, version, request, slug_meta)
        if snapshot is None:
            raise HTTPException(status_code=404, detail="Plan not found")
    plan_block = snapshot.get("plan") or {}
    await _ensure_symbol_stream(plan_block.get("symbol"))
    try:
        await _LIVE_PLAN_ENGINE.register_snapshot(snapshot)
    except Exception:
        logger.exception("live plan engine registration failed during ensure", extra={"plan_id": plan_id})
    return snapshot


@app.get("/idea/{plan_id}")
async def get_latest_idea(plan_id: str, request: Request) -> Any:
    snapshot = await _ensure_snapshot(plan_id, None, request)
    return snapshot


@app.get("/idea/{plan_id}/{version}")
async def get_idea_version(plan_id: str, version: int, request: Request) -> Any:
    snapshot = await _ensure_snapshot(plan_id, int(version), request)
    return snapshot


@app.get("/plan/{plan_id}")
async def get_plan_latest(plan_id: str, request: Request) -> Any:
    """Alias for /idea/{plan_id} to support new frontend permalinks."""
    return await _ensure_snapshot(plan_id, None, request)


@app.get("/plan/{plan_id}/{version}")
async def get_plan_version(plan_id: str, version: int, request: Request) -> Any:
    """Alias for /idea/{plan_id}/{version} to support new frontend permalinks."""
    return await _ensure_snapshot(plan_id, int(version), request)


@app.post("/idea/{plan_id}/refresh")
async def refresh_plan_snapshot(
    plan_id: str,
    request: Request,
    user: AuthedUser = Depends(require_api_key),
) -> PlanResponse:
    snapshot = await _ensure_snapshot(plan_id, None, request)
    plan_block = snapshot.get("plan") or {}
    symbol = (plan_block.get("symbol") or "").strip()
    if not symbol:
        raise HTTPException(status_code=400, detail="Plan snapshot missing symbol")
    style = plan_block.get("style")
    plan_request = PlanRequest(symbol=symbol, style=style, plan_id=plan_id)
    response = await gpt_plan(plan_request, request, user)
    return response


@app.get("/stream/market")
async def stream_market(symbol: str = Query(..., min_length=1)) -> StreamingResponse:
    async def event_generator():
        uppercase = symbol.upper()
        await _ensure_symbol_stream(uppercase)
        initial_states = await _LIVE_PLAN_ENGINE.active_plan_states(uppercase)
        if initial_states:
            payload = json.dumps({"symbol": uppercase, "event": {"t": "plan_state", "plans": initial_states}})
            yield f"data: {payload}\n\n"
        async for chunk in _stream_generator(uppercase):
            yield chunk

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/stream/{symbol}")
async def stream_symbol_sse(symbol: str) -> StreamingResponse:
    async def event_generator():
        uppercase = symbol.upper()
        await _ensure_symbol_stream(uppercase)
        initial_states = await _LIVE_PLAN_ENGINE.active_plan_states(uppercase)
        if initial_states:
            payload = json.dumps({"symbol": uppercase, "event": {"t": "plan_state", "plans": initial_states}})
            yield f"data: {payload}\n\n"
        async for chunk in _stream_generator(uppercase):
            yield chunk

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.websocket("/stream/{symbol}")
async def stream_symbol_ws(websocket: WebSocket, symbol: str) -> None:
    uppercase = symbol.upper()
    await websocket.accept()
    await _ensure_symbol_stream(uppercase)
    queue: asyncio.Queue[str] = asyncio.Queue(maxsize=100)
    async with _STREAM_LOCK:
        _STREAM_SUBSCRIBERS.setdefault(uppercase, []).append(queue)
    try:
        initial_states = await _LIVE_PLAN_ENGINE.active_plan_states(uppercase)
        if initial_states:
            payload = json.dumps({"symbol": uppercase, "event": {"t": "plan_state", "plans": initial_states}})
            await websocket.send_text(payload)
        while True:
            data = await queue.get()
            await websocket.send_text(data)
    except WebSocketDisconnect:
        return
    finally:
        async with _STREAM_LOCK:
            subscribers = _STREAM_SUBSCRIBERS.get(uppercase, [])
            if queue in subscribers:
                subscribers.remove(queue)
            if not subscribers:
                _STREAM_SUBSCRIBERS.pop(uppercase, None)
        try:
            await websocket.close()
        except RuntimeError:
            pass


@app.websocket("/ws/plans")
async def stream_plan_ws(
    websocket: WebSocket,
    symbol: str = Query(...),
    plan_id: str = Query(...),
) -> None:
    uppercase = (symbol or "").upper()
    target_plan = (plan_id or "").strip()
    if not uppercase or not target_plan:
        await websocket.close(code=4000)
        return
    await websocket.accept()
    logger.info("plan websocket connected", extra={"symbol": uppercase, "plan_id": target_plan})
    await _ensure_symbol_stream(uppercase)
    key = _plan_stream_key(uppercase, target_plan)
    queue: asyncio.Queue[str] = asyncio.Queue(maxsize=100)
    async with _PLAN_STREAM_LOCK:
        _PLAN_STREAM_SUBSCRIBERS.setdefault(key, []).append(queue)
    try:
        initial_states = await _LIVE_PLAN_ENGINE.active_plan_states(uppercase)
        snapshot = next((state for state in initial_states if state.get("plan_id") == target_plan), None)
        if snapshot:
            await websocket.send_text(
                json.dumps(
                    {
                        "symbol": uppercase,
                        "plan_id": target_plan,
                        "event": {"type": "state", "state": snapshot},
                    }
                )
            )
        while True:
            data = await queue.get()
            await websocket.send_text(data)
    except WebSocketDisconnect:
        return
    finally:
        async with _PLAN_STREAM_LOCK:
            subscribers = _PLAN_STREAM_SUBSCRIBERS.get(key, [])
            if queue in subscribers:
                subscribers.remove(queue)
            if not subscribers:
                _PLAN_STREAM_SUBSCRIBERS.pop(key, None)
        try:
            await websocket.close()
        except RuntimeError:
            pass
        logger.info("plan websocket disconnected", extra={"symbol": uppercase, "plan_id": target_plan})


@app.post("/internal/stream/push", include_in_schema=False, tags=["internal"])
async def internal_stream_push(payload: StreamPushRequest) -> Dict[str, str]:
    symbol = (payload.symbol or "").upper()
    if not symbol:
        raise HTTPException(status_code=400, detail="Symbol is required")
    await _ingest_stream_event(symbol, payload.event or {})
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

    session_info = _session_block()
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
        snapshots=contexts,
        volatility_regime=volatility_regime,
        sentiment=sentiment,
        events=events,
        earnings=earnings,
        summary=summary,
        decimals=decimals,
        data_quality=data_quality,
        contexts=contexts,
        session=session_info,
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
            "option_type": row_type.lower() if isinstance(row_type, str) else None,
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

    session_info = _session_block()

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

    prefer_delta = float(request_payload.max_delta or 0.55)
    scored_candidates: List[Dict[str, Any]] = []
    for contract in candidates:
        result = score_contract(contract, prefer_delta=prefer_delta)
        contract["liquidity_score"] = round(result.score, 4)
        contract["liquidity_components"] = {k: round(v, 4) for k, v in result.components.items()}
        scored_candidates.append(contract)
    scored_candidates.sort(key=lambda item: item.get("liquidity_score", 0.0), reverse=True)

    plan_anchor = getattr(request_payload, "plan_anchor", None)
    best = [_enrich_contract_with_plan(dict(contract), plan_anchor, risk_amount) for contract in scored_candidates[:3]]
    alternatives = [_enrich_contract_with_plan(dict(contract), plan_anchor, risk_amount) for contract in scored_candidates[3:10]]

    # Compact table view for UI rendering
    table_rows: List[Dict[str, Any]] = []
    for row in best[:6]:
        try:
            label = row.get("label") or row.get("symbol") or ""
            # Preserve a compact, ordered shape: label, dte, strike, price, bid, ask, delta, theta, iv, spread_pct, oi, liquidity_score
            price_val = row.get("price") or row.get("mid") or row.get("mark")
            if isinstance(price_val, (int, float)):
                price_val = round(float(price_val), 2)
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
                "liquidity_score": row.get("liquidity_score") or row.get("tradeability"),
            })
        except Exception:
            continue

    example_leg = build_example_leg(scored_candidates[0]) if scored_candidates else None

    payload = {
        "symbol": symbol,
        "side": side,
        "style": style,
        "risk_amount": risk_amount,
        "filters": filters,
        "relaxed_filters": relaxed,
        "best": best,
        "alternatives": alternatives,
        "table": table_rows,
        "example_leg": example_leg,
        "example": example_leg,
    }
    payload["session"] = session_info
    if underlying_price_asof is not None:
        payload["underlying_price_asof"] = underlying_price_asof
    return payload


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
    session_info = _session_block()
    if not data.get("as_of") and session_info.get("as_of"):
        data["as_of"] = session_info["as_of"]
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
    return ChartLinks(interactive=url, session=session_info)


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

    session_info = _session_block()
    as_of_ts = _session_asof_timestamp(session_info)

    frame = get_candles(symbol, interval_normalized, lookback=lookback)
    if frame.empty:
        raise HTTPException(status_code=502, detail=f"No market data available for {symbol.upper()} ({interval_normalized}).")

    if session_info.get("status") != "open" and as_of_ts is not None:
        limited = frame[frame["time"] <= as_of_ts]
        if not limited.empty:
            frame = limited

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
            if session_info.get("status") != "open" and as_of_ts is not None:
                limited_bench = bench_frame[bench_frame["time"] <= as_of_ts]
                if not limited_bench.empty:
                    bench_frame = limited_bench
            benchmark_history = bench_frame.set_index("time")
        except HTTPException:
            benchmark_history = None

    chain_df: pd.DataFrame | None = None
    polygon_bundle: Dict[str, Any] | None = None
    settings = get_settings()
    if settings.polygon_api_key:
        if session_info.get("status") != "open" and as_of_ts is not None:
            chain_df = await fetch_polygon_option_chain_asof(symbol, as_of_ts.to_pydatetime())
        else:
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
    if session_info.get("as_of"):
        chart_params["as_of"] = session_info["as_of"]
    response["charts"] = {"params": {key: str(value) for key, value in chart_params.items()}}
    response["session"] = session_info
    return response


@gpt.get("/widgets/{kind}", summary="Generate lightweight dashboard widgets")
async def gpt_widget(kind: str, symbol: str | None = None, user: AuthedUser = Depends(require_api_key)) -> Dict[str, Any]:
    if kind == "ticker_wedge" and symbol:
        payload = {
            "type": "ticker_wedge",
            "symbol": symbol.upper(),
            "pattern": "rising_wedge",
            "confidence": 0.72,
            "levels": {"support": 98.4, "resistance": 102.6},
        }
        payload["session"] = _session_block()
        return payload
    raise HTTPException(status_code=404, detail="Unknown widget kind or missing params")


# Register GPT endpoints with the application
app.include_router(tv_api)
app.include_router(gpt)
app.include_router(charts_router)
app.include_router(gpt_sentiment_router)


# ---------------------------------------------------------------------------
# Platform health endpoints & Assistant API
# ---------------------------------------------------------------------------


CHART_BASE = os.getenv("CHART_BASE", "https://app.fancytrader.io/chart").split("?", 1)[0].rstrip("?")


def _validate_chart_url(url: Optional[str]) -> None:
    if not canonical_chart_url(url):
        raise HTTPException(status_code=500, detail="chart_url must be canonical.")


def _parse_symbol_query(symbols: Optional[str]) -> List[str]:
    if not symbols:
        return []
    tokens = [token.strip().upper() for token in symbols.split(",") if token.strip()]
    return list(dict.fromkeys(tokens))


def _parse_cursor(cursor: Optional[str]) -> Optional[pd.Timestamp]:
    if not cursor:
        return None
    try:
        ts = pd.Timestamp(cursor)
    except Exception:
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


def _confidence_value(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, numeric))


def _compact_guard(setups: List[Dict[str, Any]], budget_kb: int = 80) -> List[Dict[str, Any]]:
    import json as _json

    def fits(payload: Dict[str, Any]) -> bool:
        try:
            return len(_json.dumps(payload)) < budget_kb * 1024
        except Exception:
            return True

    payload = {"setups": setups}
    if fits(payload):
        return setups
    trimmed = [dict(item) for item in setups]
    for item in trimmed:
        item.pop("rationale", None)
    if fits({"setups": trimmed}):
        return trimmed
    for item in trimmed:
        item.pop("confluence", None)
    return trimmed


async def _attach_options_example_dict(setup: Dict[str, Any]) -> Dict[str, Any]:
    options_block = setup.get("options") or {}
    if options_block.get("example"):
        return setup
    try:
        example = await best_contract_example(
            symbol=setup.get("symbol"),
            style=setup.get("style"),
            as_of=setup.get("as_of"),
        )
    except Exception:
        example = None
    if example:
        options_block.setdefault("style_horizon_applied", setup.get("style"))
        options_block.setdefault("dte_window", "auto")
        options_block["example"] = example
        setup["options"] = options_block
    return setup


def _maybe_create_hedge(setup: SetupMeta) -> Optional[SetupMeta]:
    if setup.direction == "long":
        if setup.stop is None or not setup.targets:
            return None
        atr = setup.atr_used or 0.0
        entry = setup.entry.level
        stop_break = setup.stop - max(0.1, atr * 0.2)
        first_target = setup.targets[0] if setup.targets else entry - max(0.5, atr)
        hedge = setup.model_copy(update={
            "plan_id": f"{setup.plan_id}-H",
            "direction": "short",
            "entry": EntryMeta(type="break", level=round(stop_break, 2)),
            "stop": round(entry + max(0.1, (first_target - setup.stop) * 0.5), 2),
            "targets": [round(first_target, 2)],
            "confidence": min(0.5, setup.confidence * 0.8),
            "rationale": f"Hedge: break below {setup.stop:.2f} invalidates long structure",
            "options": None,
        })
        return hedge
    if setup.direction == "short":
        if setup.stop is None or not setup.targets:
            return None
        atr = setup.atr_used or 0.0
        entry = setup.entry.level
        stop_break = setup.stop + max(0.1, atr * 0.2)
        first_target = setup.targets[0] if setup.targets else entry + max(0.5, atr)
        hedge = setup.model_copy(update={
            "plan_id": f"{setup.plan_id}-H",
            "direction": "long",
            "entry": EntryMeta(type="break", level=round(stop_break, 2)),
            "stop": round(entry - max(0.1, (setup.stop - first_target) * 0.5), 2),
            "targets": [round(first_target, 2)],
            "confidence": min(0.5, setup.confidence * 0.8),
            "rationale": f"Hedge: reclaim above {setup.stop:.2f} invalidates short structure",
            "options": None,
        })
        return hedge
    return None


@app.post("/api/v1/assistant/exec")
async def exec_assistant(
    request_payload: ExecRequest,
    request: Request,
    format: str = Query(default="json"),
    style: Optional[str] = Query(default=None),
    limit: int = Query(default=3, ge=1, le=10),
    symbols: Optional[str] = Query(default=None),
    include_series: str = Query(default="none"),
    include_inputs: str = Query(default="compact"),
    user: AuthedUser = Depends(require_api_key),
) -> Any:
    query_symbols = _parse_symbol_query(symbols)
    body_symbols = [token.upper() for token in (request_payload.symbols or [])]
    tickers = query_symbols or body_symbols
    tickers = [t for t in tickers if t]
    style_param = style or request_payload.style
    auto_universe = False
    auto_meta: Dict[str, Any] = {}
    if not tickers:
        tickers, auto_meta = _autofill_symbols(style_param)
        auto_universe = True
        logger.info(
            "No symbols supplied; using auto universe",
            extra={"symbols": tickers, "style": style_param, "meta": auto_meta},
        )
    limit_param = max(1, min(limit, request_payload.limit or limit))
    ui_mode = request_payload.ui_mode or ("chat" if format.lower() == "text" else "api")

    include_series = "none"
    session_state = session_now()
    session_info = session_state.to_dict()
    session_meta = SessionMeta.model_validate(session_info)

    universe = ScanUniverse(tickers=tickers, style=style_param)
    scan_results = await gpt_scan(
        universe,
        request,
        user,
        auto_universe=auto_universe,
        auto_meta=auto_meta,
    )

    style_token = _output_style_token(style_param) if style_param else None

    def _matches_style(result: Dict[str, Any]) -> bool:
        if style_token is None:
            return True
        result_style = result.get("style") or ((result.get("plan") or {}).get("style"))
        return _output_style_token(result_style) == style_token

    filtered_results = [item for item in scan_results if _matches_style(item)]
    selected_results = filtered_results[:limit_param]

    setup_dicts: List[Dict[str, Any]] = []
    for result in selected_results:
        symbol = (result.get("symbol") or "").upper()
        if not symbol:
            continue
        plan_block = result.get("plan") or {}
        plan_request = PlanRequest(symbol=symbol, style=style_param or result.get("style"), plan_id=plan_block.get("plan_id"))
        try:
            plan_response = await gpt_plan(plan_request, request, user)
        except HTTPException:
            continue
        try:
            main_setup = _build_setup_from_plan_response(plan_response, session_info)
        except Exception:
            logger.exception("failed to build setup for %s", symbol)
            continue
        for candidate in filter(None, [main_setup, _maybe_create_hedge(main_setup)]):
            data = candidate.model_dump()
            data["as_of"] = session_meta.as_of
            entry_payload = data.get("entry") or {}
            stop_value = data.get("stop")
            plan_version = data.get("version")
            data["chart_url"] = build_chart_url(
                CHART_BASE,
                symbol=data.get("symbol"),
                plan_id=data.get("plan_id"),
                as_of=data.get("as_of"),
                entry=entry_payload,
                stop=float(stop_value) if stop_value is not None else 0.0,
                targets=data.get("targets"),
                plan_version=str(plan_version) if plan_version is not None else None,
            )
            data = await _attach_options_example_dict(data)
            _validate_chart_url(data.get("chart_url"))
            setup_dicts.append(data)
            async with _ACTIVE_SETUPS_LOCK:
                _ACTIVE_SETUPS[data["plan_id"]] = dict(data)

    setup_dicts = _compact_guard(setup_dicts)
    setup_models = [SetupMeta.model_validate(item) for item in setup_dicts]

    exec_payload = ExecResponse(session=session_meta, count=len(setup_models), setups=setup_models)

    if format.lower() == "json" and ui_mode != "chat":
        return exec_payload

    lines: List[str] = []
    banner_text = session_info.get("banner") or f"Session status: {session_meta.status}"
    lines.append(f"{banner_text} (as_of {session_meta.as_of})")
    summary_style = style_token or "mixed"
    lines.append(f"{exec_payload.count} setups returned (style={summary_style}).")
    for setup in setup_dicts:
        lines.append(json.dumps(setup, separators=(",", ":"), sort_keys=True))
        chart_url = setup.get("chart_url")
        if isinstance(chart_url, str) and chart_url:
            lines.append(f"Open chart: {chart_url}")
    lines.append("Notes: Charts generated by Trading Coach backend; all times ET.")
    return PlainTextResponse("\n".join(lines))


@app.get("/api/v1/context")
async def context_snapshot(
    symbol: str = Query(..., min_length=1),
    as_of: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    session_state = session_now()
    as_of_value = as_of or session_state.as_of
    context = build_context_block(symbol, as_of_value)
    return {
        "symbol": symbol.upper(),
        "as_of": as_of_value,
        "context": context,
    }


@app.get("/api/v1/symbol/{symbol}/series")
async def symbol_series(
    symbol: str,
    tf: str = Query("1m"),
    limit: int = Query(1500, ge=50, le=5000),
    cursor: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    interval = normalize_interval(tf)
    frame = get_candles(symbol, interval, lookback=max(limit * 2, 200))
    if frame.empty:
        raise HTTPException(status_code=404, detail="No data available")
    df = frame.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time").sort_index()
    cursor_ts = _parse_cursor(cursor)
    if cursor_ts is not None:
        df = df.loc[df.index < cursor_ts]
    if df.empty:
        return {"symbol": symbol.upper(), "interval": interval, "bars": [], "next_cursor": None}
    window = df.tail(limit)
    bars = [
        {
            "time": idx.isoformat(),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
            "volume": float(row.get("volume") or 0.0),
        }
        for idx, row in window.iterrows()
    ]
    next_cursor = None
    if len(window) == limit and len(df) > limit:
        next_cursor = window.index[0].isoformat()
    return {
        "symbol": symbol.upper(),
        "interval": interval,
        "count": len(bars),
        "bars": bars,
        "next_cursor": next_cursor,
    }


@app.get("/api/v1/symbol/{symbol}/indicators")
async def symbol_indicators(
    symbol: str,
    tf: str = Query("1m"),
    limit: int = Query(1500, ge=50, le=5000),
    cursor: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    interval = normalize_interval(tf)
    frame = get_candles(symbol, interval, lookback=max(limit * 2, 200))
    if frame.empty:
        raise HTTPException(status_code=404, detail="No data available")
    df = frame.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    df = df.set_index("time").sort_index()
    cursor_ts = _parse_cursor(cursor)
    if cursor_ts is not None:
        df = df.loc[df.index < cursor_ts]
    if df.empty:
        return {"symbol": symbol.upper(), "interval": interval, "series": [], "next_cursor": None}
    window = df.tail(limit)
    ema9_series = ema(window["close"], 9) if len(window) >= 9 else pd.Series(dtype=float)
    ema20_series = ema(window["close"], 20) if len(window) >= 20 else pd.Series(dtype=float)
    ema50_series = ema(window["close"], 50) if len(window) >= 50 else pd.Series(dtype=float)
    vwap_series = vwap(window["close"], window["volume"])
    atr_series = atr(window["high"], window["low"], window["close"], 14)
    adx_series = adx(window["high"], window["low"], window["close"], 14)
    indicators = {
        "ema9": _series_points(ema9_series),
        "ema20": _series_points(ema20_series),
        "ema50": _series_points(ema50_series),
        "vwap": _series_points(vwap_series),
        "atr14": _series_points(atr_series),
        "adx14": _series_points(adx_series),
    }
    next_cursor = None
    if len(window) == limit and len(df) > limit:
        next_cursor = window.index[0].isoformat()
    return {
        "symbol": symbol.upper(),
        "interval": interval,
        "count": len(window),
        "indicators": indicators,
        "next_cursor": next_cursor,
    }


@app.get("/healthz", summary="Readiness probe used by Railway")
async def healthz() -> Dict[str, Any]:
    payload = {"status": "ok"}
    payload["session"] = _session_block()
    return payload


@app.get("/", summary="Service metadata")
async def root() -> Dict[str, Any]:
    payload = {
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
    payload["session"] = _session_block()
    return payload
