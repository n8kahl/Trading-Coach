"""Trading Coach backend tailored for GPT Actions integrations.

The service now focuses on a lean surface area that lets a custom GPT pull
ranked setups (with richer level-aware targets) and render interactive charts
driven by the same OHLCV data. Legacy endpoints for watchlists, notes, and
trade-following have been removed to keep the API aligned with the coaching
workflow.
"""

from __future__ import annotations

import asyncio
import base64
import json
import math
import logging
import time
import uuid
import re
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Set, Tuple, cast
from collections import Counter
import copy

import httpx
import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request, Query, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, AliasChoices
from pydantic import ConfigDict
from urllib.parse import urlencode, quote, urlsplit, urlunsplit, parse_qsl
from fastapi.responses import StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware

from .config import get_settings
from .schemas import ScanRequest, ScanPage, ScanCandidate, ScanFilters
from .universe import expand_universe
from .app.engine.index_common import INDEX_BASE_TICKERS
from .realtime_bars import PolygonRealtimeBarStreamer
from .calculations import atr, ema, bollinger_bands, keltner_channels, adx, vwap
from .charts_api import router as charts_router, get_candles, normalize_interval
from .gpt_sentiment import router as gpt_sentiment_router
from .scanner import (
    Signal,
    scan_market,
    _apply_tp_logic,
    _base_targets_for_style,
    _normalize_trade_style,
    _runner_config,
    _prepare_symbol_frame,
    _build_context,
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
from .app.engine import (
    build_target_profile,
    build_structured_plan,
    IndexPlanningMode,
    IndexPlanner,
    GammaSnapshot,
)
from .app.engine.execution_profiles import ExecutionContext as PlanExecutionContext, refine_plan as refine_execution_plan
from .app.engine.options_select import score_contract, best_contract_example
from .app.middleware import SessionMiddleware, get_session
from .app.routers.session import router as session_router
from .app.services import parse_session_as_of, make_chart_url, build_plan_layers, get_precision
from .app.providers.macro import get_event_window
from .app.providers.universe import load_universe
from .context_overlays import compute_context_overlays
from .db import (
    ensure_schema as ensure_db_schema,
    fetch_idea_snapshot as db_fetch_idea_snapshot,
    store_idea_snapshot as db_store_idea_snapshot,
)
from .data_sources import fetch_polygon_ohlcv, fetch_yahoo_ohlcv
from .symbol_streamer import SymbolStreamCoordinator, fetch_live_quote
from .live_plan_engine import LivePlanEngine
from .statistics import get_style_stats
from .ranking import (
    Features as RankingFeatures,
    Scored as RankedCandidate,
    Style as RankingStyle,
    ACTIONABILITY_GATE,
    diversify as diversify_ranked,
    rank as rank_candidates,
)
from .scan_features import Metrics, MetricsContext, compute_metrics_fast
from .indicators import get_indicator_bundle
from zoneinfo import ZoneInfo

from .market_clock import MarketClock

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
    "focus",
    "center_time",
    "data_source",
    "data_mode",
    "data_age_ms",
    "last_update",
    "market_status",
    "session_status",
    "session_phase",
    "session_banner",
}

_METRIC_COUNTER: Counter[str] = Counter()


def _record_metric(name: str, **labels: str) -> int:
    key_parts = [name] + [f"{key}={labels[key]}" for key in sorted(labels)]
    key = "|".join(key_parts)
    _METRIC_COUNTER[key] += 1
    return _METRIC_COUNTER[key]

_STYLE_DELTA_PREF: Dict[str, float] = {
    "scalp": 0.55,
    "intraday": 0.50,
    "swing": 0.45,
    "leap": 0.35,
    "leaps": 0.35,
}


def _prefer_delta_for_style(style: Optional[str]) -> float:
    token = (style or "intraday").lower()
    if token == "leap":
        token = "leaps"
    return _STYLE_DELTA_PREF.get(token, 0.5)


DATA_SYMBOL_ALIASES: Dict[str, List[str]] = {
    "SPX": ["I:SPX", "X:SPX", "^GSPC"],
    "^SPX": ["^GSPC"],
    "INDEX:SPX": ["^GSPC"],
    "SP500": ["^GSPC"],
    "I:SPX": ["SPX"],
    "X:SPX": ["SPX"],
    "NDX": ["I:NDX", "X:NDX", "^NDX"],
    "I:NDX": ["NDX"],
    "X:NDX": ["NDX"],
    "OEX": ["SPX", "^OEX", "SPY"],
    "^OEX": ["SPX", "SPY"],
    "INDEX:OEX": ["SPX", "SPY"],
}

_PLAN_SYMBOL_PATTERN = re.compile(r"^[A-Z][A-Z0-9\.:]{0,9}$")

_MARKET_DATA_CACHE: Dict[Tuple[str, str], Dict[str, Any]] = {}
_MARKET_DATA_CACHE_TTL = 300.0  # seconds

_CHART_URL_CACHE_TTL = 600.0
_CHART_URL_CACHE: Dict[str, Tuple[float, str]] = {}

_INDICATOR_CACHE_TTL = 60.0
_INDICATOR_CACHE: Dict[Tuple[str, float], Tuple[float, Dict[str, Any]]] = {}
_INDICATOR_FETCH_TIMEOUT = 1.2

_FUTURES_PROXY_MAP: Dict[str, str] = {
    "es_proxy": "SPY",
    "nq_proxy": "QQQ",
    "ym_proxy": "DIA",
    "rty_proxy": "IWM",
    "vix": "CBOE:VIX",
}

_FUTURES_CACHE: Dict[str, Any] = {"data": None, "ts": 0.0}

_SCAN_SYMBOL_REGISTRY: Dict[Tuple[str, str, str], List[str]] = {}
_SCAN_STYLE_ANY = "__any__"

_MARKET_CLOCK = MarketClock()

# Default symbols used when universe expansion fails (closed session or outage)
_FROZEN_DEFAULT_UNIVERSE: List[str] = [
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    "AAPL",
    "MSFT",
    "NVDA",
    "TSLA",
    "META",
    "AMZN",
    "GOOG",
    "NFLX",
    "AMD",
    "CRM",
    "COST",
    "JPM",
    "XLF",
    "XLK",
    "SMH",
    "TLT",
]

ETF_EVENT_SYMBOLS: set[str] = {
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    "TLT",
    "SMH",
    "XLF",
    "XLE",
    "XLK",
    "XLV",
    "XLY",
    "XLI",
    "XLB",
    "XLU",
    "XLRE",
    "HYG",
    "GDX",
    "USO",
    "GLD",
    "SLV",
    "SPX",
    "NDX",
}

_STRATEGY_CHART_HINTS: Dict[str, Tuple[str, str]] = {
    "orb_retest": ("1m", "Wait for a 1 minute break and retest of the opening range boundary before committing."),
    "power_hour_trend": ("5m", "Manage the trade on 5 minute closes while price holds trend alignment into the close."),
    "vwap_avwap": ("1m", "Use 1 minute closes to confirm the reclaim above VWAP/AVWAP cluster before scaling."),
    "gap_fill_open": ("1m", "Watch 1 minute rejection of prior close to trigger the gap-fill continuation."),
    "midday_mean_revert": ("1m", "Fade extensions with 1 minute bars; wait for a VWAP reclaim before adding risk."),
    "baseline_auto": ("5m", "Default to 5 minute management — respect plan entry/stop on close."),
}

_STYLE_CHART_HINTS: Dict[str, Tuple[str, str]] = {
    "scalp": ("1m", "Use 1 minute closes to validate the scalp trigger and manage stops."),
    "intraday": ("5m", "Manage risk on 5 minute closes relative to the plan levels."),
    "swing": ("1h", "Focus on 1 hour structure; keep stops below the HTF pivot."),
    "leap": ("d", "Use daily bars for confirmation and management."),
}


def _cache_summary(entries: Iterable[Tuple[float, Any]]) -> Dict[str, Any]:
    now = time.monotonic()
    ages = [max(now - stamp, 0.0) for stamp, _ in entries]
    if not ages:
        return {"size": 0, "max_age_seconds": 0.0, "min_age_seconds": 0.0}
    return {
        "size": len(ages),
        "max_age_seconds": round(max(ages), 3),
        "min_age_seconds": round(min(ages), 3),
    }


def _session_payload_from_request(request: Request) -> Dict[str, Any]:
    return dict(get_session(request))


def _market_snapshot_payload(
    session_payload: Mapping[str, Any] | None = None,
    *,
    simulate_open: bool = False,
) -> Tuple[Dict[str, Any], Dict[str, Any], datetime, bool]:
    snapshot = _MARKET_CLOCK.snapshot()
    if session_payload is None:
        session_payload = get_session()
    else:
        session_payload = dict(session_payload)
    as_of_dt = parse_session_as_of(session_payload) or _MARKET_CLOCK.last_rth_close(at=snapshot.now_et)
    frozen = str(session_payload.get("status")).lower() != "open"
    simulated = bool(simulate_open)
    effective_is_open = (not frozen) or simulated
    market_payload = {
        "status": snapshot.status,
        "session": snapshot.session,
        "session_state": session_payload,
        "now_et": snapshot.now_et.isoformat(),
        "next_open_et": snapshot.next_open_et.isoformat() if snapshot.next_open_et else None,
        "next_close_et": snapshot.next_close_et.isoformat() if snapshot.next_close_et else None,
        "simulated_open": simulated,
    }
    data_payload = {
        "as_of_ts": int(as_of_dt.timestamp() * 1000),
        "frozen": frozen,
        "ok": True,
        "session_state": session_payload,
        "simulated_open": simulated,
    }
    return market_payload, data_payload, as_of_dt, effective_is_open


def _format_simulated_banner(as_of: datetime) -> str:
    if as_of.tzinfo is None:
        as_of = as_of.replace(tzinfo=timezone.utc)
    local = as_of.astimezone(ZoneInfo("America/New_York"))
    return f"Simulated live — analysis as of {local.strftime('%Y-%m-%d %H:%M ET')}"


def _resolve_simulate_open(request: Request, *, explicit_value: bool, explicit_field_set: bool) -> bool:
    header_value = request.headers.get("X-Simulate-Open")
    header_flag = False
    if header_value is not None:
        token = header_value.strip().lower()
        header_flag = token in {"1", "true", "yes", "on"}
    if explicit_field_set:
        return bool(explicit_value)
    return header_flag


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


def _macro_event_block(symbol: str, session_state: Mapping[str, Any] | None) -> Dict[str, Any] | None:
    if symbol.upper() not in ETF_EVENT_SYMBOLS:
        return None
    as_of = None
    if session_state and isinstance(session_state, Mapping):
        as_of = session_state.get("as_of")
    try:
        window = get_event_window(as_of)
    except Exception:  # pragma: no cover - defensive
        logger.debug("macro event window fetch failed", exc_info=True)
        return None
    if not window:
        return None
    upcoming = window.get("upcoming") or []
    active = window.get("active") or []
    if not upcoming and not active:
        return None

    def _find_minutes(tokens: Tuple[str, ...]) -> Optional[int]:
        for collection in (active, upcoming):
            for item in collection:
                name = str(item.get("name") or "").lower()
                if any(token in name for token in tokens):
                    minutes = item.get("minutes")
                    try:
                        return int(minutes)
                    except (TypeError, ValueError):
                        continue
        return None

    block: Dict[str, Any] = {
        "label": "macro_window",
        "source": "macro_window",
        "within_event_window": bool(active),
        "active": active,
        "upcoming": upcoming,
    }
    block["next_fomc_minutes"] = _find_minutes(("fomc", "fed"))
    block["next_cpi_minutes"] = _find_minutes(("cpi", "inflation"))
    block["next_nfp_minutes"] = _find_minutes(("payroll", "jobs"))
    if window.get("min_minutes_to_event") is not None:
        block["min_minutes_to_event"] = window.get("min_minutes_to_event")
    return block


@dataclass(slots=True)
class ScanContext:
    as_of: datetime
    label: Literal["live", "frozen"]
    is_open: bool
    data_timeframe: str
    market_meta: Dict[str, Any]
    data_meta: Dict[str, Any]
    simulate_open: bool = False

    @property
    def as_of_iso(self) -> str:
        return self.as_of.isoformat()


def _session_tracking_id(session_payload: Mapping[str, Any]) -> str | None:
    status = session_payload.get("status")
    as_of = session_payload.get("as_of")
    style = session_payload.get("style")
    components = [
        str(status or "").strip().lower(),
        str(as_of or "").strip(),
        str(style or "").strip().lower(),
    ]
    token = "|".join(components).strip("|")
    return token or None


def _validate_level_invariants(
    direction: str | None,
    entry: Any,
    stop: Any,
    targets: Sequence[Any],
) -> tuple[bool, str | None]:
    dir_token = (direction or "").strip().lower()
    if dir_token not in {"long", "short"}:
        return False, "invalid_direction"
    try:
        entry_f = float(entry)
        stop_f = float(stop)
        target_values = [float(tp) for tp in targets]
    except (TypeError, ValueError):
        return False, "non_numeric_levels"
    if not target_values:
        return False, "missing_targets"
    if not math.isfinite(entry_f) or not math.isfinite(stop_f) or not all(math.isfinite(tp) for tp in target_values):
        return False, "non_finite_levels"
    if dir_token == "long":
        if not stop_f < entry_f:
            return False, "stop_not_below_entry"
        prev = entry_f
        for idx, tp in enumerate(target_values, start=1):
            if tp <= prev:
                return False, f"tp{idx}_not_above_previous"
            prev = tp
    else:
        if not stop_f > entry_f:
            return False, "stop_not_above_entry"
        prev = entry_f
        for idx, tp in enumerate(target_values, start=1):
            if tp >= prev:
                return False, f"tp{idx}_not_below_previous"
            prev = tp
    return True, None


def _apply_em_cap(
    direction: str | None,
    entry: Any,
    targets: Sequence[Any],
    expected_move: Any,
    factor: float,
) -> tuple[List[float], bool]:
    dir_token = (direction or "").strip().lower()
    try:
        entry_f = float(entry)
    except (TypeError, ValueError):
        return [float(tp) for tp in targets if tp is not None], False
    try:
        em_val = float(expected_move)
    except (TypeError, ValueError):
        em_val = float("nan")
    if dir_token not in {"long", "short"} or not math.isfinite(em_val) or em_val <= 0 or factor <= 0:
        return [float(tp) for tp in targets if tp is not None], False
    cap_distance = em_val * factor
    cap_level = entry_f + cap_distance if dir_token == "long" else entry_f - cap_distance
    adjusted: List[float] = []
    em_used = False
    for tp in targets:
        try:
            tp_f = float(tp)
        except (TypeError, ValueError):
            continue
        original = tp_f
        if dir_token == "long" and tp_f > cap_level:
            tp_f = cap_level
        elif dir_token == "short" and tp_f < cap_level:
            tp_f = cap_level
        if adjusted:
            if dir_token == "long" and tp_f <= adjusted[-1]:
                tp_f = min(cap_level, adjusted[-1] + 0.01)
            elif dir_token == "short" and tp_f >= adjusted[-1]:
                tp_f = max(cap_level, adjusted[-1] - 0.01)
        if dir_token == "long":
            tp_f = min(tp_f, cap_level)
        else:
            tp_f = max(tp_f, cap_level)
        tp_f = round(tp_f, 4)
        if not math.isclose(tp_f, round(original, 4)):
            em_used = True
        adjusted.append(tp_f)
    return adjusted, em_used


@dataclass(slots=True)
class ScanPrep:
    candidate: ScanCandidate
    metrics: Metrics
    features: RankingFeatures


def _scan_timeframe_for_style(style: str) -> str:
    mapping = {"scalp": "1", "intraday": "5", "swing": "60", "leaps": "D"}
    return mapping.get(style.lower(), "5")


def _cached_chart_url(cache_key: str, builder: Callable[[], str]) -> str:
    now = time.monotonic()
    cached = _CHART_URL_CACHE.get(cache_key)
    if cached and now - cached[0] < _CHART_URL_CACHE_TTL:
        return cached[1]
    url = builder()
    _CHART_URL_CACHE[cache_key] = (now, url)
    return url


def encode_cursor(value: int) -> str:
    payload = str(max(0, value)).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii")


def decode_cursor(token: str | None) -> int:
    if not token:
        return 0
    try:
        raw = base64.urlsafe_b64decode(token.encode("ascii")).decode("utf-8")
        parsed = int(raw)
        return parsed if parsed >= 0 else 0
    except Exception:
        return 0


def _indicator_bundle(symbol: str, history: pd.DataFrame) -> Dict[str, Any]:
    last_ts = history.index[-1]
    if not isinstance(last_ts, pd.Timestamp):
        last_ts = pd.Timestamp(last_ts, tz="UTC")
    stamp = float(last_ts.timestamp())
    key = (symbol.upper(), stamp)
    now = time.monotonic()
    cached = _INDICATOR_CACHE.get(key)
    if cached and now - cached[0] < _INDICATOR_CACHE_TTL:
        return cached[1]
    key_levels = _extract_key_levels(history)
    snapshot = _build_market_snapshot(history, key_levels)
    bundle = {
        "key_levels": key_levels,
        "snapshot": snapshot,
        "indicators": snapshot.get("indicators") or {},
    }
    _INDICATOR_CACHE[key] = (now, bundle)
    return bundle


async def _indicator_metrics(symbol: str, history: pd.DataFrame) -> Dict[str, Any]:
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(get_indicator_bundle, symbol, history),
            timeout=_INDICATOR_FETCH_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.warning("Indicator bundle timeout for %s", symbol)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Indicator bundle failed for %s: %s", symbol, exc)
    return {}


def _normalize_chart_symbol(value: str) -> str:
    token = (value or "").strip()
    if ":" in token:
        return token
    return token.upper()


def _tv_symbol(value: str) -> str:
    """Normalize symbols for the TradingView-style chart route.

    Ensures index underlyings resolve to Polygon's index tickers to avoid
    host defaults (e.g., AAPL when symbol is missing).
    """
    tok = (value or "").strip().upper()
    spx_aliases = {"SPX", "^GSPC", "X:SPX", "INDEX:SPX", "SP500"}
    ndx_aliases = {"NDX", "^NDX", "X:NDX", "INDEX:NDX"}
    if tok in spx_aliases or tok == "I:SPX":
        return "I:SPX"
    if tok in ndx_aliases or tok == "I:NDX":
        return "I:NDX"
    return tok


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


def _evaluate_scan_context(
    asof_policy: str,
    style: str,
    session_payload: Mapping[str, Any],
    *,
    simulate_open: bool = False,
) -> tuple[ScanContext, str | None, Dict[str, Any]]:
    market_meta, data_meta, as_of_dt, effective_is_open = _market_snapshot_payload(
        session_payload,
        simulate_open=simulate_open,
    )
    policy = asof_policy.lower()
    label: Literal["live", "frozen"]
    banner_parts: List[str] = []
    session_status = str(session_payload.get("status") or "").lower()
    session_is_open = session_status == "open"

    if policy == "live":
        label = "live" if effective_is_open else "frozen"
        if not session_is_open:
            banner_parts.append("Market closed — using last known good data.")
    elif policy == "frozen":
        label = "frozen"
        banner_parts.append("Frozen planning context requested.")
    else:  # live_or_lkg
        label = "live" if effective_is_open else "frozen"
        if not session_is_open:
            banner_parts.append("Live feed unavailable — using frozen data.")

    if simulate_open and not session_is_open:
        banner_parts.append(_format_simulated_banner(as_of_dt))

    banner: str | None
    if banner_parts:
        unique_parts = list(dict.fromkeys(part for part in banner_parts if part))
        banner = " • ".join(unique_parts)
    else:
        banner = None

    data_timeframe = _scan_timeframe_for_style(style)
    context = ScanContext(
        as_of=as_of_dt,
        label=label,
        is_open=effective_is_open,
        simulate_open=bool(simulate_open),
        data_timeframe=data_timeframe,
        market_meta=market_meta,
        data_meta=dict(data_meta),
    )
    dq = dict(data_meta)
    dq["simulated_open"] = bool(simulate_open)
    return context, banner, dq


def _build_stub_chart_url(
    request: Request,
    *,
    symbol: str,
    direction: str | None,
    entry: float | None,
    stop: float | None,
    targets: Sequence[float],
    interval: str,
    levels: Sequence[str] | None = None,
) -> str | None:
    if entry is None or stop is None or not targets:
        return None
    params: Dict[str, Any] = {
        "symbol": symbol.upper(),
        "interval": interval,
        "direction": direction or "long",
        "entry": f"{entry:.2f}",
        "stop": f"{stop:.2f}",
        "tp": ",".join(f"{tp:.2f}" for tp in targets[:3]),
        "focus": "plan",
        "center_time": "latest",
        "scale_plan": "auto",
    }
    if levels:
        params["levels"] = ";".join(levels)
    cache_key = f"chart:{symbol}:{direction}:{params['entry']}:{params['stop']}:{params['tp']}:{interval}"
    return _cached_chart_url(cache_key, lambda: _build_tv_chart_url(request, params))


def _candidate_reasons(signal: Signal) -> List[str]:
    reasons: List[str] = []
    description = (signal.description or "").strip()
    if description:
        reasons.append(description)
    features = signal.features or {}
    for key in ("plan_confidence_reasons", "plan_confidence_factors", "confidence_reasons", "why_this_works"):
        value = features.get(key)
        if isinstance(value, str):
            text = value.strip()
            if text and text not in reasons:
                reasons.append(text)
        elif isinstance(value, (list, tuple)):
            for item in value:
                text = str(item).strip()
                if text and text not in reasons:
                    reasons.append(text)
    return reasons[:5]


def _relative_volume(frame: pd.DataFrame) -> float:
    if "volume" not in frame.columns or frame.empty:
        return 1.0
    latest = float(frame["volume"].iloc[-1] or 0.0)
    baseline = float(frame["volume"].tail(20).mean() or 0.0)
    if baseline <= 0:
        return 1.0
    return round(max(latest / baseline, 0.0), 3)


def _liquidity_score(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    close_col = frame.get("close")
    vol_col = frame.get("volume")
    if close_col is None or vol_col is None:
        return 0.0
    try:
        price = float(close_col.iloc[-1])
        avg_vol = float(vol_col.tail(20).mean() or 0.0)
    except Exception:
        return 0.0
    return max(price * avg_vol, 0.0)


async def _build_scan_stub_candidate(
    semaphore: asyncio.Semaphore,
    *,
    signal: Signal,
    history: pd.DataFrame,
    context: ScanContext,
    request: Request,
) -> ScanPrep | None:
    async with semaphore:
        plan = signal.plan
        entry = float(plan.entry) if plan and plan.entry is not None else None
        stop = float(plan.stop) if plan and plan.stop is not None else None
        targets = [float(t) for t in (plan.targets if plan else []) if math.isfinite(t)]
        rr_t1 = float(plan.risk_reward) if plan and plan.risk_reward is not None else None
        confidence = float(plan.confidence) if plan and plan.confidence is not None else None
        direction_hint = plan.direction if plan and plan.direction else (signal.features or {}).get("direction_bias")

        bundle = _indicator_bundle(signal.symbol, history)
        key_levels = bundle["key_levels"]
        snapshot = bundle["snapshot"]
        level_labels = [f"{label}|{value:.2f}" for label, value in key_levels.items() if math.isfinite(value)]

        plan_id = None
        if plan is not None:
            plan_id = _generate_plan_slug(
                signal.symbol,
                _style_for_strategy(signal.strategy_id),
                direction_hint or "long",
                snapshot,
            )

        interval_hint, _ = _chart_hint(signal.strategy_id, _style_for_strategy(signal.strategy_id))
        chart_url = _build_stub_chart_url(
            request,
            symbol=signal.symbol,
            direction=direction_hint,
            entry=entry,
            stop=stop,
            targets=targets,
            interval=_normalize_chart_interval(interval_hint or context.data_timeframe),
            levels=level_labels if level_labels else None,
        )

        fast_indicators = await _indicator_metrics(signal.symbol, history)
        combined_indicators: Dict[str, Any] = dict(fast_indicators)
        combined_indicators.update(bundle.get("snapshot", {}).get("indicators") or {})
        style_token = _ranking_style(_style_for_strategy(signal.strategy_id))
        metrics_context = MetricsContext(
            symbol=signal.symbol.upper(),
            style=style_token,
            as_of=context.as_of,
            is_open=context.is_open,
            simulate_open=context.simulate_open,
            history=history,
            signal=signal,
            plan=plan,
            indicator_bundle=combined_indicators,
            market_meta=dict(context.market_meta),
            data_meta=dict(context.data_meta),
        )
        metrics = compute_metrics_fast(metrics_context.symbol, style_token, metrics_context)
        confidence = metrics.confidence
        entry_distance_pct_val = _safe_number(metrics.entry_distance_pct)
        entry_distance_atr_val = _safe_number(metrics.entry_distance_atr)
        bars_to_trigger_val = _safe_number(metrics.bars_to_trigger)
        actionability_score = _clamp01(metrics.actionability)
        should_gate = (
            actionability_score < ACTIONABILITY_GATE
            and (
                (entry_distance_pct_val is not None and entry_distance_pct_val > 2.0)
                or (entry_distance_atr_val is not None and entry_distance_atr_val > 1.2)
            )
        )
        if should_gate:
            logger.debug(
                "scan_candidate_gated_distance",
                extra={
                    "symbol": signal.symbol,
                    "strategy": signal.strategy_id,
                    "entry_distance_pct": entry_distance_pct_val,
                    "entry_distance_atr": entry_distance_atr_val,
                    "actionability": actionability_score,
                },
            )
            return None
        actionable_soon = False
        if entry_distance_pct_val is not None and entry_distance_pct_val <= 1.0:
            actionable_soon = True
        if entry_distance_atr_val is not None and entry_distance_atr_val <= 0.7:
            actionable_soon = True
        if bars_to_trigger_val is not None and bars_to_trigger_val <= 3:
            actionable_soon = True

        candidate = ScanCandidate(
            symbol=signal.symbol.upper(),
            rank=0,
            score=float(signal.score),
            reasons=_candidate_reasons(signal),
            plan_id=plan_id,
            entry=entry,
            stop=stop,
            tps=targets,
            rr_t1=rr_t1,
            confidence=confidence,
            chart_url=chart_url,
            entry_distance_pct=entry_distance_pct_val,
            entry_distance_atr=entry_distance_atr_val,
            bars_to_trigger=bars_to_trigger_val,
            actionable_soon=actionable_soon,
            source_paths={},
        )
        features = _metrics_to_features(metrics, style_token)
        logger.debug(
            "scan_candidate_metrics",
            extra={
                "symbol": signal.symbol,
                "strategy": signal.strategy_id,
                "score_raw": float(signal.score),
                "actionability": actionability_score,
                "entry_distance_pct": entry_distance_pct_val,
                "entry_distance_atr": entry_distance_atr_val,
                "bars_to_trigger": bars_to_trigger_val,
            },
        )
        return ScanPrep(candidate=candidate, metrics=metrics, features=features)


async def _legacy_scan_stub_payload(
    *,
    signals: List[Signal],
    market_data: Dict[str, pd.DataFrame],
    style_filter: str | None,
    context: ScanContext,
    request: Request,
) -> List[ScanPrep]:
    if not signals:
        return []

    style_token = style_filter.lower() if isinstance(style_filter, str) else None
    best_by_symbol: Dict[str, Signal] = {}
    for signal in signals:
        style = _style_for_strategy(signal.strategy_id)
        if style_token and style_token != style:
            continue
        existing = best_by_symbol.get(signal.symbol)
        if existing is None or signal.score > existing.score:
            best_by_symbol[signal.symbol] = signal

    if not best_by_symbol:
        return []

    semaphore = asyncio.Semaphore(8)
    tasks: List[asyncio.Task[ScanPrep | None]] = []
    for symbol, signal in best_by_symbol.items():
        history = market_data.get(symbol)
        if history is None or history.empty:
            continue
        tasks.append(
            asyncio.create_task(
                _build_scan_stub_candidate(
                    semaphore,
                    signal=signal,
                    history=history,
                    context=context,
                    request=request,
                )
            )
        )

    if not tasks:
        return []

    results = await asyncio.gather(*tasks)
    payload: List[ScanPrep] = []
    for prep in results:
        if prep is None:
            continue
        payload.append(prep)
    return payload


def _apply_scan_filters(
    symbols: List[str],
    market_data: Dict[str, pd.DataFrame],
    filters: ScanFilters | None,
) -> List[str]:
    if not filters:
        return symbols
    filtered = list(symbols)
    if filters.min_rvol is not None:
        threshold = float(filters.min_rvol)
        filtered = [sym for sym in filtered if _relative_volume(market_data[sym]) >= threshold]
    if filters.min_liquidity_rank is not None and filters.min_liquidity_rank > 0:
        ranked = sorted(filtered, key=lambda sym: _liquidity_score(market_data[sym]), reverse=True)
        filtered = ranked[: int(filters.min_liquidity_rank)]
    exclude = {sym.strip().upper() for sym in (filters.exclude or []) if sym}
    if exclude:
        filtered = [sym for sym in filtered if sym not in exclude]
    return filtered


def _effective_scan_filters(filters: ScanFilters | None, *, context: ScanContext) -> ScanFilters | None:
    if not filters:
        return None
    payload = filters.model_dump(exclude_none=True)
    if not context.is_open:
        payload.pop("min_rvol", None)
        payload.pop("min_liquidity_rank", None)
    elif context.label != "live":
        min_rvol = payload.get("min_rvol")
        if min_rvol is not None and min_rvol > 1.0:
            payload["min_rvol"] = 1.0
    if not payload:
        return None
    return ScanFilters(**payload)


def _empty_scan_page(
    request: ScanRequest,
    context: ScanContext,
    *,
    banner: str | None,
    dq: Dict[str, Any],
    session: Dict[str, Any],
) -> ScanPage:
    meta = {
        "style": request.style,
        "limit": request.limit,
        "universe_size": 0,
    }
    if context.simulate_open:
        meta["simulated_open"] = True
    return ScanPage(
        as_of=context.as_of_iso,
        planning_context=context.label,
        banner=banner,
        meta=meta,
        candidates=[],
        data_quality=dq,
        session=session,
        phase="scan",
        count_candidates=0,
        next_cursor=None,
    )


async def _expand_universe_tokens(symbols: List[str], *, style: str | None, limit: int) -> List[str]:
    """Expand synthetic universe tokens (e.g., TOP_ACTIVE_SETUPS) into real tickers."""
    if not symbols:
        return []

    expanded: List[str] = []
    for symbol in symbols:
        token = (symbol or "").strip().upper()
        if not token:
            continue
        if token in {"TOP_ACTIVE_SETUPS", "TOP_ACTIVE", "TOP_ACTIVE_SCALPS"}:
            try:
                dynamic = await load_universe(style=style, sector=None, limit=limit)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Dynamic universe expansion failed for %s: %s", token, exc)
                dynamic = []
            for dyn_symbol in dynamic:
                dyn_token = (dyn_symbol or "").strip().upper()
                if dyn_token and dyn_token not in expanded:
                    expanded.append(dyn_token)
            continue
        if token not in expanded:
            expanded.append(token)
    return expanded


class ChartParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    interval: str
    direction: Literal["long", "short"] | None = None
    entry: float | str | None = None
    stop: float | str | None = None
    tp: str | None = None
    title: str | None = None
    plan_id: str | None = None
    plan_version: str | None = None
    strategy: str | None = None
    range: str | None = None
    focus: str | None = None
    center_time: str | None = None
    scale_plan: str | None = None
    notes: str | None = None
    live: str | None = None
    last_update: str | None = None
    data_source: str | None = None
    data_mode: str | None = None
    data_age_ms: str | int | None = None
    runner: str | None = None
    tp_meta: str | None = None
    view: str | None = None
    levels: str | None = None
    ema: str | None = None


class ChartLinks(BaseModel):
    model_config = ConfigDict(extra="forbid")

    interactive: str


async def _fetch_option_chain_with_aliases(symbol: str, as_of_hint: Optional[datetime]) -> pd.DataFrame:
    aliases = _data_symbol_candidates(symbol)
    for candidate in aliases:
        try:
            frame = await fetch_polygon_option_chain_asof(candidate, as_of_hint)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("polygon option chain alias %s failed for %s: %s", candidate, symbol, exc)
            continue
        if frame is not None and not frame.empty:
            if candidate != symbol:
                frame = frame.copy()
                frame["underlying_symbol"] = candidate
            return frame
    return pd.DataFrame()


class AllowedHostsMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, allowed: Sequence[str]):
        super().__init__(app)
        self.allowed_hosts = _allowed_host_set(allowed or [])

    async def dispatch(self, request: Request, call_next):
        request.state.allowed_hosts = self.allowed_hosts
        response = await call_next(request)
        return response


app = FastAPI(
    title="Trading Coach GPT Backend",
    description="Backend utilities for a custom GPT that offers trading guidance.",
    version="0.2.0",
)

app.add_middleware(AllowedHostsMiddleware, allowed=get_settings().ft_allowed_hosts)
app.add_middleware(SessionMiddleware)
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

    settings = get_settings()
    polygon_key = (settings.polygon_api_key or "").strip() if hasattr(settings, "polygon_api_key") else ""
    if polygon_key:
        try:
            streamer = PolygonRealtimeBarStreamer(
                polygon_key,
                symbols=list(INDEX_BASE_TICKERS),
                on_event=_ingest_stream_event,
            )
            streamer.start()
            global _REALTIME_BAR_STREAM
            _REALTIME_BAR_STREAM = streamer
            logger.info(
                "Polygon realtime bar streamer active for %s",
                ",".join(INDEX_BASE_TICKERS),
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Unable to start realtime bar streamer: %s", exc)


@app.on_event("shutdown")
async def _shutdown_tasks() -> None:
    global _REALTIME_BAR_STREAM
    if _REALTIME_BAR_STREAM is not None:
        try:
            await _REALTIME_BAR_STREAM.stop()
        except Exception:  # pragma: no cover - defensive
            logger.exception("Error stopping realtime bar streamer")
        finally:
            _REALTIME_BAR_STREAM = None


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
    tickers: List[str] | None = Field(
        default=None,
        description="Explicit ticker symbols to analyse. When omitted, the server derives a universe from Polygon.",
    )
    style: str | None = Field(
        default=None,
        description="Optional style filter: 'scalp', 'intraday', 'swing', or 'leap'.",
    )
    sector: str | None = Field(
        default=None,
        description="Optional sector focus (e.g. 'technology', 'healthcare').",
    )
    include: List[str] | None = Field(
        default=None,
        description="Symbols that must be included in the universe.",
    )
    exclude: List[str] | None = Field(
        default=None,
        description="Symbols to remove from the derived universe.",
    )
    limit: int | None = Field(
        default=None,
        ge=10,
        le=250,
        description="Maximum number of symbols to scan (default varies by style).",
    )

    model_config = ConfigDict(extra="allow")


class ContractsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

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
    selection_mode: str | None = None
    plan_anchor: Dict[str, Any] | None = None


class PlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    style: str | None = None
    plan_id: str | None = None
    simulate_open: bool = False


class RejectedContract(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    reason: str


class PlanResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan_id: str | None = None
    version: int | None = None
    trade_detail: str | None = None
    warnings: List[str] = Field(default_factory=list)
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
    confluence_tags: List[str] = Field(default_factory=list)
    confluence: List[str] = Field(default_factory=list)
    key_levels_used: Dict[str, Any] | None = None
    risk_block: Dict[str, Any] | None = None
    execution_rules: Dict[str, Any] | None = None
    notes: str | None = None
    relevant_levels: Dict[str, Any] | None = None
    expected_move_basis: str | None = None
    sentiment: Dict[str, Any] | None = None
    events: Dict[str, Any] | None = None
    earnings: Dict[str, Any] | None = None
    charts_params: Dict[str, Any] | None = None
    chart_url: str | None = None
    chart_timeframe: str | None = None
    chart_guidance: str | None = None
    strategy_id: str | None = None
    description: str | None = None
    score: float | None = None
    plan: Dict[str, Any] | None = None
    structured_plan: Dict[str, Any] | None = None
    target_profile: Dict[str, Any] | None = None
    charts: Dict[str, Any] | None = None
    key_levels: Dict[str, Any] | None = None
    market_snapshot: Dict[str, Any] | None = None
    features: Dict[str, Any] | None = None
    options: Dict[str, Any] | None = None
    options_contracts: List[Dict[str, Any]] = Field(default_factory=list)
    options_note: str | None = None
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
    session_state: Dict[str, Any] | None = None
    confidence_visual: str | None = None
    plan_layers: Dict[str, Any] = Field(default_factory=dict)
    tp_reasons: List[Dict[str, Any]] = Field(default_factory=list)
    meta: Dict[str, Any] | None = None
    source_paths: Dict[str, str] = Field(default_factory=dict)
    accuracy_levels: List[str] = Field(default_factory=list)
    rejected_contracts: List[RejectedContract] = Field(default_factory=list)
    layers_fetched: bool | None = None
    phase: str | None = None


class AssistantExecRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    style: str | None = None
    plan_id: str | None = None


class AssistantExecResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan: Dict[str, Any]
    chart: Dict[str, Any]
    options: Dict[str, Any] | None = None
    context: Dict[str, Any]
    meta: Dict[str, Any]


class SymbolDiagnosticsResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    interval: str
    key_levels: Dict[str, Any]
    snapshot: Dict[str, Any]
    indicators: Dict[str, Any]
    session: Dict[str, Any]


class IdeaStoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

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
    model_config = ConfigDict(extra="forbid")

    plan_id: str
    trade_detail: str


class StreamPushRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    event: Dict[str, Any]


class MultiContextRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    intervals: List[str] = Field(
        ..., validation_alias=AliasChoices("intervals", "frames")
    )
    lookback: int | None = None
    include_series: bool = False


class MultiContextResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str
    snapshots: List[Dict[str, Any]]
    volatility_regime: Dict[str, Any]
    sentiment: Dict[str, Any] | None = None
    events: Dict[str, Any] | None = None
    earnings: Dict[str, Any] | None = None


def _normalize_host_token(value: str | None) -> str:
    if not value:
        return ""
    parsed = urlsplit(value)
    token = parsed.netloc or parsed.path or ""
    return token.lower()


def _allowed_host_set(tokens: Iterable[str]) -> Set[str]:
    hosts: Set[str] = set()
    for token in tokens:
        normalized = _normalize_host_token(token)
        if normalized:
            hosts.add(normalized)
    return hosts


def _ensure_allowed_host(url: str, request: Request) -> None:
    allowed_hosts: Set[str] | None = getattr(request.state, "allowed_hosts", None)
    settings = get_settings()
    if not allowed_hosts:
        allowed_hosts = _allowed_host_set(getattr(settings, "ft_allowed_hosts", []))
    for candidate in (getattr(settings, "public_base_url", None), getattr(settings, "chart_base_url", None)):
        normalized = _normalize_host_token(candidate)
        if normalized:
            allowed_hosts = (allowed_hosts or set())
            allowed_hosts.add(normalized)
    if not allowed_hosts:
        return
    host = _normalize_host_token(url)
    if host not in allowed_hosts:
        raise HTTPException(status_code=400, detail={"error": "HOST_NOT_ALLOWED", "host": host})


    summary: Dict[str, Any] | None = None
    decimals: int | None = None
    data_quality: Dict[str, Any] | None = None
    contexts: List[Dict[str, Any]] | None = None  # backward compatibility


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


def _ranking_style(style: str | None) -> RankingStyle:
    token = (_normalize_style(style) or "intraday").lower()
    if token == "leap":
        token = "leaps"
    if token not in {"scalp", "intraday", "swing", "leaps"}:
        token = "intraday"
    return cast(RankingStyle, token)


def _clamp01(value: float) -> float:
    if math.isnan(value):
        return 0.0
    return max(0.0, min(1.0, value))


def _metrics_to_features(metrics: Metrics, style: RankingStyle) -> RankingFeatures:
    rr1_norm = min(metrics.rr_t1, 2.5) / 2.5 if metrics.rr_t1 > 0 else 0.0
    rr2_norm = min(metrics.rr_t2, 3.5) / 3.5 if metrics.rr_t2 > 0 else 0.0
    rr_multi_norm = min(metrics.rr_multi, 3.0) / 3.0 if metrics.rr_multi > 0 else 0.0
    penalties = metrics.penalties
    return RankingFeatures(
        symbol=metrics.symbol.upper(),
        style=style,
        sector=metrics.sector,
        entry_quality=_clamp01(metrics.entry_quality),
        rr1=_clamp01(rr1_norm),
        rr2=_clamp01(rr2_norm),
        liquidity=_clamp01(metrics.liquidity),
        confluence=_clamp01(metrics.confluence_micro),
        momentum=_clamp01(metrics.momentum_micro),
        vol_constraint=_clamp01(metrics.vol_ok),
        htf_structure=_clamp01(metrics.struct_d1),
        confluence_htf=_clamp01(metrics.conf_htf),
        vol_regime=_clamp01(metrics.vol_regime),
        momentum_htf=_clamp01(metrics.mom_htf),
        context=_clamp01(metrics.context_score),
        actionability=_clamp01(metrics.actionability),
        weekly_structure=_clamp01(metrics.struct_w1),
        daily_confluence=_clamp01(metrics.conf_d1),
        option_efficiency=_clamp01(metrics.opt_eff),
        rr_multi=_clamp01(rr_multi_norm),
        macrofit=_clamp01(metrics.macro_fit),
        confidence=_clamp01(metrics.confidence),
        pen_event=_clamp01(penalties.pen_event),
        pen_dq=_clamp01(penalties.pen_dq),
        pen_spread=_clamp01(penalties.pen_spread),
        pen_chop=_clamp01(penalties.pen_chop),
        pen_cluster=_clamp01(penalties.pen_cluster),
    )


def _fallback_scan_candidates(
    symbols: Sequence[str],
    market_data: Dict[str, pd.DataFrame],
    *,
    limit: int,
) -> List[ScanCandidate]:
    ranked: List[Tuple[float, ScanCandidate]] = []
    fallback_counter = 0
    for idx, symbol in enumerate(symbols):
        frame = market_data.get(symbol)
        if frame is None or frame.empty:
            fallback_counter += 1
            score = max(0.2, 0.45 - (idx * 0.01))
            confidence = 0.32 + (fallback_counter * 0.005)
            reasons = [
                "Fallback candidate — market data unavailable",
                "Ranking via universe order",
            ]
        else:
            rvol = _relative_volume(frame)
            liq = _liquidity_score(frame)
            liquidity_component = min(liq / 1_000_000_000.0, 0.3)
            score = max(rvol, 0.05) + liquidity_component
            confidence = max(min(0.35 + 0.2 * min(rvol, 1.5), 0.65), 0.3)
            reasons = [
                f"Fallback candidate — RVOL {rvol:.2f}",
                "Liquidity baseline applied",
            ]
        candidate = ScanCandidate(
            symbol=symbol.upper(),
            rank=0,
            score=float(round(score, 4)),
            reasons=reasons,
            confidence=float(round(confidence, 4)),
            entry=None,
            stop=None,
            tps=[],
            rr_t1=None,
            plan_id=None,
            chart_url=None,
            entry_distance_pct=None,
            entry_distance_atr=None,
            bars_to_trigger=None,
            actionable_soon=None,
            source_paths={},
        )
        ranked.append((score, candidate))
    ranked.sort(
        key=lambda item: (
            -item[1].confidence if item[1].confidence is not None else -item[0],
            -item[0],
            item[1].symbol,
        )
    )
    fallback_limit = max(1, min(limit, len(ranked)))
    ordered: List[ScanCandidate] = []
    for position, (_, candidate) in enumerate(ranked[:fallback_limit], start=1):
        ordered.append(candidate.model_copy(update={"rank": position}))
    return ordered


def _fallback_scan_page(
    request: ScanRequest,
    context: ScanContext,
    *,
    symbols: Sequence[str],
    market_data: Dict[str, pd.DataFrame],
    dq: Dict[str, Any],
    banner: str | None,
    limit: int,
    mode: str | None = None,
    label: str | None = None,
    session: Dict[str, Any],
) -> ScanPage | None:
    # Primary frozen list for requested horizon
    is_live = context.label == "live" and context.is_open
    max_items = 20 if is_live else 10
    primary_limit = max(1, min(limit, max_items))
    candidates = _fallback_scan_candidates(symbols, market_data, limit=primary_limit)
    if not candidates:
        return None
    # Prepare alternate horizon preview (intraday vs swing)
    alt_key = "swing" if request.style != "swing" else "intraday"
    alt_list = _fallback_scan_candidates(symbols, market_data, limit=10)
    alt_compact = [
        {"symbol": c.symbol, "score": c.score, "confidence": c.confidence}
        for c in alt_list
    ]
    meta = {
        "style": request.style,
        "limit": request.limit,
        "universe_size": len(symbols),
        "mode": mode or ("live_default" if is_live else "frozen_preview"),
        "label": label or ("Live liquidity leaders" if is_live else "Frozen leaders (as of RTH close)"),
        "alt_candidates": {alt_key: alt_compact},
    }
    if context.simulate_open:
        meta["simulated_open"] = True
    banner_value = banner if banner is not None else (None if is_live else "Market closed — frozen data")
    return ScanPage(
        as_of=context.as_of_iso,
        planning_context=context.label,
        banner=banner_value,
        meta=meta,
        candidates=candidates,
        data_quality=dq,
        session=session,
        phase="scan",
        count_candidates=len(candidates),
        next_cursor=None,
    )


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


def _attach_market_chart_params(
    payload: Dict[str, Any],
    market_meta: Optional[Dict[str, Any]],
    data_meta: Optional[Dict[str, Any]],
) -> None:
    if not isinstance(payload, dict):
        return
    if isinstance(market_meta, dict):
        status_token = str(market_meta.get("status") or "").strip().lower()
        if status_token:
            payload.setdefault("market_status", status_token)
        session_state = market_meta.get("session_state")
        if isinstance(session_state, dict):
            session_status = str(session_state.get("status") or "").strip().lower()
            if session_status:
                payload.setdefault("session_status", session_status)
            phase_token = str(session_state.get("phase") or "").strip().lower()
            if not phase_token:
                if session_status in {"open", "regular", "rth"}:
                    phase_token = "regular"
                elif session_status in {"pre", "pre-market", "premarket"}:
                    phase_token = "premarket"
                elif session_status in {"post", "afterhours", "after-hours"}:
                    phase_token = "afterhours"
                elif session_status in {"closed"}:
                    phase_token = "closed"
            if phase_token:
                payload.setdefault("session_phase", phase_token)
            banner = session_state.get("banner")
            if isinstance(banner, str) and banner.strip():
                payload.setdefault("session_banner", banner.strip())
        session_token = str(market_meta.get("session") or "").strip().lower()
        if session_token and "session_phase" not in payload:
            if session_token in {"regular", "rth"}:
                payload.setdefault("session_phase", "regular")
    if isinstance(data_meta, dict):
        error = data_meta.get("error")
        if error and "data_error" not in payload:
            payload.setdefault("data_error", str(error))
        mode = data_meta.get("mode")
        if mode and "data_mode" not in payload:
            payload.setdefault("data_mode", str(mode))
        freshness = data_meta.get("data_freshness_ms")
        if freshness is not None and "data_age_ms" not in payload:
            try:
                payload.setdefault("data_age_ms", str(int(freshness)))
            except Exception:
                payload.setdefault("data_age_ms", str(freshness))


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
    root = f"{scheme}://{host}" if host else _resolved_base_url(request)
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
    is_plan_live = str(first.get("planning_context") or "").strip().lower() == "live"
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


def _json_compatible_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _json_compatible_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_compatible_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_compatible_value(v) for v in value.tolist()]
    if isinstance(value, (pd.Series, pd.Index)):
        return [_json_compatible_value(v) for v in value.tolist()]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, pd.Timestamp):
        ts = value
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts.isoformat()
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except Exception:
            return value.hex()
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def _normalize_snapshot(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    return cast(Dict[str, Any], _json_compatible_value(snapshot))


async def _store_idea_snapshot(plan_id: str, snapshot: Dict[str, Any]) -> None:
    normalized_snapshot = _normalize_snapshot(snapshot)
    await _cache_snapshot(plan_id, normalized_snapshot)
    persisted = False
    version = _extract_snapshot_version(normalized_snapshot)
    if _IDEA_PERSISTENCE_ENABLED:
        if version is not None:
            persisted = await db_store_idea_snapshot(plan_id, version, normalized_snapshot)
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
            "snapshot_keys": list(normalized_snapshot.keys()),
            "persisted": persisted,
        },
    )
    plan_block = normalized_snapshot.get("plan") or {}
    symbol_for_event = (plan_block.get("symbol") or "").upper()
    plan_full_event = None
    try:
        plan_state_event = await _LIVE_PLAN_ENGINE.register_snapshot(normalized_snapshot)
    except Exception:
        logger.exception("live plan engine snapshot registration failed", extra={"plan_id": plan_id})
        plan_state_event = None
    if symbol_for_event:
        await _ensure_symbol_stream(symbol_for_event)
        plan_full_event = {
            "t": "plan_full",
            "plan_id": plan_block.get("plan_id"),
            "payload": normalized_snapshot,
            "reason": (plan_state_event or {}).get("reason") if isinstance(plan_state_event, dict) else "snapshot_stored",
        }
        await _publish_stream_event(symbol_for_event, plan_full_event)
    if plan_state_event and symbol_for_event:
        await _publish_stream_event(symbol_for_event, plan_state_event)


async def _publish_stream_event(symbol: str, event: Dict[str, Any]) -> None:
    async with _STREAM_LOCK:
        symbol_queues = list(_STREAM_SUBSCRIBERS.get(symbol, []))
        plan_id = _extract_event_plan_id(event)
        plan_queues = list(_PLAN_STREAM_SUBSCRIBERS.get(plan_id, [])) if plan_id else []
    symbol_payload = json.dumps({"symbol": symbol, "event": event})
    for queue in symbol_queues:
        try:
            queue.put_nowait(symbol_payload)
        except asyncio.QueueFull:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            queue.put_nowait(symbol_payload)
    if plan_id:
        plan_payload = json.dumps({"plan_id": plan_id, "event": event})
        for queue in plan_queues:
            try:
                queue.put_nowait(plan_payload)
            except asyncio.QueueFull:
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                queue.put_nowait(plan_payload)


async def _ingest_stream_event(symbol: str, event: Dict[str, Any]) -> None:
    """Push a raw market event through the live engine before fan-out."""

    derived_events = await _LIVE_PLAN_ENGINE.handle_market_event(symbol, event)
    await _publish_stream_event(symbol, event)
    for derived in derived_events:
        await _publish_stream_event(symbol, derived)


def _extract_event_plan_id(event: Dict[str, Any]) -> Optional[str]:
    plan_id = event.get("plan_id")
    if plan_id:
        return str(plan_id)
    payload = event.get("payload")
    if isinstance(payload, dict):
        plan_block = payload.get("plan")
        if isinstance(plan_block, dict):
            candidate = plan_block.get("plan_id")
            if candidate:
                return str(candidate)
        candidate = payload.get("plan_id")
        if candidate:
            return str(candidate)
    return None


async def _ensure_stream_heartbeat(symbol: str) -> None:
    symbol_key = (symbol or "").upper()
    if not symbol_key:
        return

    async with _STREAM_LOCK:
        if symbol_key in _STREAM_HEARTBEAT_TASKS:
            return

        async def _heartbeat() -> None:
            try:
                while True:
                    await asyncio.sleep(_STREAM_HEARTBEAT_INTERVAL)
                    async with _STREAM_LOCK:
                        active = bool(_STREAM_SUBSCRIBERS.get(symbol_key))
                    if not active:
                        break
                    await _publish_stream_event(
                        symbol_key,
                        {
                            "t": "heartbeat",
                            "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                        },
                    )
            except asyncio.CancelledError:
                pass
            finally:
                async with _STREAM_LOCK:
                    _STREAM_HEARTBEAT_TASKS.pop(symbol_key, None)

        _STREAM_HEARTBEAT_TASKS[symbol_key] = asyncio.create_task(_heartbeat())


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
    response = await gpt_plan(plan_request, request, Response(), user)

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
    cloned["plan"] = plan_block
    cloned["trade_detail"] = redirect_url

    await _store_idea_snapshot(plan_id, cloned)
    return cloned


async def _stream_generator(symbol: str) -> Any:
    queue: asyncio.Queue[str] = asyncio.Queue(maxsize=100)
    async with _STREAM_LOCK:
        _STREAM_SUBSCRIBERS.setdefault(symbol, []).append(queue)
    await _ensure_symbol_stream(symbol)
    await _ensure_stream_heartbeat(symbol)
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
    if tf.endswith("m") or tf.isdigit():
        return age > pd.Timedelta(minutes=15)
    if tf.endswith("h"):
        return age > pd.Timedelta(minutes=30)
    if tf in {"d", "1d", "day"}:
        return age > pd.Timedelta(days=3)
    return age > pd.Timedelta(days=7)


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
            fresh_polygon = poly.copy()
            fresh_polygon.attrs["source"] = "polygon"
            break
        stale_frame = poly.copy()
        stale_frame.attrs["source"] = "polygon_stale"
        stale_polygon = stale_frame if stale_polygon is None else stale_polygon

    if fresh_polygon is not None:
        return fresh_polygon

    # Attempt Yahoo fallback before returning stale Polygon data
    for candidate in candidates:
        try:
            yahoo = await fetch_yahoo_ohlcv(candidate, timeframe)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("Yahoo fetch failed for %s: %s", candidate, exc)
            continue
        if yahoo is None or yahoo.empty:
            continue
        if _is_stale_frame(yahoo, timeframe):
            continue
        yahoo_copy = yahoo.copy()
        yahoo_copy.attrs["source"] = "yahoo"
        logger.info("using yahoo fallback for %s", symbol)
        return yahoo_copy

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
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str]]:
    """Fetch OHLCV for a list of tickers with caching and fallbacks."""
    tasks = [_load_remote_ohlcv(ticker, timeframe) for ticker in tickers]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    market_data: Dict[str, pd.DataFrame] = {}
    source_map: Dict[str, str] = {}

    now_monotonic = time.monotonic()
    as_of_token = None
    if as_of is not None:
        as_of_value = as_of
        if as_of_value.tzinfo is None:
            as_of_value = as_of_value.replace(tzinfo=timezone.utc)
        as_of_token = as_of_value.astimezone(timezone.utc).isoformat()
    cache_suffix = as_of_token or "live"

    for ticker, result in zip(tickers, results):
        frame: pd.DataFrame | None = None
        source_label = "unknown"
        cache_key = (ticker.upper(), timeframe, cache_suffix)
        if isinstance(result, Exception):
            logger.warning("Data fetch raised for %s: %s", ticker, result)
        elif isinstance(result, pd.DataFrame) and not result.empty:
            frame = result

        if frame is None or frame.empty:
            cached = _MARKET_DATA_CACHE.get(cache_key)
            if cached:
                age = now_monotonic - float(cached.get("ts", 0.0))
                if age < _MARKET_DATA_CACHE_TTL:
                    cached_frame = cached["frame"].copy(deep=True)
                    cached_source = cached.get("source") or "cache"
                    cached_frame.attrs["source"] = f"{cached_source}_cached"
                    frame = cached_frame
                    source_label = f"{cached_source}_cached"
                    logger.warning("Using cached market data for %s timeframe=%s", ticker, timeframe)
            if frame is None or frame.empty:
                source_map[ticker] = "missing"
                logger.warning("No market data available for %s", ticker)
                continue

        if frame is not None:
            source_label = frame.attrs.get("source", source_label or "live")
            _MARKET_DATA_CACHE[cache_key] = {
                "frame": frame.copy(deep=True),
                "source": source_label,
                "ts": now_monotonic,
                "as_of": as_of_token,
            }

        if as_of is not None:
            cutoff = pd.Timestamp(as_of).tz_convert("UTC")
            frame = frame.loc[frame.index <= cutoff]
            if frame.empty:
                source_map[ticker] = f"{source_label}_empty"
                logger.warning("No market data available for %s up to %s", ticker, cutoff)
                continue

        source_map[ticker] = source_label
        market_data[ticker] = frame

    return market_data, source_map


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


def _unique_tags(values: Iterable[Any]) -> List[str]:
    seen: set[str] = set()
    ordered: List[str] = []
    for value in values or []:
        token = str(value or "").strip()
        if not token:
            continue
        key = token.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(token)
    return ordered


def _classify_level(label: str) -> str:
    token = (label or "").strip().lower()
    if not token:
        return "structural"
    session_keywords = (
        "opening_range",
        "session_",
        "orh",
        "orl",
        "vwap",
        "avwap",
        "intraday",
        "gap",
        "lod",
        "hod",
    )
    structural_keywords = (
        "prev_",
        "vah",
        "val",
        "poc",
        "weekly",
        "monthly",
        "year",
        "supply",
        "demand",
        "swing",
    )
    if any(token.startswith(prefix) for prefix in session_keywords) or token in {"orh", "orl", "vwap"}:
        return "session"
    if any(keyword in token for keyword in structural_keywords):
        return "structural"
    return "structural"


def _ensure_plan_layers_cover_used_levels(
    plan_layers: Dict[str, Any],
    key_levels_used: Dict[str, List[Dict[str, Any]]],
    *,
    precision: int,
) -> None:
    if not isinstance(plan_layers, dict):
        return
    levels: List[Dict[str, Any]] = plan_layers.setdefault("levels", [])
    def _normalized_price(value: Any) -> float | None:
        try:
            price = float(value)
        except (TypeError, ValueError):
            return None
        return round(price, precision)

    existing_keys: set[tuple[str | None, float | None]] = set()
    for item in levels:
        if not isinstance(item, dict):
            continue
        label = item.get("label")
        price_val = _normalized_price(item.get("price"))
        existing_keys.add((label, price_val))

    for bucket in key_levels_used.values():
        for entry in bucket:
            label = entry.get("label")
            price_val = _normalized_price(entry.get("price"))
            if price_val is None:
                continue
            key = (label, price_val)
            if key in existing_keys:
                continue
            levels.append({"price": price_val, "label": label, "kind": "level"})
            existing_keys.add(key)

    meta = plan_layers.setdefault("meta", {})
    level_groups = meta.setdefault("level_groups", {"primary": [], "supplemental": []})
    primary = level_groups.setdefault("primary", [])
    supplemental = level_groups.setdefault("supplemental", [])

    def _contains(container: List[Dict[str, Any]], candidate: Dict[str, Any]) -> bool:
        for item in container:
            if not isinstance(item, dict):
                continue
            if item.get("label") == candidate.get("label") and item.get("price") == candidate.get("price"):
                return True
        return False

    for bucket in key_levels_used.values():
        for entry in bucket:
            price_val = _normalized_price(entry.get("price"))
            if price_val is None:
                continue
            candidate = {"price": price_val, "label": entry.get("label"), "kind": "level"}
            if _contains(primary, candidate) or _contains(supplemental, candidate):
                continue
            supplemental.append(candidate)


def _higher_timeframe(interval: str | None) -> str | None:
    token = (interval or "5m").lower()
    mapping = {
        "1m": "5m",
        "3m": "15m",
        "5m": "15m",
        "10m": "30m",
        "15m": "1h",
        "30m": "1h",
        "45m": "1h",
        "60m": "4h",
        "1h": "4h",
        "2h": "4h",
        "4h": "1d",
        "1d": "1w",
        "d": "1w",
    }
    return mapping.get(token, "1d")


def _compute_rsi_from_bars(bars: Sequence[Mapping[str, Any]], period: int = 14) -> float | None:
    closes: List[float] = []
    for bar in bars or []:
        close = bar.get("close")
        if isinstance(close, (int, float)):
            closes.append(float(close))
    if len(closes) <= period:
        return None
    series = pd.Series(closes[-(period + 1) :])
    delta = series.diff().dropna()
    if delta.empty:
        return None
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
    avg_loss = losses.ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return max(0.0, min(100.0, float(rsi)))


def _volume_profile_tags(entry: Any, overlays: Mapping[str, Any] | None, atr_value: Any, interval: str) -> List[str]:
    if overlays is None:
        return []
    vp = overlays.get("volume_profile") if isinstance(overlays, Mapping) else None
    if not isinstance(vp, Mapping):
        return []
    try:
        entry_val = float(entry) if entry is not None else None
    except (TypeError, ValueError):
        entry_val = None
    if entry_val is None:
        return []
    try:
        atr_val = float(atr_value) if atr_value is not None else None
    except (TypeError, ValueError):
        atr_val = None
    threshold = atr_val * 0.5 if atr_val and atr_val > 0 else max(abs(entry_val) * 0.0015, 0.1)
    tags: List[str] = []
    for label_key, vp_key in [("VAH", "vah"), ("VAL", "val"), ("POC", "poc")]:
        level_val = vp.get(vp_key)
        if not isinstance(level_val, (int, float)):
            continue
        distance = abs(float(level_val) - entry_val)
        if distance <= threshold:
            tags.append(f"Near {label_key} ({interval})")
    return tags


def _confluence_tags_for_snapshot(
    snapshot: Mapping[str, Any],
    interval: str,
    *,
    direction: str | None,
    entry: Any,
    overlays: Mapping[str, Any] | None,
    context: Mapping[str, Any] | None,
) -> List[str]:
    tags: List[str] = []
    interval_label = interval
    direction_token = (direction or "long").lower()
    indicators = snapshot.get("indicators") or {}
    price_block = snapshot.get("price") or {}
    try:
        price_val = float(price_block.get("close"))
    except (TypeError, ValueError):
        price_val = None
    trend = (snapshot.get("trend") or {}).get("ema_stack")
    if direction_token == "long" and trend == "bullish":
        tags.append(f"EMA alignment ({interval_label})")
    elif direction_token == "short" and trend == "bearish":
        tags.append(f"EMA alignment ({interval_label})")
    vwap_val = indicators.get("vwap")
    if isinstance(vwap_val, (int, float)) and isinstance(price_val, float):
        if direction_token == "long" and price_val >= float(vwap_val):
            tags.append(f"Above VWAP ({interval_label})")
        elif direction_token == "short" and price_val <= float(vwap_val):
            tags.append(f"Below VWAP ({interval_label})")
    if (snapshot.get("volatility") or {}).get("in_squeeze"):
        tags.append(f"Squeeze compression ({interval_label})")
    adx_val = indicators.get("adx14")
    if isinstance(adx_val, (int, float)) and float(adx_val) >= 25:
        tags.append(f"ADX trending ({interval_label})")
    rsi_val = None
    if context and isinstance(context.get("bars"), list):
        rsi_val = _compute_rsi_from_bars(context["bars"])
    if rsi_val is not None:
        if direction_token == "long" and rsi_val >= 55:
            tags.append(f"RSI bullish ({interval_label})")
        elif direction_token == "short" and rsi_val <= 45:
            tags.append(f"RSI bearish ({interval_label})")
    tags.extend(_volume_profile_tags(entry, overlays, indicators.get("atr14"), interval_label))
    return tags


async def _compute_multi_timeframe_confluence(
    symbol: str,
    base_interval: str,
    *,
    snapshot: Mapping[str, Any],
    direction: str | None,
    entry: Any,
    overlays: Mapping[str, Any] | None,
) -> List[str]:
    base_interval_norm = base_interval or "5m"
    tags: List[str] = []
    tags.extend(
        _confluence_tags_for_snapshot(
            snapshot,
            base_interval_norm,
            direction=direction,
            entry=entry,
            overlays=overlays,
            context=None,
        )
    )
    higher = _higher_timeframe(base_interval_norm)
    if higher:
        try:
            context = await _build_interval_context(symbol, higher, 300)
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug(
                "mtf_confluence_fetch_failed",
                extra={"symbol": symbol, "interval": higher, "detail": str(exc)},
            )
        else:
            tags.extend(
                _confluence_tags_for_snapshot(
                    context.get("snapshot") or {},
                    higher,
                    direction=direction,
                    entry=entry,
                    overlays=None,
                    context=context,
                )
            )
    return _unique_tags(tags)


def _extract_runner_multiple(runner: Mapping[str, Any] | None) -> float | None:
    if not isinstance(runner, Mapping):
        return None
    for key in ("atr_multiple", "multiple", "trail_multiple"):
        value = runner.get(key)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                continue
    trail = runner.get("trail")
    if isinstance(trail, str):
        match = re.search(r"x\s*(\d*\.?\d+)", trail.lower())
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
    return None


def _build_risk_block(
    *,
    entry: Any,
    stop: Any,
    targets: Sequence[Any] | None,
    atr: Any,
    stop_multiple: Any,
    expected_move: Any,
    runner: Mapping[str, Any] | None,
) -> Dict[str, Any] | None:
    try:
        entry_val = float(entry)
        stop_val = float(stop)
    except (TypeError, ValueError):
        return None
    risk_points = abs(entry_val - stop_val)
    if risk_points <= 0:
        return None
    risk_percent = None
    if entry_val != 0:
        risk_percent = abs(risk_points / abs(entry_val)) * 100.0
    rr_map: Dict[str, float] = {}
    clean_targets: List[float] = []
    for idx, target in enumerate(targets or [], start=1):
        try:
            target_val = float(target)
        except (TypeError, ValueError):
            continue
        clean_targets.append(target_val)
        reward = abs(target_val - entry_val)
        if risk_points > 0:
            rr_map[f"tp{idx}"] = round(reward / risk_points, 2)
    atr_val = None
    try:
        atr_val = float(atr) if atr is not None else None
    except (TypeError, ValueError):
        atr_val = None
    stop_multiple_val = None
    try:
        stop_multiple_val = float(stop_multiple) if stop_multiple is not None else None
    except (TypeError, ValueError):
        stop_multiple_val = None
    if stop_multiple_val is None and atr_val and atr_val > 0:
        stop_multiple_val = risk_points / atr_val
    expected_move_val = None
    try:
        expected_move_val = float(expected_move) if expected_move is not None else None
    except (TypeError, ValueError):
        expected_move_val = None
    expected_move_map: Dict[str, float] = {}
    if expected_move_val and expected_move_val > 0:
        for idx, target_val in enumerate(clean_targets, start=1):
            reward = abs(target_val - entry_val)
            expected_move_map[f"tp{idx}"] = round(reward / expected_move_val, 2)
    runner_multiple = _extract_runner_multiple(runner)
    block: Dict[str, Any] = {
        "risk_points": round(risk_points, 4),
        "reward_to_target": rr_map or None,
    }
    if risk_percent is not None:
        block["risk_percent"] = round(risk_percent, 2)
    if atr_val is not None:
        block["atr_value"] = round(atr_val, 4)
    if stop_multiple_val is not None:
        block["atr_stop_multiple"] = round(stop_multiple_val, 2)
    if expected_move_val is not None:
        block["expected_move"] = round(expected_move_val, 4)
    if expected_move_map:
        block["expected_move_fraction"] = expected_move_map
    if runner_multiple is not None:
        block["runner_trail_multiple"] = round(runner_multiple, 2)
    return {key: value for key, value in block.items() if value is not None}


def _find_level_by_role(key_levels_used: Dict[str, List[Dict[str, Any]]] | None, role: str) -> Dict[str, Any] | None:
    if not key_levels_used:
        return None
    for bucket in key_levels_used.values():
        for level in bucket:
            if str(level.get("role")) == role:
                return level
    return None


def _format_level_label(level: Dict[str, Any] | None, fallback_price: Any, precision: int) -> str:
    try:
        price_val = float(fallback_price) if fallback_price is not None else None
    except (TypeError, ValueError):
        price_val = None
    if level:
        label = level.get("label")
        try:
            level_price = float(level.get("price"))
        except (TypeError, ValueError):
            level_price = price_val
        if label and level_price is not None:
            return f"{label} @ {level_price:.{precision}f}"
        if level_price is not None:
            return f"{level_price:.{precision}f}"
    if price_val is not None:
        return f"{price_val:.{precision}f}"
    return "defined level"


def _build_execution_rules(
    *,
    entry: Any,
    stop: Any,
    targets: Sequence[Any] | None,
    direction: str | None,
    precision: int,
    key_levels_used: Dict[str, List[Dict[str, Any]]] | None,
    runner: Mapping[str, Any] | None,
) -> Dict[str, str] | None:
    if entry is None or stop is None:
        return None
    direction_token = (direction or "long").lower()
    entry_level = _find_level_by_role(key_levels_used, "entry")
    stop_level = _find_level_by_role(key_levels_used, "stop")
    tp1_level = _find_level_by_role(key_levels_used, "tp1")
    trigger_level = _format_level_label(entry_level, entry, precision)
    stop_label = _format_level_label(stop_level, stop, precision)
    tp1_display = None
    if targets:
        tp1_value = targets[0]
        tp1_display = _format_level_label(tp1_level, tp1_value, precision)
    runner_multiple = _extract_runner_multiple(runner)
    if direction_token == "short":
        trigger = f"Trigger on sustained breakdown below {trigger_level}."
        invalidation = f"Invalidate on closes above {stop_label}."
        scale = f"Cover partial size at {tp1_display}" if tp1_display else "Scale risk once 1R is achieved."
        reload = f"Reload on failed reclaim of {trigger_level} after bounce."
    else:
        trigger = f"Trigger on acceptance above {trigger_level}."
        invalidation = f"Invalidate on closes below {stop_label}."
        scale = f"Trim partial size at {tp1_display}" if tp1_display else "Scale risk once 1R is achieved."
        reload = f"Reload on defended retest of {trigger_level}."
    if runner_multiple is not None:
        scale = f"{scale} Trail runner with ATR x{runner_multiple:.2f}."
    return {
        "trigger": trigger,
        "invalidation": invalidation,
        "scale": scale,
        "reload": reload,
    }


def _tp_reason_entries(meta_list: Sequence[Mapping[str, Any]] | None) -> List[Dict[str, Any]]:
    if not meta_list:
        return []
    reasons: List[Dict[str, Any]] = []
    for idx, meta in enumerate(meta_list, start=1):
        if not isinstance(meta, Mapping):
            continue
        label = str(meta.get("label") or f"TP{idx}")
        parts: List[str] = []
        source = str(meta.get("source") or "").strip().lower()
        if source == "stats":
            parts.append("Stats-derived")
        elif source == "fallback":
            parts.append("Fallback sizing")
        elif source == "constructed":
            parts.append("Constructed spacing")
        elif source:
            parts.append(source.replace("_", " ").title())
        snap_tag = meta.get("snap_tag")
        if snap_tag:
            parts.append(f"Snapped to {snap_tag}")
        em_fraction = meta.get("em_fraction")
        if em_fraction is not None:
            try:
                pct = float(em_fraction) * 100.0
                parts.append(f"{pct:.0f}% expected move cap")
            except (TypeError, ValueError):
                pass
        mfe_quantile = meta.get("mfe_quantile")
        if mfe_quantile:
            parts.append(f"MFE {str(mfe_quantile).upper()}")
        prob_touch = meta.get("prob_touch") or meta.get("probability")
        if prob_touch is not None:
            try:
                parts.append(f"{float(prob_touch) * 100.0:.0f}% touch probability")
            except (TypeError, ValueError):
                pass
        rr_val = meta.get("rr")
        if rr_val is not None:
            try:
                parts.append(f"R:R {float(rr_val):.2f}")
            except (TypeError, ValueError):
                pass
        distance_val = meta.get("distance")
        if isinstance(distance_val, (int, float)) and math.isfinite(distance_val):
            parts.append(f"Δ {float(distance_val):.2f}")
        reason = "; ".join(parts) if parts else "Algorithmic target"
        reasons.append({
            "label": label,
            "reason": reason,
            "source": source or None,
        })
    return reasons


def _extract_options_contracts(options_payload: Mapping[str, Any] | None) -> List[Dict[str, Any]]:
    if not isinstance(options_payload, Mapping):
        return []
    best = options_payload.get("best")
    if not isinstance(best, list):
        return []
    contracts: List[Dict[str, Any]] = []
    for item in best:
        if not isinstance(item, Mapping):
            continue
        contract = {
            "symbol": item.get("symbol"),
            "label": item.get("label") or item.get("symbol"),
            "type": item.get("option_type") or item.get("type"),
            "dte": item.get("dte"),
            "strike": _safe_number(item.get("strike")),
            "price": _safe_number(item.get("price") or item.get("mark") or item.get("mid")),
            "bid": _safe_number(item.get("bid")),
            "ask": _safe_number(item.get("ask")),
            "delta": _safe_number(item.get("delta")),
            "gamma": _safe_number(item.get("gamma")),
            "theta": _safe_number(item.get("theta")),
            "vega": _safe_number(item.get("vega")),
            "open_interest": item.get("open_interest") or item.get("oi"),
            "volume": item.get("volume"),
            "liquidity_score": _safe_number(item.get("liquidity_score") or item.get("tradeability")),
            "spread_pct": _safe_number(item.get("spread_pct")),
            "iv": _safe_number(item.get("implied_volatility") or item.get("iv")),
        }
        pnl_block = item.get("pnl") if isinstance(item.get("pnl"), Mapping) else None
        if pnl_block:
            contract["pnl"] = {
                key: value
                for key, value in pnl_block.items()
                if value is not None
            }
        projection_block = item.get("pl_projection") if isinstance(item.get("pl_projection"), Mapping) else None
        if projection_block:
            contract["pl_projection"] = {
                key: value
                for key, value in projection_block.items()
                if value is not None
            }
        cost_basis = item.get("cost_basis") if isinstance(item.get("cost_basis"), Mapping) else None
        if cost_basis:
            contract["cost_basis"] = {
                key: value
                for key, value in cost_basis.items()
                if value is not None
            }
        trimmed = {key: val for key, val in contract.items() if val is not None}
        if trimmed:
            contracts.append(trimmed)
    return contracts[:3]


def _apply_option_guardrails(
    contracts: Sequence[Dict[str, Any]],
    *,
    max_spread_pct: float,
    min_open_interest: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, str]]]:
    filtered: List[Dict[str, Any]] = []
    rejected: List[Dict[str, str]] = []
    for contract in contracts:
        if not isinstance(contract, dict):
            continue
        symbol = str(contract.get("symbol") or contract.get("label") or "")
        spread = _safe_number(contract.get("spread_pct"))
        bid = _safe_number(contract.get("bid"))
        ask = _safe_number(contract.get("ask"))
        if spread is None and bid is not None and ask is not None and (bid + ask) > 0:
            mid = max((bid + ask) / 2.0, 1e-6)
            spread = abs(ask - bid) / mid * 100.0
            contract["spread_pct"] = round(spread, 2)
        oi_val = contract.get("open_interest")
        oi_numeric: float | None
        try:
            oi_numeric = float(oi_val) if oi_val is not None else None
        except (TypeError, ValueError):
            oi_numeric = None
        iv_val = _safe_number(contract.get("iv"))

        reason: str | None = None
        if spread is not None and float(spread) > max_spread_pct:
            reason = f"SPREAD_GT_{max_spread_pct:g}"
        elif oi_numeric is None:
            reason = "OI_MISSING"
        elif oi_numeric < min_open_interest:
            reason = f"OI_LT_{min_open_interest}"
        elif iv_val is None:
            reason = "IV_MISSING"

        if reason:
            rejected.append({"symbol": symbol or "UNKNOWN", "reason": reason})
            continue
        filtered.append(contract)

    return filtered[:3], rejected


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
    mapping = {"scalp": "1d", "intraday": "5d", "swing": "3M", "leap": "1Y"}
    return mapping.get(normalized, "fit")


def _range_for_style(style: str | None) -> str:
    normalized = _normalize_style(style) or ""
    mapping = {
        "scalp": "5d",
        "intraday": "15d",
        "swing": "6m",
        "leap": "1y",
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


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _fallback_style_token(style: str | None) -> str:
    token = _normalize_style(style)
    if token:
        return token
    return "intraday"


def _fallback_levels_param(levels: Dict[str, float], limit: int = 6) -> str | None:
    if not levels:
        return None
    items: List[Tuple[float, str]] = []
    for name, value in levels.items():
        try:
            price = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(price):
            continue
        label = name.replace("_", " ").title()
        items.append((abs(price), f"{price:.2f}|{label}"))
    if not items:
        return None
    items.sort(key=lambda pair: pair[0])
    tokens = [token for _, token in items[:limit]]
    return ";".join(tokens) if tokens else None


def _fallback_confidence(trend_component: float, liquidity_component: float, regime_component: float) -> float:
    composite = 0.6 * trend_component + 0.2 * liquidity_component + 0.2 * regime_component
    return _clamp(composite, 0.45, 0.92)


def _risk_reward(entry: float, stop: float, target: float, direction: str) -> float | None:
    try:
        entry_f = float(entry)
        stop_f = float(stop)
        target_f = float(target)
    except (TypeError, ValueError):
        return None
    if direction == "long":
        risk = entry_f - stop_f
        reward = target_f - entry_f
    else:
        risk = stop_f - entry_f
        reward = entry_f - target_f
    if risk <= 0 or reward <= 0:
        return None
    return reward / risk


def _confidence_visual(confidence: Optional[float]) -> Optional[str]:
    if confidence is None:
        return None
    try:
        numeric = float(confidence)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric):
        return None
    rating = _clamp(numeric, 0.0, 1.0)
    if rating >= 0.8:
        color = "🟢"
    elif rating >= 0.6:
        color = "🟡"
    elif rating >= 0.4:
        color = "🟠"
    else:
        color = "🔴"
    filled = max(0, min(5, round(rating * 5)))
    stars = "⭐" * filled + "☆" * (5 - filled)
    return f"{color} {rating:.2f} · {stars}"


def _chart_hint(strategy_id: Optional[str], style: Optional[str]) -> Tuple[str, str]:
    if strategy_id:
        hint = _STRATEGY_CHART_HINTS.get(str(strategy_id))
        if hint:
            return hint
    style_norm = _normalize_style(style)
    if style_norm and style_norm in _STYLE_CHART_HINTS:
        return _STYLE_CHART_HINTS[style_norm]
    return ("5m", "Use 5 minute closes to validate the trigger and manage stops.")


def _resolved_base_url(request: Request) -> str:
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
    return f"{scheme}://{host}".rstrip("/")


def _build_tv_chart_url(request: Request, params: Dict[str, Any]) -> str:
    base_root = _resolved_base_url(request)
    base = f"{base_root}/tv"
    query: Dict[str, str] = {}
    for key, value in params.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            value = ",".join(str(item) for item in value if item is not None)
        if key == "symbol":
            query[key] = _tv_symbol(str(value))
        else:
            query[key] = str(value)
    if not query:
        url = base
        _ensure_allowed_host(url, request)
        return url
    # Ensure symbol is present; if absent, use a safe default to avoid host fallback
    if "symbol" not in query or not query["symbol"]:
        query["symbol"] = "SPY"
    url = f"{base}?{urlencode(query, safe=',|:;+-_() ')}"
    _ensure_allowed_host(url, request)
    return url


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

PREFER_DELTA_BY_STYLE: Dict[str, float] = {
    "scalp": 0.55,
    "intraday": 0.50,
    "swing": 0.45,
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
_PLAN_STREAM_SUBSCRIBERS: Dict[str, List[asyncio.Queue]] = {}
_STREAM_LOCK = asyncio.Lock()
_STREAM_HEARTBEAT_TASKS: Dict[str, asyncio.Task] = {}
_STREAM_HEARTBEAT_INTERVAL = 5.0
_REALTIME_BAR_STREAM: Optional[PolygonRealtimeBarStreamer] = None
_IV_METRICS_CACHE_TTL = 120.0
_IV_METRICS_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}
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


async def _auto_replan(symbol: str, style: Optional[str], origin_plan_id: str, exit_reason: Optional[str]) -> None:
    if not style:
        return
    settings = get_settings()
    base_url = (settings.self_base_url or "").rstrip("/")
    if not base_url:
        logger.info(
            "auto replan skipped; SELF_API_BASE_URL not configured",
            extra={"symbol": symbol, "style": style, "plan": origin_plan_id},
        )
        return

    payload = {"symbol": symbol.upper(), "style": style}
    headers: Dict[str, str] = {"Accept": "application/json"}
    if settings.backend_api_key:
        headers["Authorization"] = f"Bearer {settings.backend_api_key}"

    url = f"{base_url}/gpt/plan"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
        data = response.json()
        new_plan_id = str(data.get("plan_id")) if isinstance(data, dict) else None
        new_plan_version = data.get("version") if isinstance(data, dict) else None
        new_plan_style = None
        if isinstance(data, dict):
            new_plan_style = data.get("style") or (data.get("plan") or {}).get("style")
        logger.info(
            "auto replan triggered",
            extra={"symbol": symbol, "style": style, "origin_plan_id": origin_plan_id, "exit_reason": exit_reason, "new_plan_id": new_plan_id},
        )
        if new_plan_id:
            note = f"Plan replanned ({style}) to {new_plan_id}." if exit_reason is None else f"Plan replanned after {exit_reason}; new plan {new_plan_id}."
            timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            changes: Dict[str, Any] = {
                "status": "auto_replanned",
                "note": note,
                "next_plan_id": new_plan_id,
                "timestamp": timestamp,
            }
            if new_plan_version is not None:
                changes["next_plan_version"] = new_plan_version
            if new_plan_style:
                changes["next_plan_style"] = new_plan_style
            await _publish_stream_event(
                symbol,
                {
                    "t": "plan_delta",
                    "plan_id": origin_plan_id,
                    "version": 1,
                    "changes": changes,
                    "reason": "auto_replan",
                },
            )
    except Exception as exc:
        logger.warning(
            "auto replan request failed",
            extra={
                "symbol": symbol,
                "style": style,
                "origin_plan_id": origin_plan_id,
                "exit_reason": exit_reason,
                "error": str(exc),
            },
        )


_LIVE_PLAN_ENGINE = LivePlanEngine()
_SYMBOL_STREAM_COORDINATOR: Optional[SymbolStreamCoordinator] = None

_LIVE_PLAN_ENGINE.set_replan_callback(_auto_replan)

_INDEX_PLANNING_MODE: Optional[IndexPlanningMode] = None


def _get_index_mode() -> Optional[IndexPlanningMode]:
    settings = get_settings()
    if not getattr(settings, "index_sniper_mode", False):
        return None
    global _INDEX_PLANNING_MODE
    if _INDEX_PLANNING_MODE is None:
        _INDEX_PLANNING_MODE = IndexPlanningMode()
    return _INDEX_PLANNING_MODE


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

    return {
        "status": "ok",
        "services": {
            "polygon": polygon_status,
            "tradier": tradier_status,
        },
    }


@gpt.get("/health/data", summary="Data cache freshness snapshot")
async def gpt_health_data(_: AuthedUser = Depends(require_api_key)) -> Dict[str, Any]:
    indicator_summary = _cache_summary(_INDICATOR_CACHE.values())
    chart_summary = _cache_summary(_CHART_URL_CACHE.values())
    futures_age = 0.0
    futures_ts = float(_FUTURES_CACHE.get("ts") or 0.0)
    if futures_ts:
        futures_age = max(time.time() - futures_ts, 0.0)
    return {
        "status": "ok",
        "indicator_cache": indicator_summary,
        "chart_cache": chart_summary,
        "market_cache_entries": len(_MARKET_DATA_CACHE),
        "futures_cache_age_seconds": round(futures_age, 3),
    }


@gpt.post("/admin/flush-caches", summary="Clear in-memory caches (admin)")
async def gpt_admin_flush_caches(_: AuthedUser = Depends(require_api_key)) -> Dict[str, Any]:
    # Flush light-weight in-memory caches (does not touch DB)
    try:
        _INDICATOR_CACHE.clear()
    except Exception:
        pass
    try:
        _CHART_URL_CACHE.clear()
    except Exception:
        pass
    try:
        _MARKET_DATA_CACHE.clear()
    except Exception:
        pass
    try:
        _FUTURES_CACHE.clear()
    except Exception:
        pass
    # Idea snapshots are used for permalinks; do not wipe without persistence
    return {
        "status": "ok",
        "flushed": [
            "indicator_cache",
            "chart_url_cache",
            "market_data_cache",
            "futures_cache",
        ],
    }

@gpt.get("/futures-snapshot", summary="Overnight/offsessions market tape (ETF proxies via Finnhub)")
async def gpt_futures_snapshot(_: AuthedUser = Depends(require_api_key)) -> Dict[str, Any]:
    now_ts = time.time()
    cached = _FUTURES_CACHE.get("data")
    ts = float(_FUTURES_CACHE.get("ts") or 0)
    if cached and (now_ts - ts < 180):
        payload = {k: (dict(v) if isinstance(v, dict) else v) for k, v in cached.items()}
        payload["stale_seconds"] = int(now_ts - ts)
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
    _FUTURES_CACHE["data"] = copy.deepcopy(payload)
    _FUTURES_CACHE["ts"] = now_ts
    return payload


@gpt.post("/scan", summary="Rank trade setups across a universe", response_model=ScanPage)
async def gpt_scan_endpoint(
    request_payload: ScanRequest,
    request: Request,
    response: Response,
    user: AuthedUser = Depends(require_api_key),
) -> ScanPage:
    started = time.perf_counter()
    fields_set = getattr(request_payload, "model_fields_set", set())
    simulate_open = _resolve_simulate_open(
        request,
        explicit_value=request_payload.simulate_open,
        explicit_field_set="simulate_open" in fields_set,
    )
    planning_mode = bool(getattr(request_payload, "planning_mode", False))
    session_payload = _session_payload_from_request(request)
    settings = get_settings()
    allow_fallback_trades = not bool(getattr(settings, "ft_no_fallback_trades", True))
    if planning_mode:
        allow_fallback_trades = True
    no_setups_banner = "NO_ELIGIBLE_SETUPS"
    session_ticker = _session_tracking_id(session_payload)
    if session_ticker:
        request.state.last_scan_session = session_ticker
        request.state.last_scan_cursor_symbols = None
    user_id = getattr(user, "user_id", "anonymous")
    style_norm = (request_payload.style or "").strip().lower()
    style_registry_token = style_norm or _SCAN_STYLE_ANY

    def _record_symbols(symbols: List[str]) -> None:
        if not session_ticker:
            return
        normalized = [sym.upper() for sym in symbols if sym]
        primary_key = (user_id, session_ticker, style_registry_token)
        _SCAN_SYMBOL_REGISTRY[primary_key] = normalized
        _SCAN_SYMBOL_REGISTRY[(user_id, session_ticker, _SCAN_STYLE_ANY)] = normalized

    def _finalize(page: ScanPage) -> ScanPage:
        response.headers["X-No-Fabrication"] = "1"
        symbols = [candidate.symbol.upper() for candidate in page.candidates]
        if session_ticker:
            request.state.last_scan_cursor_symbols = symbols
            _record_symbols(symbols)
        if page.phase == "scan" and hasattr(request.state, "scan_phase_update"):
            try:
                request.state.scan_phase_update(page.phase)
            except Exception:
                pass
        if planning_mode:
            meta_map = dict(page.meta or {})
            meta_map["planning_mode"] = True
            page.meta = meta_map
            dq_map = dict(page.data_quality or {})
            dq_map["planning_mode"] = True
            if planning_banner:
                notes_field = dq_map.setdefault("notes", [])
                if isinstance(notes_field, list) and planning_banner not in notes_field:
                    notes_field.append(planning_banner)
            page.data_quality = dq_map
            if planning_banner and not page.banner:
                page.banner = planning_banner
            if isinstance(page.session, dict):
                session_map = dict(page.session)
                session_map.setdefault("planning_mode", True)
                if planning_banner:
                    session_map.setdefault("banner", planning_banner)
                page.session = session_map
        return page

    try:
        context, context_banner, dq = _evaluate_scan_context(
            request_payload.asof_policy,
            request_payload.style,
            session_payload,
            simulate_open=simulate_open,
        )
    except TypeError:
        context, context_banner, dq = _evaluate_scan_context(
            request_payload.asof_policy,
            request_payload.style,
            session_payload,
        )

    planning_banner: str | None = None
    if planning_mode:
        banner_token = session_payload.get("banner") if isinstance(session_payload, Mapping) else None
        if banner_token:
            planning_banner = f"Planning mode — {banner_token}"
        else:
            try:
                planning_banner = f"Planning mode — market closed as of {context.as_of_iso}"
            except Exception:
                planning_banner = "Planning mode — market closed"

    def _fallback_or_empty(
        *,
        symbols: Sequence[str],
        market_data: Dict[str, pd.DataFrame],
        banner: str | None,
        limit: int,
        mode: str | None = None,
        label: str | None = None,
        empty_banner: str | None = None,
    ) -> ScanPage:
        banner_override = banner
        empty_override = empty_banner
        if planning_mode and planning_banner:
            banner_override = planning_banner if not banner_override else banner_override
            empty_override = planning_banner
        if allow_fallback_trades:
            fallback_page = _fallback_scan_page(
                request_payload,
                context,
                symbols=symbols,
                market_data=market_data,
                dq=dq,
                banner=banner_override,
                limit=limit,
                mode=mode,
                label=label,
                session=session_payload,
            )
            if fallback_page is not None:
                return fallback_page
        if not allow_fallback_trades and empty_override:
            if banner and banner != empty_override:
                notes_field = dq.setdefault("notes", [])
                if isinstance(notes_field, list):
                    notes_field.append(banner)
            resolved_banner = empty_override
        else:
            resolved_banner = banner_override or empty_override
        return _empty_scan_page(
            request_payload,
            context,
            banner=resolved_banner,
            dq=dq,
            session=session_payload,
        )
    try:
        universe_limit = max(request_payload.limit, 100)
        tickers = await expand_universe(request_payload.universe, style=request_payload.style, limit=universe_limit)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Universe expansion failed: %s", exc)
        # Present frozen leaders from default universe rather than empty
        return _finalize(
            _fallback_or_empty(
                symbols=_FROZEN_DEFAULT_UNIVERSE,
                market_data={},
                banner=None if context.is_open else (context_banner or "Universe unavailable — showing frozen leaders."),
                limit=request_payload.limit,
                empty_banner=no_setups_banner,
            )
        )

    if request_payload.filters and request_payload.filters.exclude:
        exclude = {token.strip().upper() for token in request_payload.filters.exclude if token}
        tickers = [symbol for symbol in tickers if symbol not in exclude]

    if not tickers:
        return _finalize(
            _fallback_or_empty(
                symbols=_FROZEN_DEFAULT_UNIVERSE,
                market_data={},
                banner=None if context.is_open else (context_banner or "Empty universe — showing frozen leaders."),
                limit=request_payload.limit,
                empty_banner=no_setups_banner,
            )
        )

    data_as_of = None if context.label == "live" and context.is_open else context.as_of
    banner_override: str | None = None
    try:
        market_data, data_source_map = await _collect_market_data(
            tickers,
            timeframe=context.data_timeframe,
            as_of=data_as_of,
        )
    except HTTPException as exc:
        logger.warning("Market data collection failed live: %s", exc)
        dq = dict(dq)
        dq["ok"] = False
        dq["mode"] = "degraded"
        dq["error"] = "market_data_unavailable"
        try:
            market_data, data_source_map = await _collect_market_data(
                tickers,
                timeframe=context.data_timeframe,
                as_of=context.as_of,
            )
            context = ScanContext(
                as_of=context.as_of,
                label="frozen",
                is_open=False,
                simulate_open=context.simulate_open,
                data_timeframe=context.data_timeframe,
                market_meta=context.market_meta,
                data_meta=dq,
            )
            banner_override = "Live feed unavailable — using frozen context."
        except HTTPException:
            # Absolute outage: still present leaders list instead of empty
            return _finalize(
                _fallback_or_empty(
                    symbols=tickers,
                    market_data={},
                    banner=None if context.is_open else "Market data unavailable — showing frozen leaders.",
                    limit=request_payload.limit,
                    empty_banner=no_setups_banner,
                )
            )

    dq = dict(dq)
    dq["sources"] = data_source_map
    banner = banner_override or context_banner

    index_mode = _get_index_mode()
    if index_mode:
        synthetic_count = 0
        for symbol in list(tickers):
            if not index_mode.applies(symbol):
                continue
            frame = market_data.get(symbol)
            if frame is not None and not frame.empty:
                continue
            try:
                synthetic = await index_mode.synthetic_ohlcv(symbol, context.data_timeframe)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("synthetic index build failed", extra={"symbol": symbol, "error": str(exc)})
                synthetic = None
            if synthetic is not None and not synthetic.empty:
                market_data[symbol] = synthetic
                data_source_map[symbol] = synthetic.attrs.get("source", "proxy_gamma")
                synthetic_count += 1
        if synthetic_count:
            dq.setdefault("mode", "degraded")
            banner = banner or "Index bars unavailable — translating via ETF proxy."

    missing_symbols = [sym for sym, src in data_source_map.items() if src.startswith("missing") or src.endswith("_empty")]
    if missing_symbols:
        dq["ok"] = False
        dq.setdefault("mode", "degraded")
        dq["error"] = "data_gap"
        banner = banner or "Live feed degraded — partial data missing."

    available_symbols = [sym for sym in tickers if sym in market_data and not market_data[sym].empty]
    requested_limit = max(request_payload.limit, 1)
    rank_limit = 100 if requested_limit < 100 else min(requested_limit, 100)
    page_limit = min(requested_limit, rank_limit)
    if context.is_open and available_symbols:
        now_utc = pd.Timestamp.utcnow()
        tolerance_ms = 15 * 60 * 1000  # 15 minutes
        fresh = []
        for sym in available_symbols:
            frame = market_data.get(sym)
            if frame is None or frame.empty:
                continue
            ts = frame.index[-1]
            if not isinstance(ts, pd.Timestamp):
                continue
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            age_ms = max((now_utc - ts).total_seconds() * 1000.0, 0.0)
            if age_ms <= tolerance_ms:
                fresh.append(sym)
        if fresh:
            available_symbols = fresh
        else:
            logger.warning("Live scan: all symbols stale (>%d ms); continuing with best-effort data", tolerance_ms)
    if not available_symbols:
        return _finalize(
            _fallback_or_empty(
                symbols=tickers,
                market_data=market_data,
                banner=None if context.is_open else (banner or "Market closed — showing frozen leaders."),
                limit=page_limit,
                empty_banner=no_setups_banner,
            )
        )

    effective_filters = _effective_scan_filters(request_payload.filters, context=context)
    filtered_symbols = _apply_scan_filters(available_symbols, market_data, effective_filters)
    if not filtered_symbols:
        return _finalize(
            _fallback_or_empty(
                symbols=available_symbols,
                market_data=market_data,
                banner=None if context.is_open else (banner or "Filters removed all symbols — showing frozen leaders."),
                limit=page_limit,
                empty_banner=no_setups_banner,
            )
        )
    subset = filtered_symbols[: rank_limit * 2]
    try:
        market_subset = {symbol: market_data[symbol] for symbol in subset if symbol in market_data}
        try:
            signals = await scan_market(
                subset,
                market_subset,
                as_of=context.as_of,
                simulate_open=context.simulate_open,
            )
        except TypeError:
            signals = await scan_market(
                subset,
                market_subset,
                as_of=context.as_of,
            )
        strategy_counts = Counter(signal.strategy_id for signal in signals)
        symbol_set = {signal.symbol for signal in signals}
        logger.info(
            "scan_market_signals",
            extra={
                "symbols_evaluated": len(subset),
                "signals": len(signals),
                "unique_symbols": len(symbol_set),
                "top_strategies": dict(strategy_counts.most_common(5)),
                "planning_context": context.label,
                "data_mode": dq.get("mode"),
                "sim_open": context.simulate_open,
            },
        )
        if not signals:
            logger.warning(
                "scan_market_empty",
                extra={
                    "symbols_considered": len(subset),
                    "planning_context": context.label,
                    "filters": request_payload.filters.model_dump() if request_payload.filters else None,
                },
            )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("scan_market failed: %s", exc, exc_info=True)
        dq["ok"] = False
        dq.setdefault("mode", "degraded")
        dq["error"] = "scan_failed"
        return _finalize(
            _empty_scan_page(
                request_payload,
                context,
                banner=banner or "Scan unavailable — using frozen context.",
                dq=dq,
                session=session_payload,
            )
        )

    stub_context = ScanContext(
        as_of=context.as_of,
        label=context.label,
        is_open=context.is_open,
        simulate_open=context.simulate_open,
        data_timeframe=context.data_timeframe,
        market_meta=context.market_meta,
        data_meta=dq,
    )
    preps = await _legacy_scan_stub_payload(
        signals=signals,
        market_data=market_subset,
        style_filter=request_payload.style,
        context=stub_context,
        request=request,
    )

    if not preps:
        return _finalize(
            _fallback_or_empty(
                symbols=available_symbols,
                market_data=market_subset,
                banner=None if context.is_open else (banner or "Scanner returned no signals — showing frozen leaders."),
                limit=page_limit,
                empty_banner=no_setups_banner,
            )
        )

    style_literal = _ranking_style(request_payload.style)
    features = [prep.features for prep in preps]
    ranked_rollup = rank_candidates(features, style_literal)
    diversified = diversify_ranked(ranked_rollup, limit=rank_limit)
    prep_lookup = {prep.candidate.symbol: prep for prep in preps}

    ordered_candidates: List[ScanCandidate] = []
    if diversified:
        for scored in diversified:
            prep = prep_lookup.get(scored.symbol)
            if not prep:
                continue
            candidate = prep.candidate.model_copy(
                update={
                    "score": float(scored.score),
                    "confidence": float(scored.confidence),
                }
            )
            ordered_candidates.append(candidate)
    else:
        fallback = sorted(
            preps,
            key=lambda item: (-item.candidate.score, item.candidate.symbol),
        )
        ordered_candidates = [item.candidate for item in fallback[:rank_limit]]

    if not ordered_candidates:
        return _finalize(
            _fallback_or_empty(
                symbols=available_symbols,
                market_data=market_subset,
                banner=None if context.is_open else (banner or "Ranking produced no results — showing frozen leaders."),
                limit=page_limit,
                empty_banner=no_setups_banner,
            )
        )

    if ordered_candidates:
        start_index = decode_cursor(request_payload.cursor)
        page_size = min(50, page_limit)
        cutoff = min(page_limit, len(ordered_candidates))
        start = min(start_index, cutoff)
        end = min(start + page_size, cutoff)
        next_cursor = encode_cursor(end) if end < cutoff else None
        ranked_candidates: List[ScanCandidate] = []
        for idx, candidate in enumerate(ordered_candidates, start=1):
            ranked_candidates.append(candidate.model_copy(update={"rank": idx}))
        page_candidates = ranked_candidates[start:end]
    else:
        page_candidates = []
        next_cursor = None

    meta = {
        "style": request_payload.style,
        "limit": request_payload.limit,
        "universe_size": len(filtered_symbols),
        "symbols": filtered_symbols[: min(len(filtered_symbols), 50)],
    }
    if context.simulate_open:
        meta["simulated_open"] = True

    planning_context = "frozen" if str(dq.get("mode")).lower() == "degraded" else context.label
    latency_ms = int((time.perf_counter() - started) * 1000)
    metric_count = _record_metric(
        "gpt_scan",
        session=str(session_payload.get("status") or "unknown"),
        context=planning_context,
    )
    logger.info(
        "scan completed",
        extra={
            "scan_universe_size": len(tickers),
            "screened": len(filtered_symbols),
            "returned": len(ordered_candidates),
            "planning_context": planning_context,
            "banner_present": bool(banner),
            "page_size": len(page_candidates),
            "latency_ms": latency_ms,
            "session_status": session_payload.get("status"),
            "session_as_of": session_payload.get("as_of"),
            "metric_count": metric_count,
            "sim_open": context.simulate_open,
        },
    )
    return _finalize(
        ScanPage(
            as_of=context.as_of_iso,
            planning_context=planning_context,
            banner=banner,
            meta=meta,
            candidates=page_candidates,
            data_quality=dq,
            session=session_payload,
            phase="scan",
            count_candidates=len(page_candidates),
            next_cursor=next_cursor,
        )
    )


async def _legacy_scan(
    universe: ScanUniverse,
    request: Request,
    user: AuthedUser,
    *,
    simulate_open: bool = False,
    stub_mode: bool = False,
    fetch_options: bool | None = None,
) -> List[Dict[str, Any]]:
    resolved_tickers: List[str] = []
    exclude_set = {symbol.upper() for symbol in (universe.exclude or []) if symbol}
    include_symbols = [symbol.upper() for symbol in (universe.include or []) if symbol]
    requested_limit = universe.limit or 60
    limit = max(10, min(requested_limit, 250))

    if fetch_options is None:
        fetch_options = not stub_mode

    if universe.tickers:
        base_symbols = [symbol.upper() for symbol in universe.tickers if symbol]
        resolved_tickers = await _expand_universe_tokens(base_symbols, style=universe.style, limit=limit)
    else:
        try:
            resolved_tickers = await load_universe(
                style=universe.style,
                sector=universe.sector,
                limit=limit,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("auto-universe build failed: %s", exc)
            resolved_tickers = []
        if not resolved_tickers:
            raise HTTPException(status_code=502, detail="Ticker universe unavailable")

    if include_symbols:
        expanded_includes = await _expand_universe_tokens(include_symbols, style=universe.style, limit=limit)
        resolved_tickers = list(dict.fromkeys(expanded_includes + resolved_tickers))
    if exclude_set:
        resolved_tickers = [symbol for symbol in resolved_tickers if symbol not in exclude_set]

    if not resolved_tickers:
        raise HTTPException(status_code=400, detail="No tickers available after applying filters")
    if len(resolved_tickers) > limit:
        resolved_tickers = resolved_tickers[:limit]

    session_payload = _session_payload_from_request(request)
    market_meta, data_meta, as_of_dt, is_open = _market_snapshot_payload(
        session_payload,
        simulate_open=simulate_open,
    )
    simulated_banner_text = _format_simulated_banner(as_of_dt) if simulate_open else None

    style_filter = _normalize_style(universe.style)
    data_timeframe = {"scalp": "1", "intraday": "5", "swing": "60", "leap": "D"}.get(style_filter, "5")

    settings = get_settings()
    index_mode = _get_index_mode()
    index_symbols = {symbol for symbol in resolved_tickers if index_mode and index_mode.applies(symbol)}
    index_counters = {"success": 0, "fallback": 0}
    market_data, data_source_map = await _collect_market_data(
        resolved_tickers,
        timeframe=data_timeframe,
        as_of=None if is_open else as_of_dt,
    )
    if index_mode and index_symbols:
        synthetic_count = 0
        for index_symbol in list(index_symbols):
            frame = market_data.get(index_symbol)
            if frame is not None and not frame.empty:
                continue
            try:
                synthetic = await index_mode.synthetic_ohlcv(index_symbol, data_timeframe)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("synthetic index build failed", extra={"symbol": index_symbol, "error": str(exc)})
                synthetic = None
            if synthetic is not None and not synthetic.empty:
                market_data[index_symbol] = synthetic
                data_source_map[index_symbol] = synthetic.attrs.get("source", "proxy_gamma")
                synthetic_count += 1
                logger.warning("Using proxy-translated data for %s", index_symbol)
        if synthetic_count:
            data_meta.setdefault("mode", "degraded")
            data_meta.setdefault("banners", []).append("Index bars unavailable — translating via ETF proxy.")
    data_meta["sources"] = data_source_map
    missing_symbols = [
        symbol
        for symbol, src in data_source_map.items()
        if src.startswith("missing") or src.endswith("_empty")
    ]
    degraded_sources = [src for src in data_source_map.values() if src != "polygon"]
    if missing_symbols:
        data_meta["ok"] = False
        data_meta["error"] = "data_gap"
        data_meta["missing_symbols"] = missing_symbols
        data_meta.setdefault("mode", "degraded")
        logger.warning("Market data missing for %s", ",".join(missing_symbols))
    if degraded_sources and not missing_symbols:
        data_meta.setdefault("mode", "degraded")
        if data_meta.get("ok", True):
            data_meta["ok"] = False
    if data_meta.get("mode") == "degraded":
        data_meta.setdefault("banner", "Live feed degraded — using cached market data.")
    if not market_data:
        logger.warning("No market data available after fallbacks for %s", resolved_tickers)
        return []
    try:
        signals = await scan_market(
            resolved_tickers,
            market_data,
            as_of=as_of_dt,
            simulate_open=simulate_open,
        )
    except TypeError:
        signals = await scan_market(
            resolved_tickers,
            market_data,
            as_of=as_of_dt,
        )

    unique_symbols = sorted({signal.symbol for signal in signals})

    polygon_enabled = bool(settings.polygon_api_key)
    tradier_enabled = bool(settings.tradier_token)

    if stub_mode:
        stub_context = ScanContext(
            as_of=as_of_dt,
            label="live" if is_open else "frozen",
            is_open=is_open,
            simulate_open=simulate_open,
            data_timeframe=data_timeframe,
            market_meta=market_meta,
            data_meta=dict(data_meta),
        )
        stub_preps = await _legacy_scan_stub_payload(
            signals=signals,
            market_data=market_data,
            style_filter=style_filter,
            context=stub_context,
            request=request,
        )
        return [prep.candidate.model_dump() for prep in stub_preps]

    benchmark_symbol = "SPY"
    benchmark_history: pd.DataFrame | None = market_data.get(benchmark_symbol)
    if benchmark_history is None:
        try:
            benchmark_history = await _load_remote_ohlcv(benchmark_symbol, data_timeframe)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Benchmark data fetch failed for %s: %s", benchmark_symbol, exc)
            benchmark_history = None
    if benchmark_history is not None and as_of_dt is not None and not is_open:
        cutoff = pd.Timestamp(as_of_dt).tz_convert("UTC")
        benchmark_history = benchmark_history.loc[benchmark_history.index <= cutoff]
        if benchmark_history.empty:
            benchmark_history = None

    symbol_freshness: Dict[str, float] = {}
    data_meta.setdefault("ok", True)
    stale_ms_threshold = 90000
    if is_open:
        now_utc = pd.Timestamp.utcnow()
        for symbol_key, frame in market_data.items():
            last_ts = frame.index[-1]
            age_ms = max((now_utc - last_ts).total_seconds() * 1000.0, 0.0)
            symbol_freshness[symbol_key] = age_ms
        if symbol_freshness:
            stale_symbols = [sym for sym, age in symbol_freshness.items() if age > stale_ms_threshold]
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
                if max_age > stale_ms_threshold:
                    logger.warning("Detected potentially stale market data during RTH (max age %.0f ms)", max_age)
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

    data_meta["stale_threshold_ms"] = stale_ms_threshold if is_open else None

    polygon_chains: Dict[str, pd.DataFrame] = {}
    if fetch_options and unique_symbols and polygon_enabled:
        try:
            as_of_hint = None if is_open else as_of_dt
            tasks = [
                _fetch_option_chain_with_aliases(symbol, as_of_hint)
                for symbol in unique_symbols
            ]
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
    if polygon_enabled and unique_symbols:
        if any(not df.empty for df in polygon_chains.values()):
            data_meta.setdefault("options_mode", "polygon")
        else:
            data_meta.setdefault("options_mode", "tradier")
            data_meta.setdefault("mode", "degraded")
            data_meta["ok"] = False
    elif tradier_enabled:
        data_meta.setdefault("options_mode", "tradier")

    tradier_suggestions: Dict[str, Dict[str, Any] | None] = {}
    if fetch_options and unique_symbols and tradier_enabled:
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
        last_bar_ts = history.index[-1]
        if isinstance(last_bar_ts, pd.Timestamp):
            if last_bar_ts.tzinfo is None:
                last_bar_ts = last_bar_ts.tz_localize("UTC")
            else:
                last_bar_ts = last_bar_ts.tz_convert("UTC")
        else:
            last_bar_ts = pd.Timestamp.utcnow().tz_localize("UTC")
        last_update_iso = last_bar_ts.isoformat()
        freshness_val = None
        if symbol_freshness:
            try:
                freshness_val = float(symbol_freshness.get(signal.symbol))
            except (TypeError, ValueError):
                freshness_val = None
        source_label = data_source_map.get(signal.symbol)
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

        base_url = _resolved_base_url(request)
        interval = _timeframe_for_style(style)
        chart_query: Dict[str, str] = {}
        hint_interval_raw, hint_guidance = _chart_hint(signal.strategy_id, style)
        try:
            chart_interval_hint = normalize_interval(hint_interval_raw)
        except ValueError:
            chart_interval_hint = interval
        chart_query.update(
            {
                "symbol": signal.symbol.upper(),
                "interval": chart_interval_hint,
                "ema": ",".join(str(span) for span in ema_spans),
                "view": _view_for_style(style),
                "vwap": "1",
                "theme": "dark",
            }
        )
        chart_query["last_update"] = last_update_iso
        if freshness_val is not None:
            try:
                chart_query["data_age_ms"] = str(int(freshness_val))
            except (TypeError, ValueError):
                chart_query["data_age_ms"] = str(freshness_val)
        if source_label:
            chart_query["data_source"] = source_label
        if data_meta.get("mode"):
            chart_query["data_mode"] = str(data_meta.get("mode"))
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
        plan_id = None
        version = 1
        if signal.plan is not None:
            indicators = snapshot.get("indicators") or {}
            session_snapshot = snapshot.get("session") or {}
            volatility_snapshot = snapshot.get("volatility") or {}
            atr_val = indicators.get("atr14")
            try:
                atr_numeric = float(atr_val) if atr_val is not None else None
            except (TypeError, ValueError):
                atr_numeric = None
            expected_move_val = volatility_snapshot.get("expected_move_horizon")
            try:
                expected_move_numeric = float(expected_move_val) if expected_move_val is not None else None
            except (TypeError, ValueError):
                expected_move_numeric = None
            price_close = snapshot.get("price", {}).get("close") or entry_price
            try:
                price_numeric = float(price_close)
            except (TypeError, ValueError):
                price_numeric = float(entry_price)
            exec_context = PlanExecutionContext(
                symbol=signal.symbol,
                style=style,
                direction=signal.plan.direction or plan_direction or direction_hint,
                price=price_numeric,
                key_levels=dict(key_levels),
                atr14=atr_numeric,
                expected_move=expected_move_numeric,
                vwap=_safe_number(indicators.get("vwap")),
                ema_stack=(snapshot.get("trend") or {}).get("ema_stack"),
                session_phase=session_snapshot.get("phase"),
                minutes_to_close=session_snapshot.get("minutes_to_close"),
                data_mode=data_meta.get("mode"),
            )
            refined_plan, adjustments = refine_execution_plan(signal.plan, exec_context)
            signal.plan = refined_plan
            signal.score = refined_plan.confidence if refined_plan.confidence is not None else signal.score
            feature_updates = dict(signal.features or {})
            feature_updates.update(adjustments.feature_updates())
            feature_updates["plan_entry"] = refined_plan.entry
            feature_updates["plan_stop"] = refined_plan.stop
            feature_updates["plan_targets"] = list(refined_plan.targets)
            feature_updates["plan_confidence"] = refined_plan.confidence
            feature_updates["plan_risk_reward"] = refined_plan.risk_reward
            signal.features = feature_updates
        if signal.plan is not None:
            plan_payload = signal.plan.as_dict()
            plan_entry = float(signal.plan.entry)
            plan_stop = float(signal.plan.stop)
            plan_targets = [float(target) for target in signal.plan.targets]
            plan_direction = signal.plan.direction or plan_direction
            chart_query["entry"] = f"{signal.plan.entry:.2f}"
            chart_query["stop"] = f"{signal.plan.stop:.2f}"
            chart_query["tp"] = ",".join(f"{target:.2f}" for target in signal.plan.targets)
            chart_query.setdefault("direction", signal.plan.direction)
            if signal.plan.atr and "atr" not in chart_query:
                chart_query["atr"] = f"{float(signal.plan.atr):.4f}"
            raw_plan_id = str(plan_payload.get("plan_id") or "").strip()
            plan_id = raw_plan_id or _generate_plan_slug(
                signal.symbol,
                style,
                plan_direction or direction_hint,
                snapshot,
            )
            try:
                version = int(plan_payload.get("version") or 1)
            except (TypeError, ValueError):
                version = 1
        else:
            plan_id = _generate_plan_slug(
                signal.symbol,
                style,
                plan_direction or direction_hint,
                snapshot,
            )
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
        if plan_id:
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
            plan_payload.setdefault("chart_timeframe", chart_interval_hint)
            plan_payload.setdefault("chart_guidance", hint_guidance)
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
        if chart_query and required_chart_keys.issubset(chart_query.keys()):
            try:
                chart_links = await gpt_chart_url(ChartParams(**chart_query), request)
            except HTTPException as exc:
                logger.debug("chart link generation failed for %s: %s", signal.symbol, exc)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("chart link generation error for %s: %s", signal.symbol, exc)

        charts_payload: Dict[str, Any] = {
            "params": chart_query,
            "timeframe": chart_interval_hint,
            "guidance": hint_guidance,
        }
        charts_payload["last_update"] = last_update_iso
        if source_label:
            charts_payload["data_source"] = source_label
        if data_meta.get("mode"):
            charts_payload["data_mode"] = str(data_meta.get("mode"))
        if freshness_val is not None:
            charts_payload["data_age_ms"] = int(freshness_val)
        if chart_links:
            chart_query["interval"] = chart_interval_hint
            charts_payload["interactive"] = chart_links.interactive
        elif chart_query and required_chart_keys.issubset(chart_query.keys()):
            fallback_chart_url = _build_tv_chart_url(request, chart_query)
            charts_payload["interactive"] = fallback_chart_url
            logger.debug(
                "chart link fallback used",
                extra={"symbol": signal.symbol, "strategy_id": signal.strategy_id, "url": fallback_chart_url},
            )

        index_decision = None
        index_execution_proxy: Optional[Dict[str, object]] = None
        index_snapshot: Optional[GammaSnapshot] = None
        fallback_banner: Optional[str] = None
        settlement_note: Optional[str] = None
        index_metadata: Dict[str, Any] | None = None

        prefer_delta = _prefer_delta_for_style(style)
        if index_mode and signal.symbol in index_symbols:
            try:
                index_contract, index_decision = await index_mode.select_contract(
                    signal.symbol,
                    prefer_delta=prefer_delta,
                    style=style,
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("index contract selection failed", extra={"symbol": signal.symbol, "error": str(exc)})
                index_contract = None
                index_decision = None

            if index_decision:
                index_metadata = {
                    "mode": "index_sniper",
                    "preference": list(index_mode.contract_preference),
                    "source": index_decision.source,
                }
                proxy_symbol = index_mode.liquidity_proxy(signal.symbol)
                if proxy_symbol:
                    index_metadata["proxy"] = proxy_symbol
                if index_decision.health:
                    index_metadata["health"] = index_decision.health.to_dict()
                if index_decision.diagnostics:
                    index_metadata["diagnostics"] = dict(index_decision.diagnostics)
                if index_decision.execution_proxy:
                    index_execution_proxy = index_decision.execution_proxy
                    index_metadata["execution_proxy"] = index_decision.execution_proxy
                if index_decision.proxy_snapshot:
                    index_snapshot = index_decision.proxy_snapshot
                    index_metadata["gamma_snapshot"] = {
                        "gamma": round(index_snapshot.gamma_current, 6),
                        "gamma_mean": round(index_snapshot.gamma_mean, 6),
                        "gamma_drift": round(index_snapshot.drift, 6),
                        "ratio": round(index_snapshot.spot_ratio, 6),
                        "samples": index_snapshot.samples,
                    }
                if index_decision.fallback_note:
                    fallback_banner = index_decision.fallback_note
                if index_decision.source == "ETF_PROXY":
                    settlement_note = "ETF options are American, watch assignment/settlement."
                    proxy_symbol = None
                    if index_execution_proxy:
                        proxy_symbol = index_execution_proxy.get("symbol")
                    if not proxy_symbol and index_metadata:
                        proxy_symbol = index_metadata.get("proxy")
                    gamma_val = None
                    if index_execution_proxy:
                        gamma_val = index_execution_proxy.get("gamma")
                    if proxy_symbol:
                        if isinstance(gamma_val, (int, float)):
                            fallback_banner = (
                                f"Index chain degraded — executing via {proxy_symbol} with γ={gamma_val:.3f}; parity OK."
                            )
                        else:
                            fallback_banner = f"Index chain degraded — executing via {proxy_symbol}; parity OK."
                logger.info(
                    "index_mode_decision",
                    extra={
                        "symbol": signal.symbol,
                        "style": style,
                        "source": index_decision.source,
                        "contracts_source": index_decision.source,
                        "execution_proxy": index_decision.execution_proxy,
                        "gamma": (index_execution_proxy or {}).get("gamma"),
                        "fallback_note": fallback_banner,
                        "index_health": index_decision.health.to_dict() if index_decision.health else None,
                    },
                )
                if index_decision.source == "ETF_PROXY":
                    index_counters["fallback"] += 1
                elif index_decision.source in {"INDEX_POLYGON", "INDEX_TRADIER"}:
                    index_counters["success"] += 1

            best_contract = index_contract if index_contract else None
        else:
            best_contract = None

        if index_metadata and index_snapshot and plan_entry is not None and plan_stop is not None and plan_targets:
            try:
                index_metadata["translated_levels"] = {
                    "entry": index_mode.planner.translate_level(plan_entry, index_snapshot),
                    "stop": index_mode.planner.translate_level(plan_stop, index_snapshot),
                    "targets": index_mode.planner.translate_targets(plan_targets, index_snapshot),
                }
            except Exception:  # pragma: no cover - defensive
                logger.debug("index level translation failed", exc_info=True)

        polygon_bundle: Dict[str, Any] | None = None
        if polygon_chains:
            cache_key = (signal.symbol, signal.strategy_id)
            polygon_bundle = options_cache.get(cache_key)
            if polygon_bundle is None:
                chain = polygon_chains.get(signal.symbol)
                rules = signal.options_rules if isinstance(signal.options_rules, dict) else None
                polygon_bundle = (
                    summarize_polygon_chain(chain, rules=rules, top_n=3) if chain is not None else None
                )
                options_cache[cache_key] = polygon_bundle

        if best_contract is None:
            if polygon_bundle and polygon_bundle.get("best"):
                best_contract = polygon_bundle.get("best")
            else:
                best_contract = tradier_suggestions.get(signal.symbol)

        if plan_payload is not None and index_execution_proxy:
            plan_payload.setdefault("execution_proxy", index_execution_proxy)
        if plan_payload is not None and fallback_banner:
            plan_payload.setdefault("banners", []).append(fallback_banner)
        if plan_payload is not None and settlement_note:
            plan_payload.setdefault("execution_notes", []).append(settlement_note)
        if fallback_banner:
            charts_payload.setdefault("banners", []).append(fallback_banner)
        if settlement_note:
            charts_payload.setdefault("notes", []).append(settlement_note)

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
                "chart_timeframe": chart_interval_hint,
                "chart_guidance": hint_guidance,
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
                **({"index_mode": index_metadata} if index_metadata else {}),
                **({"fallback_banner": fallback_banner} if fallback_banner else {}),
                **({"settlement_note": settlement_note} if settlement_note else {}),
                **({"execution_proxy": index_execution_proxy} if index_execution_proxy else {}),
            }
        )

    if index_mode and (index_counters["success"] or index_counters["fallback"]):
        data_meta["index_mode_counts"] = dict(index_counters)

    logger.info("scan universe=%s user=%s results=%d", resolved_tickers, user.user_id, len(payload))
    return payload


async def gpt_scan(
    universe: ScanUniverse,
    request: Request,
    user: AuthedUser,
    *,
    simulate_open: bool = False,
) -> List[Dict[str, Any]]:
    """Backward-compatible helper used by legacy tests and gpt_plan."""

    return await _legacy_scan(
        universe,
        request,
        user,
        simulate_open=simulate_open,
    )


async def _generate_fallback_plan(
    symbol: str,
    style: str | None,
    request: Request,
    user: AuthedUser,
    *,
    simulate_open: bool = False,
) -> PlanResponse | None:
    settings = get_settings()
    style_token = _fallback_style_token(style)
    include_plan_layers = bool(
        getattr(settings, "ff_chart_canonical_v1", False) or getattr(settings, "ff_layers_endpoint", False)
    )
    include_options_contracts = True
    rejected_contracts: List[Dict[str, str]] = []
    options_contracts: List[Dict[str, Any]] | None = None
    options_note: str | None = None
    em_cap_used = False
    session_payload = _session_payload_from_request(request)
    market_meta, data_meta, as_of_dt, is_open = _market_snapshot_payload(
        session_payload,
        simulate_open=simulate_open,
    )
    is_plan_live = bool(is_open)
    timeframe_map = {"scalp": "1", "intraday": "5", "swing": "60", "leap": "D"}
    timeframe = timeframe_map.get(style_token, "5")
    try:
        market_data, data_sources = await _collect_market_data(
            [symbol],
            timeframe=timeframe,
            as_of=None if is_open else as_of_dt,
        )
    except HTTPException:
        return None
    data_meta["sources"] = data_sources
    index_mode = _get_index_mode()
    if index_mode and index_mode.applies(symbol):
        frame = market_data.get(symbol)
        if frame is None or frame.empty:
            try:
                synthetic = await index_mode.synthetic_ohlcv(symbol, timeframe)
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("fallback synthetic build failed for %s: %s", symbol, exc)
                synthetic = None
            if synthetic is not None and not synthetic.empty:
                market_data[symbol] = synthetic
                data_sources[symbol] = synthetic.attrs.get("source", "proxy_gamma")
                data_meta.setdefault("mode", "degraded")
                data_meta.setdefault("banners", []).append("Index bars unavailable — translating via ETF proxy.")
    frame = market_data.get(symbol)
    if frame is None or frame.empty:
        return None
    prepared = _prepare_symbol_frame(frame)
    if prepared.empty:
        return None
    context = _build_context(prepared)
    latest = context.get("latest")
    if latest is None:
        latest = prepared.iloc[-1]
    try:
        close_price = float(latest["close"])
    except (TypeError, ValueError, KeyError):
        close_price = float(prepared["close"].iloc[-1])
    ema9 = float(context.get("ema9") or latest.get("ema9") or close_price)
    ema20 = float(context.get("ema20") or latest.get("ema20") or close_price)
    ema50 = float(context.get("ema50") or latest.get("ema50") or close_price)
    vwap_value = context.get("vwap")
    atr_value = context.get("atr")
    if not isinstance(atr_value, (int, float)) or not math.isfinite(atr_value) or atr_value <= 0:
        atr_series = prepared.get("atr14")
        if atr_series is not None:
            atr_candidates = atr_series.dropna()
            if not atr_candidates.empty:
                atr_value = float(atr_candidates.iloc[-1])
    if not isinstance(atr_value, (int, float)) or not math.isfinite(atr_value) or atr_value <= 0:
        atr_value = max(close_price * 0.006, 0.25)
    key_levels = context.get("key") or {}
    snapshot = _build_market_snapshot(prepared, key_levels)
    ema_trend_up = ema9 > ema20 > ema50
    ema_trend_down = ema9 < ema20 < ema50
    expected_move_abs = context.get("expected_move_horizon")
    if ema_trend_up or (close_price >= ema20 >= ema50):
        direction = "long"
    elif ema_trend_down or (close_price <= ema20 <= ema50):
        direction = "short"
    else:
        direction = "long" if close_price >= ema50 else "short"
    session_high = key_levels.get("session_high")
    session_low = key_levels.get("session_low")
    or_high = key_levels.get("opening_range_high")
    or_low = key_levels.get("opening_range_low")
    if direction == "long":
        candidates = [close_price, session_high, or_high]
        entry_base = max(
            float(val)
            for val in candidates
            if isinstance(val, (int, float)) and math.isfinite(float(val))
        )
        entry = round(entry_base, 2)
        stop = round(entry - atr_value * 1.25, 2)
        if stop <= 0 or stop >= entry:
            stop = round(entry - max(atr_value, close_price * 0.004), 2)
        target_multipliers = {
            "scalp": (0.9, 1.4, 2.0),
            "intraday": (1.2, 1.9, 2.8),
            "swing": (2.2, 3.4, 4.8),
            "leap": (3.6, 5.2, 7.0),
        }.get(style_token, (1.2, 1.9, 2.8))
        targets = [round(entry + atr_value * mult, 2) for mult in target_multipliers]
        if session_high and isinstance(session_high, (int, float)) and session_high > entry:
            targets[0] = round(float(session_high), 2)
            targets[1] = round(entry + atr_value * 1.9, 2)
    else:
        candidates = [close_price, session_low, or_low]
        entry_base = min(
            float(val)
            for val in candidates
            if isinstance(val, (int, float)) and math.isfinite(float(val))
        )
        entry = round(entry_base, 2)
        stop = round(entry + atr_value * 1.25, 2)
        if stop <= entry:
            stop = round(entry + max(atr_value, close_price * 0.004), 2)
        target_multipliers = {
            "scalp": (0.9, 1.4, 2.0),
            "intraday": (1.2, 1.9, 2.8),
            "swing": (2.2, 3.4, 4.8),
            "leap": (3.6, 5.2, 7.0),
        }.get(style_token, (1.2, 1.9, 2.8))
        targets = [round(entry - atr_value * mult, 2) for mult in target_multipliers]
        if session_low and isinstance(session_low, (int, float)) and session_low < entry:
            targets[0] = round(float(session_low), 2)
            targets[1] = round(entry - atr_value * 1.9, 2)
    rr_to_t1 = _risk_reward(entry, stop, targets[0], direction)
    if rr_to_t1 is None or rr_to_t1 < 1.2:
        scale = 1.2 / max(rr_to_t1 or 0.6, 0.4)
        if direction == "long":
            targets = [round(entry + (target - entry) * scale, 2) for target in targets]
        else:
            targets = [round(entry - (entry - target) * scale, 2) for target in targets]
        rr_to_t1 = _risk_reward(entry, stop, targets[0], direction)
    em_factor = float(getattr(settings, "ft_em_factor", 1.1))
    capped_targets, cap_used = _apply_em_cap(direction, entry, targets, expected_move_abs, em_factor)
    if capped_targets and len(capped_targets) == len(targets):
        targets = [round(val, 2) for val in capped_targets]
        em_cap_used = cap_used
        rr_to_t1 = _risk_reward(entry, stop, targets[0], direction)
    else:
        targets = [round(float(val), 2) for val in targets]
    valid_levels, invariant_reason = _validate_level_invariants(direction, entry, stop, targets)
    if not valid_levels:
        raise HTTPException(
            status_code=422,
            detail={"code": "INVARIANT_BROKEN", "message": invariant_reason or "invalid_levels"},
        )
    rr_to_t1 = float(rr_to_t1) if rr_to_t1 is not None else None
    trend_component = 0.8 if (direction == "long" and ema_trend_up) or (direction == "short" and ema_trend_down) else 0.6
    liquidity_component = 0.65 if isinstance(context.get("volume_median"), (int, float)) and context["volume_median"] > 0 else 0.5
    regime_component = 0.6 if isinstance(context.get("atr_1d"), (int, float)) and math.isfinite(context["atr_1d"] or float("nan")) else 0.52
    confidence = round(_fallback_confidence(trend_component, liquidity_component, regime_component), 2)
    confidence_factors: List[str] = []
    if (direction == "long" and ema_trend_up) or (direction == "short" and ema_trend_down):
        confidence_factors.append("EMA alignment")
    if isinstance(vwap_value, (int, float)) and math.isfinite(vwap_value):
        if direction == "long" and close_price >= vwap_value:
            confidence_factors.append("Above VWAP")
        if direction == "short" and close_price <= vwap_value:
            confidence_factors.append("Below VWAP")
    confluence_tags = _unique_tags(confidence_factors)
    expected_move_basis = None
    if isinstance(expected_move_abs, (int, float)) and math.isfinite(expected_move_abs) and expected_move_abs > 0:
        expected_move_pct = (expected_move_abs / close_price) * 100 if close_price else 0.0
        expected_move_basis = f"EM ≈ {expected_move_abs:.2f} ({expected_move_pct:.1f}%)"
    confidence_visual = _confidence_visual(confidence)
    target_meta = []
    for idx, target in enumerate(targets, start=1):
        distance = round(abs(target - entry), 2)
        prob_base = 0.58 if idx == 1 else 0.34 if idx == 2 else 0.2
        trend_bonus = 0.07 if confidence > 0.75 and idx == 1 else 0.0
        prob = _clamp(prob_base + trend_bonus, 0.15, 0.92)
        target_meta.append(
            {
                "label": f"TP{idx}",
                "prob_touch": round(prob, 2),
                "distance": distance,
            }
        )
    tp_reasons = _tp_reason_entries(target_meta)
    runner_trail = "ATR(14) x 0.8" if style_token in {"scalp", "intraday"} else "ATR(14) x 1.0"
    runner = {"trail": runner_trail}
    direction_label = "Long" if direction == "long" else "Short"
    notes = (
        f"{direction_label} bias with EMA stack supportive; manage risk using {runner_trail} and watch VWAP for continuation."
    )
    options_payload: Dict[str, Any] | None = None
    rejected_contracts: List[Dict[str, str]] = []
    style_public = (public_style(style_token) or style_token or "intraday").lower()
    side_hint = _infer_contract_side(None, direction)
    if include_options_contracts and side_hint:
        plan_anchor = {
            "underlying_entry": entry,
            "stop": stop,
            "targets": targets[:2],
            "horizon_minutes": 60 if style_public in {"swing", "leaps"} else 30,
        }
        contract_request = ContractsRequest(
            symbol=symbol,
            side=side_hint,
            style=style_public,
            selection_mode="analyze",
            plan_anchor=plan_anchor,
        )
        try:
            options_payload = await gpt_contracts(contract_request, user)
        except HTTPException as exc:
            logger.info("fallback contract lookup skipped for %s: %s", symbol, exc)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("fallback contract lookup error for %s: %s", symbol, exc)
    if include_options_contracts:
        extracted_contracts = _extract_options_contracts(options_payload)
        filtered_contracts, guardrail_rejections = _apply_option_guardrails(
            extracted_contracts,
            max_spread_pct=float(getattr(settings, "ft_max_spread_pct", 8.0)),
            min_open_interest=int(getattr(settings, "ft_min_oi", 300)),
        )
        if guardrail_rejections:
            rejected_contracts.extend(guardrail_rejections)
        if filtered_contracts:
            options_contracts = filtered_contracts
        else:
            options_contracts = []
            if guardrail_rejections and not options_note:
                options_note = "Contracts filtered by liquidity guardrails"
            elif isinstance(options_payload, dict) and options_payload.get("quotes_notice"):
                options_note = str(options_payload["quotes_notice"])
            elif not side_hint:
                options_note = "Options side unavailable for this plan"
            else:
                options_note = "No tradeable contracts met filters"
    if include_options_contracts and event_window_blocked:
        options_contracts = []
        options_note = "Blocked by event window"
        if not any(rc.get("reason") == "EVENT_WINDOW_BLOCKED" for rc in rejected_contracts):
            rejected_contracts.append({"symbol": symbol.upper(), "reason": "EVENT_WINDOW_BLOCKED"})
    if include_options_contracts and event_window_blocked:
        options_contracts = []
        options_note = "Blocked by event window"
        if not any(rc.get("reason") == "EVENT_WINDOW_BLOCKED" for rc in rejected_contracts):
            rejected_contracts.append({"symbol": symbol.upper(), "reason": "EVENT_WINDOW_BLOCKED"})
    else:
        options_payload = None
    hint_interval_raw, hint_guidance = _chart_hint("baseline_auto", style_token)
    try:
        chart_timeframe_hint = normalize_interval(hint_interval_raw)
    except ValueError:
        chart_timeframe_hint = "5m"
    mtf_confluence_tags = []
    try:
        mtf_confluence_tags = await _compute_multi_timeframe_confluence(
            symbol=symbol,
            base_interval=chart_timeframe_hint or "5m",
            snapshot=snapshot,
            direction=direction,
            entry=entry,
            overlays=None,
        )
    except Exception:  # pragma: no cover - defensive
        mtf_confluence_tags = []
    stop_multiple_val = None
    if atr_value and atr_value > 0:
        stop_multiple_val = abs(entry - stop) / atr_value
    risk_block = _build_risk_block(
        entry=entry,
        stop=stop,
        targets=targets,
        atr=atr_value,
        stop_multiple=stop_multiple_val,
        expected_move=expected_move_abs,
        runner=runner,
    )
    execution_rules = _build_execution_rules(
        entry=entry,
        stop=stop,
        targets=targets,
        direction=direction,
        precision=get_precision(symbol),
        key_levels_used=None,
        runner=runner,
    )
    plan_ts = context.get("timestamp")
    if isinstance(plan_ts, pd.Timestamp):
        plan_ts_utc = plan_ts.tz_convert("UTC") if plan_ts.tzinfo else plan_ts.tz_localize("UTC")
    else:
        plan_ts_utc = pd.Timestamp.utcnow()
    if is_plan_live:
        plan_ts_utc = pd.Timestamp.utcnow()
    plan_id = f"{symbol.upper()}-{plan_ts_utc.strftime('%Y%m%dT%H%M%S')}-{style_token}-auto"
    view_token = _view_for_style(style_token)
    range_token = _range_for_style(style_token)
    interval_map = {"1": "1m", "5": "5m", "60": "1h", "D": "d"}
    interval_token = interval_map.get(timeframe, chart_timeframe_hint or "5m")
    chart_params: Dict[str, Any] = {
        "symbol": _tv_symbol(symbol),
        "interval": chart_timeframe_hint or interval_token,
        "direction": direction,
        "entry": f"{entry:.2f}",
        "stop": f"{stop:.2f}",
        "tp": ",".join(f"{target:.2f}" for target in targets),
        "ema": "9,20,50",
        "focus": "plan",
        "center_time": "latest",
        "scale_plan": "auto",
        "view": view_token,
        "range": range_token,
        "theme": "dark",
    }
    _attach_market_chart_params(chart_params, market_meta, data_meta)
    if is_plan_live:
        live_stamp = datetime.now(timezone.utc).isoformat()
        chart_params["live"] = "1"
        chart_params["last_update"] = live_stamp
    levels_param = _fallback_levels_param({k: v for k, v in key_levels.items() if isinstance(v, (int, float))})
    if levels_param:
        chart_params["levels"] = levels_param
    chart_links = None
    try:
        chart_links = await gpt_chart_url(ChartParams(**chart_params), request)
    except HTTPException:
        chart_links = None
    chart_url = chart_links.interactive if chart_links else _build_tv_chart_url(request, chart_params)
    chart_url_with_ids = _append_query_params(
        chart_url,
        {
            "plan_id": plan_id,
            "plan_version": "1",
        },
    )
    plan_block = {
        "symbol": symbol.upper(),
        "style": style_token,
        "direction": direction,
        "entry": entry,
        "stop": stop,
        "targets": targets,
        "target_meta": target_meta,
        "runner": runner,
        "confidence": confidence,
        "notes": notes,
        "rr_to_t1": rr_to_t1,
        "expected_move": expected_move_abs,
        "atr": atr_value,
        "interval": interval_token,
        "warnings": [],
        "strategy": "baseline_auto",
        "debug": {},
    }
    plan_block["trade_detail"] = chart_url_with_ids
    plan_block["chart_timeframe"] = chart_timeframe_hint
    plan_block["chart_guidance"] = hint_guidance
    plan_block["within_event_window"] = within_event_window
    if minutes_to_event is not None:
        plan_block["minutes_to_event"] = minutes_to_event
    if em_cap_used:
        plan_block["em_used"] = True
    if simulated_banner_text:
        banners_list = plan_block.get("banners") if isinstance(plan_block.get("banners"), list) else None
        if banners_list is None:
            banners_list = []
            plan_block["banners"] = banners_list
        if simulated_banner_text not in banners_list:
            banners_list.append(simulated_banner_text)
    if confidence_visual:
        plan_block["confidence_visual"] = confidence_visual
    if confluence_tags:
        plan_block["confluence_tags"] = confluence_tags
    if tp_reasons:
        plan_block["tp_reasons"] = tp_reasons
    if include_options_contracts and options_contracts is not None:
        plan_block["options_contracts"] = options_contracts
    if include_options_contracts and options_note:
        plan_block["options_note"] = options_note
    if rejected_contracts:
        plan_block["rejected_contracts"] = rejected_contracts
    if mtf_confluence_tags:
        plan_block["mtf_confluence"] = mtf_confluence_tags
    if risk_block:
        plan_block["risk_block"] = risk_block
    if execution_rules:
        plan_block["execution_rules"] = execution_rules
    plan_warnings: List[str] = list(event_warnings)
    target_profile = build_target_profile(
        entry=entry,
        stop=stop,
        targets=targets,
        target_meta=plan_block.get("target_meta"),
        debug=plan_block.get("debug"),
        runner=runner,
        warnings=plan_warnings,
        atr_used=atr_value,
        expected_move=expected_move_abs,
        style=style_token,
        bias=direction,
    )
    structured_plan = build_structured_plan(
        plan_id=plan_id,
        symbol=symbol.upper(),
        style=style_token,
        direction=direction,
        profile=target_profile,
        confidence=confidence,
        rationale=notes,
        options_block=None,
        chart_url=chart_url_with_ids,
        session=session_state_payload,
        confluence=confluence_tags or None,
    )
    if confidence_visual:
        structured_plan["confidence_visual"] = confidence_visual
    structured_plan["trade_detail"] = chart_url_with_ids
    structured_plan["within_event_window"] = within_event_window
    if minutes_to_event is not None:
        structured_plan["minutes_to_event"] = minutes_to_event
    if em_cap_used:
        structured_plan["em_used"] = True
    if tp_reasons:
        structured_plan["tp_reasons"] = tp_reasons
    if confluence_tags:
        structured_plan["confluence_tags"] = confluence_tags
    if include_options_contracts and options_contracts is not None:
        structured_plan["options_contracts"] = options_contracts
    if include_options_contracts and options_note:
        structured_plan["options_note"] = options_note
    if rejected_contracts:
        structured_plan["rejected_contracts"] = rejected_contracts
    structured_plan["target_profile"] = target_profile_dict
    if mtf_confluence_tags:
        structured_plan["mtf_confluence"] = mtf_confluence_tags
    if risk_block:
        structured_plan["risk_block"] = risk_block
    if execution_rules:
        structured_plan["execution_rules"] = execution_rules
    if simulated_banner_text:
        banners_list = structured_plan.get("banners") if isinstance(structured_plan.get("banners"), list) else None
        if banners_list is None:
            banners_list = []
            structured_plan["banners"] = banners_list
        if simulated_banner_text not in banners_list:
            banners_list.append(simulated_banner_text)
    target_profile_dict = target_profile.to_dict()
    if em_cap_used:
        target_profile_dict["em_used"] = True
    warnings_payload = list(dict.fromkeys(plan_warnings)) if plan_warnings else []
    chart_params_payload = {key: str(value) for key, value in chart_params.items()}
    charts_payload: Dict[str, Any] = {"params": chart_params_payload, "interactive": chart_url_with_ids}
    charts_payload["live"] = is_plan_live
    charts_payload["timeframe"] = chart_timeframe_hint
    charts_payload["guidance"] = hint_guidance
    if simulated_banner_text:
        charts_payload.setdefault("banners", []).append(simulated_banner_text)
    enrichment = None
    events_block: Dict[str, Any] | None = None
    earnings_block: Dict[str, Any] | None = None
    sentiment_block: Dict[str, Any] | None = None
    try:
        enrichment = await _fetch_context_enrichment(symbol)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("fallback enrichment fetch failed for %s: %s", symbol, exc)
        enrichment = None
    if enrichment:
        events_block = enrichment.get("events")
        earnings_block = enrichment.get("earnings")
        sentiment_block = enrichment.get("sentiment")
    session_state_payload_raw = market_meta.get("session_state") if isinstance(market_meta, dict) else None
    if session_state_payload_raw is None and isinstance(data_meta, dict):
        session_state_payload_raw = data_meta.get("session_state")
    session_state_payload = dict(session_state_payload_raw) if isinstance(session_state_payload_raw, Mapping) else {}
    if not events_block:
        macro_window = _macro_event_block(symbol, session_state_payload)
        if macro_window:
            events_block = macro_window
    within_event_window = bool((events_block or {}).get("within_event_window"))
    minutes_to_event_token = (events_block or {}).get("min_minutes_to_event")
    minutes_to_event: Optional[int] = None
    if minutes_to_event_token is not None:
        try:
            minutes_to_event = int(float(minutes_to_event_token))
        except (TypeError, ValueError):
            minutes_to_event = None
    session_state_payload["within_event_window"] = within_event_window
    if minutes_to_event is not None:
        session_state_payload["minutes_to_event"] = minutes_to_event
    blocked_styles = {
        token.strip().lower()
        for token in getattr(settings, "ft_event_blocked_styles", [])
        if token and token.strip()
    }
    style_lower = (style_token or "").strip().lower()
    event_window_blocked = within_event_window and (style_lower in blocked_styles)
    event_warnings: List[str] = []
    if event_window_blocked:
        event_warnings.append("EVENT_WINDOW_BLOCKED")
    if symbol.upper() in ETF_EVENT_SYMBOLS:
        earnings_block = None
    if isinstance(data_meta, dict):
        data_meta = dict(data_meta)
        data_meta["earnings_present"] = bool(earnings_block)
        data_meta["events_present"] = bool(events_block)
    data_payload = dict(data_meta)
    bars_url = f"{_resolved_base_url(request)}/gpt/context/{symbol.upper()}?interval={interval_token}&lookback=300"
    if is_plan_live:
        bars_url += "&live=1"
    data_payload["bars"] = bars_url
    data_payload["session_state"] = session_state_payload
    data_payload["earnings_present"] = bool(earnings_block)
    data_payload["events_present"] = bool(events_block)
    data_payload["within_event_window"] = within_event_window
    if minutes_to_event is not None:
        data_payload["minutes_to_event"] = minutes_to_event
    data_quality = {
        "series_present": True,
        "iv_present": bool(expected_move_abs) and math.isfinite(expected_move_abs),
        "events_present": bool(events_block),
        "earnings_present": bool(earnings_block),
    }
    data_quality["within_event_window"] = within_event_window
    if minutes_to_event is not None:
        data_quality["minutes_to_event"] = minutes_to_event
    if event_window_blocked:
        data_quality["event_window_blocked"] = True
    confluence_payload = list(confluence_tags) if confluence_tags else []
    mtf_confluence_payload = list(mtf_confluence_tags) if mtf_confluence_tags else []
    tp_reasons_payload = list(tp_reasons) if tp_reasons else []
    options_contracts_payload: List[Dict[str, Any]] = (
        list(options_contracts) if include_options_contracts and options_contracts else []
    )
    plan_response = PlanResponse(
        plan_id=plan_id,
        version=1,
        trade_detail=chart_url_with_ids,
        warnings=warnings_payload,
        planning_context="live" if is_plan_live else "frozen",
        symbol=symbol.upper(),
        style=style_token,
        bias=direction,
        setup="baseline_auto",
        entry=entry,
        stop=stop,
        targets=targets,
        target_meta=target_meta,
        rr_to_t1=rr_to_t1,
        confidence=confidence,
        confidence_factors=confidence_factors or None,
        confluence_tags=confluence_payload,
        confluence=mtf_confluence_payload,
        notes=notes,
        relevant_levels={k: float(v) for k, v in key_levels.items() if isinstance(v, (int, float))},
        expected_move_basis=expected_move_basis,
        sentiment=sentiment_block,
        events=events_block,
        earnings=earnings_block,
        charts_params=chart_params_payload,
        chart_url=chart_url_with_ids,
        strategy_id="baseline_auto",
        description=None,
        score=confidence,
        plan=plan_block,
        structured_plan=structured_plan,
        target_profile=target_profile_dict,
        charts=charts_payload,
        key_levels={k: float(v) for k, v in key_levels.items() if isinstance(v, (int, float))},
        key_levels_used=None,
        market_snapshot=market_meta,
        features={
            "ema_trend_up": ema_trend_up,
            "ema_trend_down": ema_trend_down,
            "vwap": vwap_value,
        },
        options=options_payload if isinstance(options_payload, dict) else None,
        options_contracts=options_contracts_payload,
        options_note=options_note if include_options_contracts else None,
        calc_notes={
            "atr14": round(float(atr_value), 4),
            "rr_inputs": {"entry": entry, "stop": stop, "tp1": targets[0]},
        },
        htf={
            "bias": direction,
            "snapped_targets": [],
        },
        decimals=2,
        data_quality=data_quality,
        debug={"source": "auto_fallback_plan"},
        runner=runner,
        risk_block=risk_block,
        execution_rules=execution_rules,
        market=market_meta,
        data=data_payload,
        session_state=session_state_payload,
        chart_timeframe=chart_timeframe_hint,
        chart_guidance=hint_guidance,
        confidence_visual=confidence_visual,
        tp_reasons=tp_reasons_payload,
        rejected_contracts=rejected_contracts,
    )
    if simulate_open:
        response_meta = {"simulated_open": True}
        if simulated_banner_text:
            response_meta["banner"] = simulated_banner_text
        plan_response.meta = response_meta
    if include_plan_layers:
        layers = build_plan_layers(
            symbol=symbol.upper(),
            interval=str(chart_timeframe_hint or timeframe),
            as_of=session_payload.get("as_of"),
            planning_context="live" if is_plan_live else "frozen",
            key_levels={k: v for k, v in key_levels.items() if isinstance(v, (int, float))},
            overlays=None,
        )
        layers["plan_id"] = plan_id
        meta = layers.setdefault("meta", {})
        if confidence_factors:
            meta["confidence_factors"] = list(confidence_factors)
        if confluence_tags:
            meta["confluence"] = list(confluence_tags)
        else:
            meta.setdefault(
                "confluence",
                [
                    label
                    for label in (
                        "EMA alignment" if (direction == "long" and ema_trend_up) or (direction == "short" and ema_trend_down) else None,
                        "VWAP confirmation" if isinstance(vwap_value, (int, float)) else None,
                    )
                    if label
                ],
            )
        meta.setdefault(
            "features",
            {
                "ema_trend_up": bool(ema_trend_up),
                "ema_trend_down": bool(ema_trend_down),
                "vwap": float(vwap_value) if isinstance(vwap_value, (int, float)) else None,
            },
        )
        if mtf_confluence_tags:
            meta["mtf_confluence"] = mtf_confluence_tags
        plan_response.plan_layers = layers
    plan_response.phase = "hydrate"
    plan_response.layers_fetched = bool(plan_response.plan_layers)
    meta_payload = dict(plan_response.meta or {})
    meta_payload["within_event_window"] = within_event_window
    if minutes_to_event is not None:
        meta_payload["minutes_to_event"] = minutes_to_event
    if em_cap_used:
        meta_payload["em_used"] = True
    plan_response.meta = meta_payload or None
    return _finalize_plan_response(plan_response)


@gpt.post("/plan", summary="Return a single trade plan for a symbol", response_model=PlanResponse)
@gpt.post('/plan', summary='Return a single trade plan for a symbol', response_model=PlanResponse)
async def gpt_plan(
    request_payload: PlanRequest,
    request: Request,
    response: Response,
    user: AuthedUser = Depends(require_api_key),
) -> PlanResponse:
    """Compatibility endpoint that returns the top plan for a single symbol.

    Internally reuses /gpt/scan to keep plan logic centralized.
    """
    symbol = (request_payload.symbol or "").strip().upper()
    if not symbol:
        raise HTTPException(status_code=400, detail="Symbol is required")
    if not _PLAN_SYMBOL_PATTERN.fullmatch(symbol):
        raise HTTPException(status_code=400, detail="Invalid symbol; use /gpt/scan for universe requests.")
    forced_plan_id = (request_payload.plan_id or "").strip()
    settings = get_settings()
    allow_plan_fallback = not bool(getattr(settings, "ft_no_fallback_trades", True))
    fields_set = getattr(request_payload, "model_fields_set", set())
    simulate_open = _resolve_simulate_open(
        request,
        explicit_value=request_payload.simulate_open,
        explicit_field_set="simulate_open" in fields_set,
    )
    session_payload = _session_payload_from_request(request)
    include_plan_layers = bool(
        getattr(settings, "ff_chart_canonical_v1", False) or getattr(settings, "ff_layers_endpoint", False)
    )
    include_options_contracts = True
    logger.info(
        "gpt_plan received",
        extra={
            "symbol": symbol,
            "style": request_payload.style,
            "user_id": getattr(user, "user_id", None),
            "sim_open": simulate_open,
        },
    )
    universe = ScanUniverse(tickers=[symbol], style=request_payload.style)
    try:
        results = await gpt_scan(universe, request, user, simulate_open=simulate_open)
    except TypeError:
        results = await gpt_scan(universe, request, user)
    if not results:
        if allow_plan_fallback:
            fallback_plan = await _generate_fallback_plan(
                symbol,
                request_payload.style,
                request,
                user,
                simulate_open=simulate_open,
            )
            if fallback_plan is not None:
                logger.info(
                    "gpt_plan fallback_generated",
                    extra={"symbol": symbol, "style": request_payload.style, "mode": "no_signals"},
                )
                return _finalize_plan_response(fallback_plan)
        raise HTTPException(status_code=404, detail=f"No valid plan for {symbol}")
    first = next((item for item in results if (item.get("symbol") or "").upper() == symbol), results[0])
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
    expected_move_abs: float | None = None
    if isinstance(plan.get("expected_move"), (int, float, str)):
        try:
            candidate = float(plan.get("expected_move"))
            if math.isfinite(candidate) and candidate > 0:
                expected_move_abs = candidate
        except (TypeError, ValueError):
            expected_move_abs = None
    if expected_move_abs is None and isinstance(volatility.get("expected_move_horizon"), (int, float, str)):
        try:
            candidate = float(volatility.get("expected_move_horizon"))
            if math.isfinite(candidate) and candidate > 0:
                expected_move_abs = candidate
        except (TypeError, ValueError):
            expected_move_abs = None
    if expected_move_abs is None and isinstance(data_meta, dict):
        try:
            candidate = float(data_meta.get("expected_move_horizon"))
            if math.isfinite(candidate) and candidate > 0:
                expected_move_abs = candidate
        except (TypeError, ValueError, TypeError):
            expected_move_abs = None
    em_cap_used = False
    # Build calc_notes + htf from available payload
    raw_plan = first.get("plan") or {}
    if not raw_plan:
        if allow_plan_fallback:
            fallback_plan = await _generate_fallback_plan(
                symbol,
                request_payload.style,
                request,
                user,
                simulate_open=simulate_open,
            )
            if fallback_plan is not None:
                logger.info(
                    "gpt_plan fallback_generated",
                    extra={"symbol": symbol, "style": request_payload.style, "mode": "empty_plan"},
                )
                return _finalize_plan_response(fallback_plan)
        raise HTTPException(status_code=404, detail=f"No valid plan for {symbol}")
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

    strategy_id_value = first.get("strategy_id") or plan.get("strategy")
    style_token = plan.get("style") or first.get("style") or request_payload.style
    hint_interval_raw, hint_guidance = _chart_hint(strategy_id_value, style_token)
    try:
        chart_timeframe_hint = normalize_interval(hint_interval_raw)
    except ValueError:
        chart_timeframe_hint = "5m"
    plan["chart_timeframe"] = chart_timeframe_hint
    plan["chart_guidance"] = hint_guidance
    is_plan_live = str(first.get("planning_context") or "").strip().lower() == "live"
    simulated_banner_text: str | None = None
    if simulate_open:
        first["planning_context"] = "live"
        is_plan_live = True
        for key in ("meta", "data", "market"):
            container = first.get(key)
            if isinstance(container, dict):
                container.setdefault("simulated_open", True)
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
    planning_context_value: str | None = None
    market_meta_context = first.get("market") or first.get("meta")
    data_meta_context = first.get("data") or first.get("meta")
    market_meta = market_meta_context if isinstance(market_meta_context, dict) else None
    data_meta = data_meta_context if isinstance(data_meta_context, dict) else None
    if market_meta is None or data_meta is None:
        fallback_market, fallback_data, _, _ = _market_snapshot_payload(session_payload)
        if market_meta is None:
            market_meta = fallback_market
        if data_meta is None:
            data_meta = fallback_data
        else:
            data_meta.setdefault("as_of_ts", fallback_data["as_of_ts"])
            data_meta.setdefault("frozen", fallback_data["frozen"])
            data_meta.setdefault("ok", fallback_data.get("ok", True))
    if simulate_open:
        banner_dt: datetime | None = None
        if isinstance(data_meta, dict):
            ts_value = data_meta.get("as_of_ts")
            try:
                if ts_value is not None:
                    banner_dt = datetime.fromtimestamp(float(ts_value) / 1000.0, tz=timezone.utc)
            except (TypeError, ValueError):
                banner_dt = None
        if banner_dt is None:
            banner_dt = parse_session_as_of(session_payload)
        if banner_dt is None:
            banner_dt = datetime.now(timezone.utc)
        simulated_banner_text = _format_simulated_banner(banner_dt)
        if simulated_banner_text:
            existing_banners = plan.get("banners")
            if isinstance(existing_banners, list):
                banners_list = existing_banners
            else:
                banners_list = []
                plan["banners"] = banners_list
            if simulated_banner_text not in banners_list:
                banners_list.append(simulated_banner_text)
    session_state_payload: Dict[str, Any] | None = None
    if isinstance(market_meta, dict):
        session_state_payload = market_meta.get("session_state")
    if session_state_payload is None and isinstance(data_meta, dict):
        session_state_payload = data_meta.get("session_state")
    if isinstance(data_meta, dict):
        mode_token = str(data_meta.get("mode") or "").lower()
        if mode_token == "degraded" and planning_context_value != "frozen":
            planning_context_value = "degraded"
    if simulate_open and planning_context_value not in {"degraded"}:
        planning_context_value = "live"
    charts_container = first.get("charts") or {}
    charts = charts_container.get("params") if isinstance(charts_container, dict) else None
    chart_params_payload: Dict[str, Any] = charts if isinstance(charts, dict) else {}
    chart_url_value: Optional[str] = charts_container.get("interactive") if isinstance(charts_container, dict) else None

    # Ensure plan levels are embedded for /tv rendering even if upstream omitted them.
    entry_level = plan.get("entry")
    if isinstance(entry_level, dict):
        entry_level = entry_level.get("level")
    stop_level = plan.get("stop")
    if isinstance(stop_level, dict):
        stop_level = stop_level.get("level")
    targets_list = plan.get("targets") or []
    try:
        if entry_level is not None:
            chart_params_payload.setdefault("entry", f"{float(entry_level):.2f}")
    except (TypeError, ValueError):
        pass
    try:
        if stop_level is not None:
            chart_params_payload.setdefault("stop", f"{float(stop_level):.2f}")
    except (TypeError, ValueError):
        pass
    cleaned_targets: List[str] = []
    for target in targets_list:
        try:
            cleaned_targets.append(f"{float(target):.2f}")
        except (TypeError, ValueError):
            continue
    if cleaned_targets:
        chart_params_payload.setdefault("tp", ",".join(cleaned_targets))

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
    direction_for_levels = plan.get("direction") or direction_hint
    if entry_val is not None and stop_val is not None and targets_list:
        em_factor = float(getattr(settings, "ft_em_factor", 1.1))
        capped_targets, cap_used = _apply_em_cap(direction_for_levels, entry_val, targets_list, expected_move_abs, em_factor)
        if capped_targets and len(capped_targets) == len(targets_list):
            targets_list = [round(val, 2) for val in capped_targets]
            em_cap_used = cap_used
        else:
            targets_list = [round(float(val), 2) for val in targets_list]
        valid_levels, invariant_reason = _validate_level_invariants(direction_for_levels, entry_val, stop_val, targets_list)
        if not valid_levels:
            raise HTTPException(
                status_code=422,
                detail={"code": "INVARIANT_BROKEN", "message": invariant_reason or "invalid_levels"},
            )
    else:
        targets_list = [round(float(val), 2) for val in targets_list]

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
        chart_params_payload.setdefault("symbol", _tv_symbol(symbol))
        chart_params_payload.setdefault("range", _range_for_style(first.get("style")))
        chart_params_payload.setdefault("interval", chart_timeframe_hint)
        chart_params_payload.setdefault("focus", "plan")
        chart_params_payload.setdefault("center_time", "latest")
        chart_params_payload.setdefault("scale_plan", "auto")
        interval_token = chart_params_payload.get("interval") or plan.get("interval") or first.get("style_interval")
        try:
            chart_params_payload["interval"] = normalize_interval(interval_token or "5m")
        except ValueError:
            chart_params_payload["interval"] = "5m"
        if entry_val is not None or stop_val is not None or targets_list:
            default_note = _format_chart_note(symbol, first.get("style"), entry_val, stop_val, targets_list)
            if default_note and not chart_params_payload.get("notes"):
                chart_params_payload["notes"] = default_note
        _attach_market_chart_params(chart_params_payload, market_meta, data_meta)
        try:
            chart_model = ChartParams(**chart_params_payload)
            chart_links = await gpt_chart_url(chart_model, request)
            chart_url_value = chart_links.interactive
            try:
                chart_params_payload["interval"] = normalize_interval(chart_params_payload.get("interval") or chart_model.interval)
            except Exception:
                pass
        except HTTPException as exc:
            logger.debug("plan chart link validation failed for %s: %s", symbol, exc)
            chart_url_value = None
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("plan chart link error for %s: %s", symbol, exc)
            chart_url_value = None

    live_stamp_payload: Optional[str] = None
    charts_payload: Dict[str, Any] = {}
    if chart_params_payload:
        if is_plan_live:
            live_stamp_payload = chart_params_payload.get("last_update")
            if not live_stamp_payload:
                live_stamp_payload = datetime.now(timezone.utc).isoformat()
                chart_params_payload["last_update"] = live_stamp_payload
            chart_params_payload["live"] = "1"
        else:
            chart_params_payload.pop("live", None)
        charts_payload["params"] = chart_params_payload
        charts_payload["timeframe"] = chart_timeframe_hint
        charts_payload["guidance"] = hint_guidance
        if simulated_banner_text:
            charts_payload.setdefault("banners", []).append(simulated_banner_text)
    if chart_url_value:
        chart_url_value = _append_query_params(
            chart_url_value,
            {
                "plan_id": plan_id,
                "plan_version": str(version),
            },
        )
        if is_plan_live:
            live_param_stamp = live_stamp_payload or chart_params_payload.get("last_update") if chart_params_payload else None
            if not live_param_stamp:
                live_param_stamp = datetime.now(timezone.utc).isoformat()
            chart_url_value = _append_query_params(chart_url_value, {"live": "1", "last_update": live_param_stamp})
        charts_payload["interactive"] = chart_url_value
    elif chart_params_payload and {"direction", "entry", "stop", "tp"}.issubset(chart_params_payload.keys()):
        fallback_chart_url = _build_tv_chart_url(request, chart_params_payload)
        fallback_chart_url = _append_query_params(
            fallback_chart_url,
            {
                "plan_id": plan_id,
                "plan_version": str(version),
            },
        )
        if is_plan_live:
            live_param_stamp = chart_params_payload.get("last_update") or datetime.now(timezone.utc).isoformat()
            chart_params_payload["last_update"] = live_param_stamp
            chart_params_payload["live"] = "1"
            fallback_chart_url = _append_query_params(
                fallback_chart_url,
                {"live": "1", "last_update": live_param_stamp},
            )
        charts_payload["interactive"] = fallback_chart_url
        chart_url_value = fallback_chart_url
        logger.debug(
            "plan chart fallback used",
            extra={"symbol": symbol, "plan_id": plan_id, "url": fallback_chart_url},
        )

    if not chart_url_value:
        minimal_params = {
            "symbol": symbol,
            "interval": chart_timeframe_hint or normalize_interval(plan.get("interval") or chart_params_payload.get("interval") or "15"),
            "plan_id": plan_id,
            "plan_version": str(version),
        }
        if entry_val is not None:
            minimal_params["entry"] = f"{entry_val:.2f}"
        if stop_val is not None:
            minimal_params["stop"] = f"{stop_val:.2f}"
        if targets_list:
            minimal_params["tp"] = ",".join(f"{target:.2f}" for target in targets_list)
        minimal_params.setdefault("focus", "plan")
        minimal_params.setdefault("center_time", "latest")
        minimal_params.setdefault("scale_plan", "auto")
        if is_plan_live:
            minimal_params["live"] = "1"
            minimal_params["last_update"] = datetime.now(timezone.utc).isoformat()
        chart_url_value = _build_tv_chart_url(request, minimal_params)
        charts_payload.setdefault("params", minimal_params)
        charts_payload["interactive"] = chart_url_value

    trade_detail_url = chart_url_value
    plan["trade_detail"] = trade_detail_url

    atr_val = _safe_number(indicators.get("atr14"))
    precision_for_levels = get_precision(symbol)
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
    calc_notes_output = calc_notes or None
    if calc_notes_output is not None and not calc_notes_output:
        calc_notes_output = None
    targets_for_rules = list(targets_list)
    risk_block = _build_risk_block(
        entry=entry_val,
        stop=stop_val,
        targets=targets_for_rules,
        atr=calc_notes.get("atr14") if calc_notes else None,
        stop_multiple=calc_notes.get("stop_multiple") if calc_notes else None,
        expected_move=plan.get("expected_move") or (snapshot.get("volatility") or {}).get("expected_move_horizon"),
        runner=plan.get("runner") if plan else None,
    )
    execution_rules = _build_execution_rules(
        entry=entry_val,
        stop=stop_val,
        targets=targets_for_rules,
        direction=plan.get("direction") or direction_hint,
        precision=precision_for_levels,
        key_levels_used=None,  # populated later once matches resolved
        runner=plan.get("runner") if plan else None,
    )
    mtf_confluence_tags: List[str] = []
    try:
        mtf_confluence_tags = await _compute_multi_timeframe_confluence(
            symbol=symbol,
            base_interval=chart_timeframe_hint or "5m",
            snapshot=snapshot,
            direction=plan.get("direction") or direction_hint,
            entry=entry_val,
            overlays=first.get("context_overlays"),
        )
    except Exception:  # pragma: no cover - defensive
        mtf_confluence_tags = []
    # Infer snapped_targets by comparing target prices to named levels (key_levels + overlays)
    snapped_names: List[str] = []
    key_level_matches: List[Dict[str, Any]] = []
    try:
        targets_for_snap = list(targets_list)
        atr_for_window = float(indicators.get("atr14") or 0.0)
        baseline_price = entry_val if entry_val is not None else (targets_for_snap[0] if targets_for_snap else None)
        fallback_window = 0.1
        if baseline_price is not None:
            try:
                fallback_window = max(abs(float(baseline_price)) * 0.002, 0.1)
            except (TypeError, ValueError):
                fallback_window = 0.1
        window = atr_for_window * 0.30 if atr_for_window and atr_for_window > 0 else fallback_window
        window = max(window, fallback_window)
        levels_dict = first.get("key_levels") or {}
        overlays = first.get("context_overlays") or {}
        level_candidates: List[Dict[str, Any]] = []
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
                    price_val = float(v)
                except Exception:
                    continue
                label = alias.get(k, k)
                level_candidates.append(
                    {
                        "label": label,
                        "price": price_val,
                        "category": _classify_level(label),
                        "source": "key_levels",
                    }
                )
        vp = overlays.get("volume_profile") or {}
        for lab, key in [("VAH", "vah"), ("VAL", "val"), ("POC", "poc"), ("VWAP", "vwap")]:
            val = vp.get(key)
            if isinstance(val, (int, float)):
                try:
                    price_val = float(val)
                except Exception:
                    continue
                level_candidates.append(
                    {
                        "label": lab,
                        "price": price_val,
                        "category": _classify_level(lab),
                        "source": "volume_profile",
                    }
                )
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
                    price_val = float(val)
                except Exception:
                    continue
                level_candidates.append(
                    {
                        "label": lab,
                        "price": price_val,
                        "category": _classify_level(lab),
                        "source": "anchored_vwap",
                    }
                )

        def _nearest_level(price: float) -> Dict[str, Any] | None:
            best_candidate: Dict[str, Any] | None = None
            best_distance: float | None = None
            for candidate in level_candidates:
                distance = abs(candidate["price"] - price)
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_candidate = candidate
            if best_candidate is None or best_distance is None:
                return None
            if not math.isfinite(best_distance) or best_distance > window:
                return None
            return {
                "label": best_candidate["label"],
                "price": best_candidate["price"],
                "category": best_candidate.get("category", "structural"),
                "source": best_candidate.get("source"),
                "distance": best_distance,
            }

        for tp in targets_for_snap[:2]:
            try:
                tp_f = float(tp)
            except Exception:
                continue
            match = _nearest_level(tp_f)
            if match:
                snapped_names.append(match["label"])

        def _record_match(role: str, raw_price: float | None) -> None:
            if raw_price is None:
                return
            try:
                price_val = float(raw_price)
            except (TypeError, ValueError):
                return
            match = _nearest_level(price_val)
            if not match:
                return
            entry = dict(match)
            entry["role"] = role
            key_level_matches.append(entry)

        _record_match("entry", entry_val)
        _record_match("stop", stop_val)
        for idx, target_value in enumerate(targets_for_snap, start=1):
            _record_match(f"tp{idx}", target_value)
    except Exception:
        snapped_names = []
        key_level_matches = []

    htf = {
        "bias": ((snapshot.get("trend") or {}).get("ema_stack") or "unknown"),
        "snapped_targets": snapped_names,
    }
    data_quality = {
        "series_present": True,
        "iv_present": True,
        "events_present": False,
        "earnings_present": False,
    }
    raw_warnings = first.get("warnings") or plan.get("warnings") or []
    if isinstance(raw_warnings, list):
        plan_warnings: List[str] = list(raw_warnings)
    elif raw_warnings:
        plan_warnings = [str(raw_warnings)]
    else:
        plan_warnings = []
    plan_warnings = [w for w in plan_warnings if "watch plan" not in str(w).lower()]
    if event_warnings:
        plan_warnings.extend(event_warnings)
    plan_warnings = [str(w) for w in plan_warnings if w]
    plan_warnings = list(dict.fromkeys(plan_warnings))

    entry_for_engine = entry_val
    if entry_for_engine is None:
        try:
            entry_for_engine = float(plan.get("entry")) if plan.get("entry") is not None else None
        except Exception:
            entry_for_engine = None
    stop_for_engine = stop_val
    if stop_for_engine is None:
        try:
            stop_for_engine = float(plan.get("stop")) if plan.get("stop") is not None else None
        except Exception:
            stop_for_engine = None
    targets_for_engine = list(targets_list)
    target_profile = None
    target_profile_dict: Dict[str, Any] | None = None
    structured_plan_payload: Dict[str, Any] | None = None
    if (
        entry_for_engine is not None
        and stop_for_engine is not None
        and targets_for_engine
    ):
        try:
            target_profile = build_target_profile(
                entry=float(entry_for_engine),
                stop=float(stop_for_engine),
                targets=targets_for_engine,
                target_meta=plan.get("target_meta"),
                debug=plan.get("debug") or first.get("debug"),
                runner=plan.get("runner"),
                warnings=plan_warnings,
                atr_used=plan.get("atr"),
                expected_move=plan.get("expected_move"),
                style=style_token,
                bias=plan.get("direction") or direction_hint,
            )
            target_profile_dict = target_profile.to_dict()
            if em_cap_used:
                target_profile_dict["em_used"] = True
            structured_plan_payload = build_structured_plan(
                plan_id=plan_id,
                symbol=symbol,
                style=style_token,
                direction=plan.get("direction") or direction_hint,
                profile=target_profile,
                confidence=plan.get("confidence"),
                rationale=(plan.get("notes") or "").strip() or None,
                options=first.get("options"),
                chart_url=chart_url_value,
                session=session_state_payload,
                confluence=snapped_names or None,
            )
        except Exception as exc:
            logger.debug("structured plan build failed for %s: %s", symbol, exc)
            target_profile_dict = None
            structured_plan_payload = None
    if structured_plan_payload is not None and simulated_banner_text:
        banners_list = structured_plan_payload.get("banners") if isinstance(structured_plan_payload.get("banners"), list) else None
        if banners_list is None:
            banners_list = []
            structured_plan_payload["banners"] = banners_list
        if simulated_banner_text not in banners_list:
            banners_list.append(simulated_banner_text)
    if (
        isinstance(market_meta_context, dict)
        and market_meta_context.get("status") != "open"
        and not simulate_open
    ):
        planning_context_value = "frozen"

    tp_meta_source: Sequence[Mapping[str, Any]] | None = None
    if target_profile is not None:
        tp_meta_source = target_profile.meta
    elif isinstance(plan.get("target_meta"), list):
        tp_meta_source = plan.get("target_meta")  # type: ignore[assignment]
    tp_reasons = _tp_reason_entries(tp_meta_source)

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
            options_payload = await gpt_contracts(contract_request, user)
            first["options"] = options_payload
        except HTTPException as exc:
            logger.info("contract lookup skipped for %s: %s", symbol, exc)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("contract lookup error for %s: %s", symbol, exc)

    options_contracts: List[Dict[str, Any]] | None = None
    options_note: str | None = None
    if include_options_contracts:
        extracted_contracts = _extract_options_contracts(options_payload)
        filtered_contracts, guardrail_rejections = _apply_option_guardrails(
            extracted_contracts,
            max_spread_pct=float(getattr(settings, "ft_max_spread_pct", 8.0)),
            min_open_interest=int(getattr(settings, "ft_min_oi", 300)),
        )
        if guardrail_rejections:
            rejected_contracts.extend(guardrail_rejections)
        if filtered_contracts:
            options_contracts = filtered_contracts
        else:
            options_contracts = []
            if guardrail_rejections and not options_note:
                options_note = "Contracts filtered by liquidity guardrails"
            elif isinstance(options_payload, dict) and options_payload.get("quotes_notice"):
                options_note = str(options_payload["quotes_notice"])
            elif not side_hint:
                options_note = "Options side unavailable for this plan"
            else:
                options_note = "No tradeable contracts met filters"

    price_close = snapshot.get("price", {}).get("close")
    decimals_value = 2
    if isinstance(price_close, (int, float)):
        try:
            scale = _price_scale_for(float(price_close))
            if scale > 0:
                decimals_value = int(round(math.log10(scale)))
        except Exception:
            decimals_value = 2

    precision_for_levels = decimals_value if isinstance(decimals_value, int) else 2
    key_levels_used: Dict[str, List[Dict[str, Any]]] | None = None
    if key_level_matches:
        formatted: Dict[str, List[Dict[str, Any]]] = {"session": [], "structural": []}
        seen_pairs: set[tuple[str | None, str | None]] = set()
        distance_precision = max(precision_for_levels + 2, 4)
        for match in key_level_matches:
            role = match.get("role")
            label = match.get("label")
            pair = (role, label)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            try:
                price_val = float(match.get("price"))
            except (TypeError, ValueError):
                continue
            try:
                distance_val = float(match.get("distance"))
            except (TypeError, ValueError):
                distance_val = 0.0
            bucket = "session" if str(match.get("category" or "")).lower() == "session" else "structural"
            entry_payload: Dict[str, Any] = {
                "role": role,
                "label": label,
                "price": round(price_val, precision_for_levels),
                "distance": round(abs(distance_val), distance_precision),
            }
            source = match.get("source")
            if source:
                entry_payload["source"] = source
            formatted[bucket].append(entry_payload)
        key_levels_used = {bucket: items for bucket, items in formatted.items() if items}

    execution_rules = _build_execution_rules(
        entry=entry_val,
        stop=stop_val,
        targets=targets_for_rules,
        direction=plan.get("direction") or direction_hint,
        precision=precision_for_levels,
        key_levels_used=key_levels_used,
        runner=plan.get("runner") if plan else None,
    )

    plan_core = _extract_plan_core(first, plan_id, version, decimals_value)
    if plan_core.get("setup") in {"watch_plan", "offline"}:
        inferred_setup = first.get("strategy_id") or plan.get("setup")
        plan_core["setup"] = inferred_setup
        plan["setup"] = inferred_setup
    else:
        plan_core.setdefault("setup", first.get("strategy_id"))
        plan.setdefault("setup", plan_core.get("setup"))
    plan_core["trade_detail"] = trade_detail_url
    plan_core["chart_timeframe"] = chart_timeframe_hint
    plan_core["chart_guidance"] = hint_guidance
    if updated_from_version:
        plan_core["updated_from_version"] = updated_from_version
    if update_reason:
        plan_core["update_reason"] = update_reason
    if plan_warnings:
        plan_core["warnings"] = plan_warnings
    if structured_plan_payload:
        plan_core["structured_plan"] = structured_plan_payload
    if target_profile_dict:
        plan_core["target_profile"] = target_profile_dict
    if session_state_payload:
        plan_core.setdefault("session_state", session_state_payload)
    if key_levels_used:
        plan_core["key_levels_used"] = key_levels_used
        plan.setdefault("key_levels_used", key_levels_used)
    if risk_block:
        plan_core["risk_block"] = risk_block
        plan.setdefault("risk_block", risk_block)
    if execution_rules:
        plan_core["execution_rules"] = execution_rules
        plan.setdefault("execution_rules", execution_rules)
    if mtf_confluence_tags:
        plan_core["mtf_confluence"] = mtf_confluence_tags
        plan.setdefault("mtf_confluence", mtf_confluence_tags)
    plan_layers: Dict[str, Any] | None = None
    if include_plan_layers:
        overlays_payload = first.get("context_overlays")
        interval_token = None
        if chart_params_payload and chart_params_payload.get("interval"):
            interval_token = chart_params_payload.get("interval")
        interval_token = interval_token or chart_timeframe_hint or "5m"
        plan_layers = build_plan_layers(
            symbol=symbol,
            interval=str(interval_token),
            as_of=session_payload.get("as_of"),
            planning_context=planning_context_value,
            key_levels=first.get("key_levels"),
            overlays=overlays_payload,
        )
        if key_levels_used:
            _ensure_plan_layers_cover_used_levels(plan_layers, key_levels_used, precision=precision_for_levels)
        plan_layers["plan_id"] = plan_id
        plan_core["plan_layers"] = plan_layers
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
        "plan_layers": plan_layers,
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
            "sim_open": simulate_open,
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
    confidence_visual_value = _confidence_visual(confidence_output)
    if confidence_visual_value:
        if plan:
            plan["confidence_visual"] = confidence_visual_value
        if structured_plan_payload:
            structured_plan_payload["confidence_visual"] = confidence_visual_value
    notes_output = (plan.get("notes") or "").strip() if plan else ""
    if plan and "watch plan" in notes_output.lower():
        plan["notes"] = None
        notes_output = ""
    elif "watch plan" in notes_output.lower():
        notes_output = ""
    bias_output = plan.get("direction") or ((snapshot.get("trend") or {}).get("direction_hint"))
    relevant_levels = first.get("key_levels") or {}
    expected_move_basis = None
    if isinstance(expected_move_abs, (int, float)) and math.isfinite(expected_move_abs) and expected_move_abs > 0:
        price_ref = entry_val if isinstance(entry_val, (int, float)) else (snapshot.get("price", {}).get("close") if isinstance(snapshot.get("price", {}), Mapping) else None)
        try:
            pct = (expected_move_abs / float(price_ref)) * 100 if price_ref else None
        except (TypeError, ValueError):
            pct = None
        if pct is not None:
            expected_move_basis = f"EM ≈ {expected_move_abs:.2f} ({pct:.1f}%)"
        else:
            expected_move_basis = f"EM ≈ {expected_move_abs:.2f}"
    sentiment_block = first.get("sentiment")
    events_block = first.get("events")
    earnings_block = first.get("earnings")
    events_source = "payload" if events_block else None
    earnings_source = "payload" if earnings_block else None
    if not events_block or not earnings_block:
        try:
            enrichment = await _fetch_context_enrichment(symbol)
        except Exception as exc:
            logger.debug("enrichment fetch failed for %s: %s", symbol, exc)
            enrichment = None
        if not events_block:
            enriched_events = (enrichment or {}).get("events")
            if enriched_events:
                events_block = enriched_events
                events_source = events_source or "enrichment"
        if not earnings_block:
            enriched_earnings = (enrichment or {}).get("earnings")
            if enriched_earnings:
                earnings_block = enriched_earnings
                earnings_source = earnings_source or "enrichment"
    if not events_block:
        macro_events = _macro_event_block(symbol, session_state_payload)
        if macro_events:
            events_block = macro_events
            events_source = events_source or "macro_window"
    data_quality["events_present"] = bool(events_block)
    data_quality["earnings_present"] = bool(earnings_block)
    confidence_factors: List[str] | None = None
    feature_block = first.get("features") or {}
    for key in ("plan_confidence_factors", "plan_confidence_reasons", "confidence_reasons"):
        raw = feature_block.get(key)
        if isinstance(raw, (list, tuple)):
            confidence_factors = [str(item) for item in raw if item]
            break
    confidence_values = list(confidence_factors or [])
    confluence_tags = _unique_tags(confidence_values + list(snapped_names))
    if plan_layers is not None:
        meta = plan_layers.setdefault("meta", {})
        if not isinstance(meta.get("level_groups"), dict):
            meta["level_groups"] = {"primary": [], "supplemental": []}
        if not confidence_values:
            confidence_values = list(meta.get("confidence_factors") or [])
        meta["confidence_factors"] = confidence_values
        if not confluence_tags:
            existing_tags = meta.get("confluence") or []
            confluence_tags = _unique_tags(confidence_values + list(snapped_names) + list(existing_tags))
        meta["confluence"] = confluence_tags if confluence_tags else list(meta.get("confluence") or [])
        if feature_block:
            meta["features"] = {
                key: value
                for key, value in feature_block.items()
                if isinstance(value, (bool, int, float, str))
            }
        else:
            meta.setdefault("features", {})
        if isinstance(options_payload, dict) and options_payload.get("quotes_notice"):
            meta["quotes_notice"] = options_payload["quotes_notice"]
        if mtf_confluence_tags:
            meta["mtf_confluence"] = mtf_confluence_tags
    if structured_plan_payload is not None:
        if confluence_tags:
            structured_plan_payload["confluence"] = confluence_tags
            structured_plan_payload["confluence_tags"] = confluence_tags
        if tp_reasons:
            structured_plan_payload["tp_reasons"] = tp_reasons
    if include_options_contracts:
        if options_contracts is not None:
            structured_plan_payload["options_contracts"] = options_contracts
        if options_note:
            structured_plan_payload["options_note"] = options_note
    if rejected_contracts:
        structured_plan_payload["rejected_contracts"] = rejected_contracts
        if key_levels_used:
            structured_plan_payload["key_levels_used"] = key_levels_used
        if risk_block:
            structured_plan_payload["risk_block"] = risk_block
        if execution_rules:
            structured_plan_payload["execution_rules"] = execution_rules
        if mtf_confluence_tags:
            structured_plan_payload["mtf_confluence"] = mtf_confluence_tags
    if confluence_tags:
        plan_core["confluence_tags"] = confluence_tags
        plan.setdefault("confluence_tags", confluence_tags)
    if tp_reasons:
        plan_core["tp_reasons"] = tp_reasons
        plan.setdefault("tp_reasons", tp_reasons)
    if include_options_contracts:
        if options_contracts is not None:
            plan_core["options_contracts"] = options_contracts
            plan.setdefault("options_contracts", options_contracts)
        if options_note:
            plan_core["options_note"] = options_note
            plan.setdefault("options_note", options_note)
    if rejected_contracts:
        plan_core["rejected_contracts"] = rejected_contracts
        plan.setdefault("rejected_contracts", rejected_contracts)
    calc_notes_output = calc_notes or None
    if calc_notes_output is not None and not calc_notes_output:
        calc_notes_output = None
    charts_field = charts_payload or None
    charts_params_output = chart_params_payload or None
    chart_url_output = chart_url_value or None
    if structured_plan_payload and confidence_visual_value:
        structured_plan_payload["confidence_visual"] = confidence_visual_value
    market_meta = market_meta_context if isinstance(market_meta_context, dict) else None
    data_meta = data_meta_context if isinstance(data_meta_context, dict) else None
    if market_meta is None or data_meta is None:
        fallback_market, fallback_data, _, _ = _market_snapshot_payload(session_payload)
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

    is_live_plan = planning_context_value == "live"
    if chart_params_payload:
        if is_live_plan:
            live_stamp = chart_params_payload.get("last_update")
            if not live_stamp:
                live_stamp = datetime.now(timezone.utc).isoformat()
                chart_params_payload["last_update"] = live_stamp
            chart_params_payload["live"] = "1"
        else:
            chart_params_payload.pop("live", None)
            chart_params_payload.pop("last_update", None)
    if chart_url_value and is_live_plan:
        live_stamp = chart_params_payload.get("last_update") if chart_params_payload else None
        extra_params = {"live": "1"}
        if live_stamp:
            extra_params["last_update"] = live_stamp
        chart_url_value = _append_query_params(chart_url_value, extra_params)
        if charts_field is not None:
            charts_field["interactive"] = chart_url_value
    if charts_field is not None:
        if chart_params_payload:
            charts_field["params"] = chart_params_payload
        charts_field["live"] = is_live_plan
    if data_meta is not None and isinstance(data_meta, dict) and is_live_plan:
        bars_url = data_meta.get("bars")
        if isinstance(bars_url, str):
            data_meta["bars"] = _append_query_params(bars_url, {"live": "1"})

    if isinstance(data_meta, dict):
        data_meta.setdefault("events_present", bool(events_block))
        data_meta.setdefault("earnings_present", bool(earnings_block))

    logger.info(
        "plan_enrichment_status",
        extra={
            "symbol": symbol,
            "events_present": bool(events_block),
            "earnings_present": bool(earnings_block),
            "events_source": events_source or "none",
            "earnings_source": earnings_source or "none",
        },
    )

    metric_count = _record_metric(
        "gpt_plan",
        session=str(session_payload.get("status") or "unknown"),
        context=planning_context_value,
    )
    logger.info(
        "plan response ready",
        extra={
            "symbol": symbol,
            "plan_id": plan_id,
            "version": version,
            "planning_context": planning_context_value,
            "trade_detail": trade_detail_url,
            "session_status": session_payload.get("status"),
            "session_as_of": session_payload.get("as_of"),
            "metric_count": metric_count,
            "sim_open": simulate_open,
        },
    )

    response_meta: Dict[str, Any] | None = None
    if simulate_open:
        response_meta = {"simulated_open": True}
        if simulated_banner_text:
            response_meta["banner"] = simulated_banner_text

    warnings_payload = list(dict.fromkeys(plan_warnings)) if plan_warnings else []
    confluence_payload = list(confluence_tags) if confluence_tags else []
    mtf_confluence_payload = list(mtf_confluence_tags) if mtf_confluence_tags else []
    tp_reasons_payload = list(tp_reasons) if tp_reasons else []
    options_contracts_payload: List[Dict[str, Any]] = (
        list(options_contracts) if include_options_contracts and options_contracts else []
    )
    plan_response = PlanResponse(
        plan_id=plan_id,
        version=version,
        trade_detail=trade_detail_url,
        warnings=warnings_payload,
        planning_context=planning_context_value,
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
        confluence_tags=confluence_payload,
        confluence=mtf_confluence_payload,
        notes=notes_output,
        relevant_levels=relevant_levels or None,
        expected_move_basis=expected_move_basis,
        sentiment=sentiment_block,
        events=events_block,
        earnings=earnings_block,
        charts_params=charts_params_output,
        chart_url=chart_url_output,
        chart_timeframe=chart_timeframe_hint,
        chart_guidance=hint_guidance,
        strategy_id=first.get("strategy_id"),
        description=first.get("description"),
        score=first.get("score"),
        plan=first.get("plan"),
        structured_plan=structured_plan_payload,
        target_profile=target_profile_dict,
        charts=charts_field,
        key_levels=first.get("key_levels"),
        key_levels_used=key_levels_used or None,
        market_snapshot=first.get("market_snapshot"),
        features=first.get("features"),
        options=first.get("options"),
        options_contracts=options_contracts_payload,
        options_note=options_note if include_options_contracts else None,
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
        risk_block=risk_block,
        execution_rules=execution_rules,
        session_state=session_state_payload,
        confidence_visual=confidence_visual_value,
        plan_layers=plan_layers or {},
        tp_reasons=tp_reasons_payload,
        meta=response_meta,
        rejected_contracts=rejected_contracts,
    )
    plan_response.phase = "hydrate"
    plan_response.layers_fetched = bool(plan_response.plan_layers)
    meta_payload = dict(plan_response.meta or {})
    meta_payload["within_event_window"] = within_event_window
    if minutes_to_event is not None:
        meta_payload["minutes_to_event"] = minutes_to_event
    if em_cap_used:
        meta_payload["em_used"] = True
    plan_response.meta = meta_payload or None
    return _finalize_plan_response(plan_response)


@gpt.post(
    "/api/v1/assistant/exec",
    summary="Structured execution payload for assistant clients",
    response_model=AssistantExecResponse,
)
async def assistant_exec(
    request_payload: AssistantExecRequest,
    request: Request,
    user: AuthedUser = Depends(require_api_key),
) -> AssistantExecResponse:
    plan_request = PlanRequest(
        symbol=request_payload.symbol,
        style=request_payload.style,
        plan_id=request_payload.plan_id,
    )
    plan_response = await gpt_plan(plan_request, request, Response(), user)

    plan_block = plan_response.structured_plan or {}
    if not plan_block:
        fallback_plan = plan_response.plan or {}
        plan_block = dict(fallback_plan)
    if not plan_block:
        raise HTTPException(status_code=502, detail="Plan data unavailable")
    plan_block.setdefault("plan_id", plan_response.plan_id)
    plan_block.setdefault("version", plan_response.version)
    plan_block.setdefault("symbol", plan_response.symbol)
    plan_block.setdefault("style", plan_response.style)
    if plan_response.chart_timeframe and "chart_timeframe" not in plan_block:
        plan_block["chart_timeframe"] = plan_response.chart_timeframe
    if plan_response.chart_guidance and "chart_guidance" not in plan_block:
        plan_block["chart_guidance"] = plan_response.chart_guidance
    if plan_response.confidence_visual and "confidence_visual" not in plan_block:
        plan_block["confidence_visual"] = plan_response.confidence_visual
    chart_block = {
        "interactive": plan_response.chart_url,
        "params": plan_response.charts_params,
    }

    options_block = plan_response.options or {}
    if not options_block and plan_response.options_contracts:
        options_block = {"best": plan_response.options_contracts, "source": "plan_snapshot"}
    if not options_block or not options_block.get("best"):
        example = None
        try:
            as_of_hint = None
            if plan_response.session_state and plan_response.session_state.get("as_of"):
                as_of_hint = plan_response.session_state.get("as_of")
            example = await best_contract_example(
                plan_response.symbol or plan_request.symbol.upper(),
                plan_response.style or plan_request.style,
                as_of_hint,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("options example lookup failed for %s: %s", plan_request.symbol, exc)
        if example:
            options_block = {
                "best": [example],
                "source": "tradier",
            }

    context_block = {
        "events": plan_response.events,
        "earnings": plan_response.earnings,
        "session": plan_response.session_state
        or (plan_response.market or {}).get("session_state"),
        "market": plan_response.market_snapshot,
    }

    meta_block = {
        "plan_id": plan_response.plan_id,
        "version": plan_response.version,
        "symbol": plan_response.symbol,
        "style": plan_response.style,
        "trade_detail": plan_response.trade_detail,
    }
    if plan_response.confluence_tags:
        meta_block["confluence_tags"] = plan_response.confluence_tags
    if plan_response.tp_reasons:
        meta_block["tp_reasons"] = plan_response.tp_reasons
    if plan_response.options_contracts:
        meta_block["options_contracts"] = plan_response.options_contracts
    if plan_response.options_note:
        meta_block["options_note"] = plan_response.options_note
    if plan_response.confluence:
        meta_block["mtf_confluence"] = plan_response.confluence
    if plan_response.key_levels_used:
        meta_block["key_levels_used"] = plan_response.key_levels_used
    if plan_response.risk_block:
        meta_block["risk_block"] = plan_response.risk_block
    if plan_response.execution_rules:
        meta_block["execution_rules"] = plan_response.execution_rules

    return AssistantExecResponse(
        plan=plan_block,
        chart=chart_block,
        options=options_block or None,
        context=context_block,
        meta=meta_block,
    )


@gpt.get(
    "/api/v1/symbol/{symbol}/diagnostics",
    summary="Lightweight diagnostics for a symbol",
    response_model=SymbolDiagnosticsResponse,
)
async def symbol_diagnostics(
    symbol: str,
    request: Request,
    interval: str = Query("5"),
    lookback: int = Query(300, ge=100, le=1000),
) -> SymbolDiagnosticsResponse:
    normalized_interval = normalize_interval(interval)
    context = await _build_interval_context(symbol.upper(), normalized_interval, lookback)
    session_payload = _session_payload_from_request(request)
    return SymbolDiagnosticsResponse(
        symbol=symbol.upper(),
        interval=normalized_interval,
        key_levels=context.get("key_levels") or {},
        snapshot=context.get("snapshot") or {},
        indicators=context.get("indicators") or {},
        session=session_payload,
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
    plan_payload.pop("idea_url", None)
    plan_payload.setdefault("trade_detail", trade_detail_url)
    snapshot["plan"] = plan_payload
    snapshot.setdefault("chart_url", None)
    await _store_idea_snapshot(plan_id, snapshot)
    return IdeaStoreResponse(plan_id=plan_id, trade_detail=trade_detail_url)


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
    response = await gpt_plan(plan_request, request, Response(), user)
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
    await _ensure_stream_heartbeat(uppercase)
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


@app.websocket("/ws/plans/{plan_id}")
async def stream_plan_ws(websocket: WebSocket, plan_id: str) -> None:
    plan_token = (plan_id or "").strip()
    if not plan_token:
        await websocket.close(code=1008, reason="plan_id required")
        return
    await websocket.accept()
    queue: asyncio.Queue[str] = asyncio.Queue(maxsize=100)
    async with _STREAM_LOCK:
        _PLAN_STREAM_SUBSCRIBERS.setdefault(plan_token, []).append(queue)
    try:
        try:
            snapshot = await _get_idea_snapshot(plan_token)
            await _LIVE_PLAN_ENGINE.register_snapshot(snapshot)
            initial_event = json.dumps({"plan_id": plan_token, "event": {"t": "plan_full", "payload": snapshot}})
            await websocket.send_text(initial_event)
        except HTTPException as exc:
            await websocket.send_text(
                json.dumps(
                    {
                        "plan_id": plan_token,
                        "event": {"t": "error", "status": exc.status_code, "detail": exc.detail},
                    }
                )
            )
        except Exception as exc:  # pragma: no cover - defensive
            await websocket.send_text(
                json.dumps({"plan_id": plan_token, "event": {"t": "error", "detail": str(exc)}})
            )
        while True:
            payload = await queue.get()
            await websocket.send_text(payload)
    except WebSocketDisconnect:
        return
    finally:
        async with _STREAM_LOCK:
            subscribers = _PLAN_STREAM_SUBSCRIBERS.get(plan_token, [])
            if queue in subscribers:
                subscribers.remove(queue)
            if not subscribers:
                _PLAN_STREAM_SUBSCRIBERS.pop(plan_token, None)
        try:
            await websocket.close()
        except RuntimeError:
            pass


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
        prefer_delta = PREFER_DELTA_BY_STYLE.get(style, 0.5)
        try:
            spread_normalized = (
                float(contract["spread_pct"]) / 100.0 if contract.get("spread_pct") is not None else None
            )
            composite = score_contract(
                {
                    "spread_pct": spread_normalized,
                    "bid": contract.get("bid"),
                    "ask": contract.get("ask"),
                    "delta": contract.get("delta"),
                    "volume": contract.get("volume"),
                    "open_interest": contract.get("oi"),
                    "iv_percentile": row.get("iv_percentile"),
                },
                prefer_delta=prefer_delta,
            )
            contract["liquidity_score"] = round(float(composite.score), 4)
            contract["liquidity_components"] = {
                key: round(float(value), 4) for key, value in composite.components.items()
            }
        except Exception:
            pass
        candidates.append(contract)

    candidates.sort(
        key=lambda item: (
            item.get("liquidity_score") or 0.0,
            item.get("tradeability") or 0.0,
        ),
        reverse=True,
    )
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
    quotes, quotes_meta = await fetch_option_quotes(option_symbols)

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
                "liquidity_score": row.get("tradeability") or row.get("liquidity_score"),
            })
        except Exception:
            continue

    response_payload = {
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
    if quotes_meta.get("notice"):
        notice = quotes_meta.get("notice")
        if notice == "sandbox_quotes_disabled":
            response_payload["quotes_notice"] = "Sandbox quotes unavailable"
        else:
            response_payload["quotes_notice"] = str(notice)
    response_payload["quotes_mode"] = quotes_meta.get("mode") or "unknown"
    return response_payload


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
    data: Dict[str, Any] = payload.model_dump(mode="python", exclude_none=True)
    raw_symbol = str(data.get("symbol") or "").strip()
    raw_interval = str(data.get("interval") or "").strip()
    direction = str(data.get("direction") or "").strip().lower()
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

    settings = get_settings()
    session_snapshot = _session_payload_from_request(request)
    canonical_flag = bool(getattr(settings, "ff_chart_canonical_v1", False))
    public_base = (getattr(settings, "public_base_url", None) or "").strip()
    origin = public_base or _resolved_base_url(request)

    symbol_token = _normalize_chart_symbol(raw_symbol)
    data["symbol"] = symbol_token
    data["interval"] = interval_norm
    data["direction"] = direction
    data["entry"] = f"{entry_f:.2f}"
    data["stop"] = f"{stop_f:.2f}"
    data["tp"] = str(tp_csv)

    metric_count = _record_metric(
        "gpt_chart_url",
        session=str(session_snapshot.get("status") or "unknown"),
    )

    if canonical_flag:
        tv_base = f"{(public_base or origin).rstrip('/')}/tv"
        canonical_payload: Dict[str, object] = {
            "symbol": symbol_token,
            "interval": interval_norm,
            "direction": direction,
            "entry": entry_f,
            "stop": stop_f,
        }

        if tp_csv:
            try:
                canonical_payload["tp"] = [float(chunk) for chunk in str(tp_csv).split(",") if chunk]
            except ValueError:
                canonical_payload["tp"] = tp_csv
        ema_value = data.get("ema")
        if ema_value:
            if isinstance(ema_value, (list, tuple)):
                canonical_payload["ema"] = ema_value
            else:
                canonical_payload["ema"] = [token.strip() for token in str(ema_value).split(",") if token.strip()]

        for optional_key in (
            "focus",
            "center_time",
            "scale_plan",
            "view",
            "range",
            "theme",
            "plan_id",
            "plan_version",
        ):
            if optional_key in data and data[optional_key] not in (None, ""):
                canonical_payload[optional_key] = data[optional_key]

        canonical_url = make_chart_url(
            canonical_payload,
            base_url=tv_base,
            precision_map=None,
        )
        _ensure_allowed_host(canonical_url, request)
        logger.info(
            "chart_url built",
            extra={
                "endpoint": "gpt_chart_url",
                "symbol": symbol_token,
                "interval": interval_norm,
                "session_status": session_snapshot.get("status"),
                "session_as_of": session_snapshot.get("as_of"),
                "metric_count": metric_count,
                "canonical": True,
            },
        )
        return ChartLinks(interactive=canonical_url)

    configured_base = (settings.chart_base_url or "").strip()
    base = (configured_base or f"{origin}/charts/html").rstrip("/")

    query: Dict[str, str] = {}
    for key, value in data.items():
        if key not in ALLOWED_CHART_KEYS or value is None:
            continue
        if isinstance(value, (list, tuple)):
            value = ",".join(str(item) for item in value if item is not None)
        query[key] = str(value)

    if "strategy" in query:
        query["strategy"] = quote(query["strategy"], safe="|;:,.+-_() ")
    if "levels" in data and data.get("levels"):
        query["levels"] = quote(str(data["levels"]), safe="|;:,.+-_() ")
    if "notes" in data and data.get("notes"):
        query["notes"] = quote(str(data["notes"])[:140], safe="|;:,.+-_() ")

    encoded = urlencode(query, doseq=False, safe=",", quote_via=quote)
    url = f"{base}?{encoded}" if encoded else base

    logger.info(
        "chart_url built",
        extra={
            "endpoint": "gpt_chart_url",
            "symbol": symbol_token,
            "interval": interval_norm,
            "session_status": session_snapshot.get("status"),
            "session_as_of": session_snapshot.get("as_of"),
            "metric_count": metric_count,
        "canonical": False,
    },
)
    _ensure_allowed_host(url, request)
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
        # Cross-source consistency: compare chain underlying vs live quote (0.2% threshold; 5s SLA)
        try:
            u = (polygon_bundle.get("underlying") or {}) if isinstance(polygon_bundle, dict) else {}
            usym = (u.get("symbol") or symbol).upper() if isinstance(u, dict) else symbol
            uprice = float(u.get("price")) if isinstance(u.get("price"), (int, float, str)) else None
            quote = await fetch_live_quote(usym)
            mismatch_pct = None
            if (uprice is not None) and (quote and isinstance(quote.price, (int, float))):
                mismatch_pct = abs(float(quote.price) - float(uprice)) / max(float(uprice), 1e-9)
            polygon_bundle.setdefault("consistency", {})
            polygon_bundle["consistency"].update(
                {
                    "quote_age_s": getattr(quote, "age_seconds", None),
                    "underlying_vs_quote_pct": round(mismatch_pct, 6) if mismatch_pct is not None else None,
                    "policy": {"threshold_pct": 0.002},
                }
            )
            if (mismatch_pct is not None and mismatch_pct > 0.002) or (
                getattr(quote, "age_seconds", 0) and getattr(quote, "age_seconds", 0) > 5
            ):
                polygon_bundle["hold_refresh"] = True
        except Exception:
            logger.debug("consistency check failed", exc_info=True)
        response["options"] = polygon_bundle
    level_tokens = _extract_levels_for_chart(key_levels)

    chart_params = {
        "symbol": _tv_symbol(symbol),
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


async def _rebuild_plan_layers(plan_id: str, snapshot: Dict[str, Any], request: Request) -> Dict[str, Any] | None:
    plan_block = snapshot.get("plan") or {}
    symbol = (plan_block.get("symbol") or "").strip()
    style = plan_block.get("style")
    slug_meta = _parse_plan_slug(plan_id)
    if not symbol and slug_meta:
        symbol = (slug_meta.get("symbol") or "").strip()
        style = style or slug_meta.get("style")
    if not symbol:
        return None

    logger.info(
        "chart_layers_rebuild_start",
        extra={"plan_id": plan_id, "symbol": symbol, "style": style},
    )
    plan_request = PlanRequest(symbol=symbol, style=style, plan_id=plan_id)
    backfill_user = AuthedUser(user_id="layers_backfill")
    try:
        response = await gpt_plan(plan_request, request, Response(), backfill_user)
    except HTTPException as exc:
        logger.warning(
            "chart_layers_rebuild_http_error",
            extra={"plan_id": plan_id, "symbol": symbol, "status": exc.status_code, "detail": exc.detail},
        )
        return None
    logger.info(
        "chart_layers_rebuild_success",
        extra={
            "plan_id": plan_id,
            "symbol": symbol,
            "has_layers": bool(response.plan_layers),
        },
    )
    return response.plan_layers


def _fallback_plan_layers(plan_block: Mapping[str, Any], plan_id: str, *, reason: str = "missing") -> Dict[str, Any]:
    session_state = plan_block.get("session_state") or {}
    as_of = str(session_state.get("as_of") or "")
    symbol = (plan_block.get("symbol") or "").strip().upper()
    interval = (
        plan_block.get("chart_timeframe")
        or plan_block.get("interval")
        or plan_block.get("charts", {}).get("params", {}).get("interval")
        or ""
    )
    precision = get_precision(symbol) if symbol else 2
    planning_context = plan_block.get("planning_context") or plan_block.get("planningContext") or "unknown"
    notes = ["Plan layers unavailable; showing chart without annotations"]
    if reason == "stale":
        notes = ["Plan layers out of date; displaying chart without annotations"]
    meta: Dict[str, Any] = {
        "confidence_factors": [],
        "confluence": [],
        "status": reason,
        "notes": notes,
    }
    logger.info(
        "chart_layers_fallback",
        extra={
            "plan_id": plan_id,
            "symbol": symbol or None,
            "interval": interval,
            "reason": reason,
        },
    )
    return {
        "plan_id": plan_id,
        "symbol": symbol or None,
        "interval": interval,
        "as_of": as_of,
        "planning_context": planning_context,
        "precision": precision,
        "levels": [],
        "zones": [],
        "annotations": [],
        "meta": meta,
    }


@app.get(
    "/api/v1/gpt/chart-layers",
    summary="Return plan-bound chart layers",
    tags=["charts"],
)
async def chart_layers_endpoint(
    request: Request,
    plan_id: str = Query(..., min_length=3),
    refresh: bool = Query(False),
) -> Dict[str, Any]:
    settings = get_settings()
    if not getattr(settings, "ff_layers_endpoint", False):
        raise HTTPException(status_code=404, detail="Plan layers unavailable")
    logger.info(
        "chart_layers_request",
        extra={"plan_id": plan_id, "refresh": refresh},
    )
    snapshot = await _ensure_snapshot(plan_id, version=None, request=request)
    plan_block = snapshot.get("plan") or {}
    layers: Dict[str, Any] | None = plan_block.get("plan_layers") or snapshot.get("plan_layers")

    refreshed = False

    while True:
        if not layers or not isinstance(layers, dict):
            if refreshed and not refresh:
                layers = _fallback_plan_layers(plan_block, plan_id, reason="missing")
                break
            rebuilt = await _rebuild_plan_layers(plan_id, snapshot, request)
            if rebuilt is None:
                logger.warning(
                    "plan_layers_missing_backfill_failed",
                    extra={"plan_id": plan_id},
                )
                layers = _fallback_plan_layers(plan_block, plan_id, reason="missing")
                break
            logger.info(
                "plan_layers_missing_rebuilt",
                extra={"plan_id": plan_id, "attempt": refreshed + 1},
            )
            refreshed = True
            snapshot = await _ensure_snapshot(plan_id, version=None, request=request)
            plan_block = snapshot.get("plan") or {}
            layers = plan_block.get("plan_layers") or snapshot.get("plan_layers")
            continue

        plan_session = plan_block.get("session_state") or {}
        plan_as_of = str(plan_session.get("as_of") or "").strip()
        layers_as_of = str(layers.get("as_of") or "").strip()

        if plan_as_of and layers_as_of and plan_as_of != layers_as_of:
            rebuilt = await _rebuild_plan_layers(plan_id, snapshot, request)
            if rebuilt:
                refreshed = True
                snapshot = await _ensure_snapshot(plan_id, version=None, request=request)
                plan_block = snapshot.get("plan") or {}
                layers = plan_block.get("plan_layers") or snapshot.get("plan_layers")
                logger.info(
                    "plan_layers_stale_rebuilt",
                    extra={
                        "plan_id": plan_id,
                        "plan_as_of": plan_as_of,
                        "layers_as_of": layers_as_of,
                    },
                )
                continue
            logger.warning(
                "plan_layers_asof_mismatch",
                extra={
                    "plan_id": plan_id,
                    "plan_as_of": plan_as_of,
                    "layers_as_of": layers_as_of,
                },
            )
            layers = _fallback_plan_layers(plan_block, plan_id, reason="stale")
            break
        break

    if not layers or not isinstance(layers, dict):
        logger.warning("plan_layers_missing", extra={"plan_id": plan_id})
        layers = _fallback_plan_layers(plan_block, plan_id)

    payload = copy.deepcopy(layers)
    payload.setdefault("plan_id", plan_id)
    if str(payload.get("plan_id")) != plan_id:
        payload["plan_id"] = plan_id
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
app.include_router(session_router)
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
            "health": "/healthz",
        },
    }
