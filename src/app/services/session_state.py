"""Session state utilities for market-aware responses."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, time, timedelta, timezone
from typing import Any, Dict, Mapping, Optional, Tuple
import os
import time as _time

import httpx
from zoneinfo import ZoneInfo

from ...market_clock import MarketClock
from src.config import get_settings, get_massive_api_key

_ET = ZoneInfo("America/New_York")
_CLOCK = MarketClock()
_BASE_CANDIDATE = os.getenv("MARKETDATA_BASE_URL", "https://api.massive.com").rstrip("/")
_POLYGON_FALLBACK = os.getenv("POLYGON_BASE_URL", "https://api.polygon.io").rstrip("/")
_POLYGON_BASE = _BASE_CANDIDATE or _POLYGON_FALLBACK
_STATUS_CACHE: Optional[Tuple[float, Dict[str, Any]]] = None
_STATUS_CACHE_TTL = 30.0


def _parse_iso_timestamp(raw: Any) -> Optional[datetime]:
    if raw is None:
        return None
    if isinstance(raw, datetime):
        dt = raw
    elif isinstance(raw, (int, float)):
        dt = datetime.fromtimestamp(float(raw), tz=timezone.utc)
    elif isinstance(raw, str):
        token = raw.strip()
        if not token:
            return None
        token = token.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(token)
        except ValueError:
            return None
    else:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _polygon_market_status() -> Optional[Dict[str, Any]]:
    global _STATUS_CACHE

    settings = get_settings()
    api_key = get_massive_api_key(settings)
    if not api_key:
        return None

    now = _time.monotonic()
    cached = _STATUS_CACHE
    if cached and now - cached[0] < _STATUS_CACHE_TTL:
        return dict(cached[1])

    url = f"{_POLYGON_BASE}/v1/marketstatus/now"
    params = {"apiKey": api_key}
    try:
        with httpx.Client(timeout=3.0) as client:
            resp = client.get(url, params=params)
            resp.raise_for_status()
    except httpx.HTTPError:
        return None

    payload = resp.json()
    _STATUS_CACHE = (now, dict(payload))
    return dict(payload)


@dataclass(slots=True)
class SessionState:
    """Represents the current US equities trading session."""

    status: str
    as_of: str
    next_open: str
    tz: str
    banner: str

    def to_dict(self) -> Dict[str, str]:
        """Return a JSON-serialisable representation."""
        return asdict(self)


def _format(dt: datetime | None) -> str:
    if dt is None:
        return ""
    aware = dt.astimezone(_ET)
    return aware.replace(microsecond=0).isoformat()


def _fallback_as_of(now_et: datetime) -> datetime:
    """Fallback to the previous regular close when outside cash hours."""
    if now_et.weekday() >= 5:
        days_back = (now_et.weekday() - 4) if now_et.weekday() >= 5 else 1
        prior_business = now_et - timedelta(days=days_back)
    else:
        prior_business = now_et
    close_time = time(hour=16, minute=0)
    as_of = prior_business.replace(hour=close_time.hour, minute=close_time.minute, second=0, microsecond=0)
    if as_of > now_et:
        as_of -= timedelta(days=1)
    while as_of.weekday() >= 5:
        as_of -= timedelta(days=1)
    return as_of


def session_now() -> SessionState:
    """Return the current session metadata (ET)."""

    snapshot = _CLOCK.snapshot()
    now_et = snapshot.now_et.astimezone(_ET)
    status = snapshot.status  # "open" | "pre" | "post" | "closed"

    if status == "open":
        as_of_dt = now_et
        banner = "Market open"
    else:
        as_of_dt = _CLOCK.last_rth_close(at=now_et)
        if as_of_dt is None:
            as_of_dt = _fallback_as_of(now_et)
        banner = "Market closed — using last regular session"
        if status == "pre":
            banner = "Premarket — using prior close data"
        elif status == "post":
            banner = "After hours — using regular session close"

    if as_of_dt.tzinfo is None:
        as_of_dt = as_of_dt.replace(tzinfo=timezone.utc).astimezone(_ET)

    next_open, _next_close = _CLOCK.next_open_close(at=now_et)

    final_status = status
    polygon_status = _polygon_market_status()
    if polygon_status:
        polygon_now = _parse_iso_timestamp(polygon_status.get("serverTime"))
        if polygon_now is not None:
            now_et = polygon_now.astimezone(_ET)
        stocks_hours = (polygon_status.get("marketHours") or {}).get("stocks") or {}
        exchanges = polygon_status.get("exchanges") or {}
        exchange_state = str(exchanges.get("nyse") or exchanges.get("stocks") or "").lower()
        market_flag = str(polygon_status.get("market") or "").lower()
        session_label = str(stocks_hours.get("session") or "").lower()
        is_open = bool(stocks_hours.get("isOpen"))

        if is_open or exchange_state == "open" or market_flag == "open":
            final_status = "open"
            as_of_dt = polygon_now.astimezone(_ET) if polygon_now else now_et
            banner = "Market open"
        else:
            final_status = "closed"
            previous_close = ((polygon_status.get("previous") or {}).get("stocks") or {}).get("close")
            previous_close_dt = _parse_iso_timestamp(previous_close)
            close_hint = _parse_iso_timestamp(stocks_hours.get("close"))
            if previous_close_dt:
                as_of_dt = previous_close_dt.astimezone(_ET)
            elif close_hint:
                as_of_dt = close_hint.astimezone(_ET)
            if session_label == "extended":
                banner = "After hours — using regular session close"
            elif session_label == "premarket":
                banner = "Premarket — using prior close data"
            else:
                market_close_label = as_of_dt.strftime("%Y-%m-%d %H:%M %Z")
                banner = f"Market closed — using {market_close_label}"

        # Update next open if Polygon supplied it
        open_hint = _parse_iso_timestamp(stocks_hours.get("open"))
        if open_hint:
            next_open = open_hint.astimezone(_ET)

    if snapshot.status == "open":
        final_status = "open"
        banner = "Market open"
        as_of_dt = now_et

    return SessionState(
        status="open" if final_status == "open" else "closed",
        as_of=_format(as_of_dt),
        next_open=_format(next_open),
        tz="America/New_York",
        banner=banner,
    )


def parse_session_as_of(session: Mapping[str, str]) -> Optional[datetime]:
    """Return the session's as_of timestamp in UTC."""
    if not session:
        return None
    raw = session.get("as_of")
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    tz_label = session.get("tz") or "America/New_York"
    tz = ZoneInfo(tz_label)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz)
    return dt.astimezone(timezone.utc)


__all__ = ["SessionState", "session_now", "parse_session_as_of"]
