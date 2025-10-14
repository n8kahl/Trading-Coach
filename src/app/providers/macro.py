"""Macro event context provider."""

from __future__ import annotations

from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

import httpx

from ...config import get_settings
from ..services.session_state import parse_session_as_of

_DEFAULT_WINDOW_MINUTES = 240


def _parse_datetime(value: str | None) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _ensure_as_of(as_of: str | None) -> datetime:
    parsed = _parse_datetime(as_of)
    if parsed:
        return parsed
    session = {"as_of": as_of, "tz": "America/New_York"} if as_of else {}
    fallback = parse_session_as_of(session)
    return fallback or datetime.now(timezone.utc)


def _summarise_event(name: str, ts: datetime, now: datetime, severity: str | None = None) -> Dict[str, Any]:
    delta_minutes = int((ts - now).total_seconds() // 60)
    if severity is None:
        severity = "medium"
        key = name.lower()
        if "fomc" in key or "fed" in key or "cpi" in key:
            severity = "high"
    return {"name": name, "ts": ts.isoformat(), "severity": severity, "minutes": delta_minutes}


def _fetch_macro_minutes() -> Dict[str, Any]:
    settings = get_settings()
    base = getattr(settings, "enrichment_service_url", None) or ""
    if not base:
        return {}
    url = f"{base.rstrip('/')}/gpt/events/earnings"
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.get(url, params={"symbol": "SPY"})
            resp.raise_for_status()
    except httpx.HTTPError:
        return {}
    payload = resp.json()
    return payload.get("events") or {}


def get_event_window(as_of: str | None) -> Dict[str, Any]:
    """Return a compact macro event summary window."""

    now = _ensure_as_of(as_of)
    events_raw = _fetch_macro_minutes()
    schedule: List[Dict[str, Any]] = []
    mapping = [
        ("FOMC decision", events_raw.get("next_fomc_minutes"), "high"),
        ("CPI release", events_raw.get("next_cpi_minutes"), "high"),
        ("Nonfarm payrolls", events_raw.get("next_nfp_minutes"), "medium"),
    ]
    for label, minutes, severity in mapping:
        if minutes is None:
            continue
        try:
            minutes_val = int(float(minutes))
        except (TypeError, ValueError):
            continue
        ts = now + timedelta(minutes=minutes_val)
        schedule.append(_summarise_event(label, ts, now, severity=severity))
    if events_raw.get("within_event_window"):
        schedule.append(_summarise_event("Event window", now, now, severity="high"))
    upcoming: List[Dict[str, Any]] = []
    active: List[Dict[str, Any]] = []
    min_minutes = None
    for item in schedule:
        minutes = item.get("minutes")
        if minutes is None:
            continue
        if minutes <= 0:
            active.append(item)
        elif minutes <= _DEFAULT_WINDOW_MINUTES:
            upcoming.append(item)
            if min_minutes is None or minutes < min_minutes:
                min_minutes = minutes
    return {
        "upcoming": upcoming,
        "active": active,
        "min_minutes_to_event": min_minutes,
    }


__all__ = ["get_event_window"]
