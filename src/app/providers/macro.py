"""Macro event context provider."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Mapping, Optional

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


def _summarise_event(name: str, ts: datetime, now: datetime) -> Dict[str, Any]:
    delta_minutes = int((ts - now).total_seconds() // 60)
    severity = "medium"
    key = name.lower()
    if "fomc" in key or "fed" in key:
        severity = "high"
    elif "cpi" in key or "employment" in key:
        severity = "high"
    return {"name": name, "ts": ts.isoformat(), "severity": severity, "minutes": delta_minutes}


def _mock_schedule(now: datetime) -> List[Dict[str, Any]]:
    base = now.replace(hour=20, minute=0, second=0, microsecond=0)
    return [
        _summarise_event("FOMC press conference", base + timedelta(days=2), now),
        _summarise_event("Initial jobless claims", base + timedelta(days=3), now),
        _summarise_event("CPI release", base + timedelta(days=10), now),
    ]


def get_event_window(as_of: str | None) -> Dict[str, Any]:
    """Return a compact macro event summary window."""

    now = _ensure_as_of(as_of)
    schedule = _mock_schedule(now)
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
