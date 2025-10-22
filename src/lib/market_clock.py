from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Callable, Literal
import zoneinfo

NY = zoneinfo.ZoneInfo("America/New_York")
MarketState = Literal["open", "closed"]

if TYPE_CHECKING:  # pragma: no cover - import guard for typing only
    from .data_source import DataRoute


def _ensure_aware(dt: datetime) -> datetime:
    """Coerce naive datetimes into UTC-aware values."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def get_market_state(
    now: datetime | None = None,
    *,
    is_holiday: Callable[[datetime], bool] | None = None,
) -> MarketState:
    """Return the current market state for the regular U.S. equity session."""
    now_dt = _ensure_aware(now or datetime.now(timezone.utc)).astimezone(NY)
    if now_dt.weekday() >= 5:
        return "closed"
    if is_holiday and is_holiday(now_dt):
        return "closed"
    open_dt = now_dt.replace(hour=9, minute=30, second=0, microsecond=0)
    close_dt = now_dt.replace(hour=16, minute=0, second=0, microsecond=0)
    return "open" if open_dt <= now_dt < close_dt else "closed"


def most_recent_regular_close(
    now: datetime | None = None,
    *,
    is_holiday: Callable[[datetime], bool] | None = None,
) -> datetime:
    """Return the most recent regular-session close (16:00 ET) in America/New_York."""
    reference = _ensure_aware(now or datetime.now(timezone.utc)).astimezone(NY)
    probe = reference
    for _ in range(14):
        if probe.weekday() < 5 and not (is_holiday and is_holiday(probe)):
            close_dt = probe.replace(hour=16, minute=0, second=0, microsecond=0)
            if close_dt <= reference:
                return close_dt
        probe = (probe - timedelta(days=1)).replace(hour=12, minute=0, second=0, microsecond=0)
    fallback = reference.replace(hour=16, minute=0, second=0, microsecond=0)
    if fallback > reference:
        fallback -= timedelta(days=1)
    return fallback


def apply_simulate_open(route: "DataRoute", *, now: datetime | None = None) -> "DataRoute":
    """Force a DataRoute into live mode for simulate_open requests."""
    from .data_source import DataRoute as _DataRoute

    as_of = _ensure_aware(now or datetime.now(timezone.utc))
    return _DataRoute(mode="live", as_of=as_of, planning_context="live")
