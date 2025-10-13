from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Iterable, Optional, Set, Tuple
from zoneinfo import ZoneInfo


_DEFAULT_HOLIDAYS_2025: Set[str] = {
    "2025-01-01",  # New Year's Day
    "2025-01-20",  # MLK Jr. Day
    "2025-02-17",  # Presidents' Day
    "2025-04-18",  # Good Friday
    "2025-05-26",  # Memorial Day
    "2025-06-19",  # Juneteenth
    "2025-07-04",  # Independence Day
    "2025-09-01",  # Labor Day
    "2025-11-27",  # Thanksgiving
    "2025-12-25",  # Christmas
}


_TZ_NEW_YORK = ZoneInfo("America/New_York")

_PRE_START = time(4, 0)
_RTH_START = time(9, 30)
_RTH_END = time(16, 0)
_POST_END = time(20, 0)


@dataclass(frozen=True)
class MarketSessionSnapshot:
    status: str
    session: str
    now_et: datetime
    next_open_et: datetime
    next_close_et: datetime


class MarketClock:
    """Utility for determining US equity market session state."""

    def __init__(self, *, holidays: Optional[Iterable[str]] = None) -> None:
        self._holidays: Set[str] = set(holidays or _DEFAULT_HOLIDAYS_2025)

    def now_et(self) -> datetime:
        return datetime.now(tz=_TZ_NEW_YORK)

    def _is_weekend(self, dt: datetime) -> bool:
        return dt.weekday() >= 5

    def _is_holiday(self, dt: datetime) -> bool:
        return dt.strftime("%Y-%m-%d") in self._holidays

    def _session_bounds(self, day: datetime) -> Tuple[datetime, datetime]:
        open_dt = day.replace(hour=_RTH_START.hour, minute=_RTH_START.minute, second=0, microsecond=0)
        close_dt = day.replace(hour=_RTH_END.hour, minute=_RTH_END.minute, second=0, microsecond=0)
        return open_dt, close_dt

    def _next_trading_day(self, dt: datetime) -> datetime:
        cursor = dt
        while True:
            cursor += timedelta(days=1)
            if cursor.weekday() < 5 and not self._is_holiday(cursor):
                return cursor

    def _previous_trading_day(self, dt: datetime) -> datetime:
        cursor = dt
        while True:
            cursor -= timedelta(days=1)
            if cursor.weekday() < 5 and not self._is_holiday(cursor):
                return cursor

    def _classify_status(self, now: datetime) -> Tuple[str, str]:
        if self._is_weekend(now) or self._is_holiday(now):
            return "closed", "CLOSED"
        open_dt, close_dt = self._session_bounds(now)
        pre_dt = now.replace(hour=_PRE_START.hour, minute=_PRE_START.minute, second=0, microsecond=0)
        post_dt = now.replace(hour=_POST_END.hour, minute=_POST_END.minute, second=0, microsecond=0)
        if open_dt <= now <= close_dt:
            return "open", "RTH"
        if pre_dt <= now < open_dt:
            return "pre", "PRE"
        if close_dt < now <= post_dt:
            return "post", "POST"
        return "closed", "CLOSED"

    def is_rth_open(self, *, at: Optional[datetime] = None) -> bool:
        now = at or self.now_et()
        status, _ = self._classify_status(now)
        return status == "open"

    def next_open_close(self, *, at: Optional[datetime] = None) -> Tuple[datetime, datetime]:
        now = at or self.now_et()
        status, _ = self._classify_status(now)
        if status == "open":
            _, close_dt = self._session_bounds(now)
            next_day = self._next_trading_day(now)
            next_open_dt, _ = self._session_bounds(next_day)
            return next_open_dt, close_dt
        if status == "pre":
            open_dt, close_dt = self._session_bounds(now)
            return open_dt, close_dt
        next_day = self._next_trading_day(now if status == "post" else now)
        open_dt, close_dt = self._session_bounds(next_day)
        return open_dt, close_dt

    def last_rth_close(self, *, at: Optional[datetime] = None) -> datetime:
        now = at or self.now_et()
        status, _ = self._classify_status(now)
        if status == "open":
            return now
        open_dt, close_dt = self._session_bounds(now)
        if status == "post":
            return close_dt
        previous_day = self._previous_trading_day(now)
        _, prev_close = self._session_bounds(previous_day)
        return prev_close

    def snapshot(self, *, at: Optional[datetime] = None) -> MarketSessionSnapshot:
        now = at or self.now_et()
        status, session = self._classify_status(now)
        next_open, next_close = self.next_open_close(at=now)
        return MarketSessionSnapshot(
            status=status,
            session=session,
            now_et=now,
            next_open_et=next_open,
            next_close_et=next_close,
        )
