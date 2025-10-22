from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Literal

from .market_clock import get_market_state, most_recent_regular_close

DataMode = Literal["live", "lkg"]


@dataclass(frozen=True)
class DataRoute:
    mode: DataMode
    as_of: datetime
    planning_context: Literal["live", "frozen"]


def _ensure_now(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(timezone.utc)
    if now.tzinfo is None:
        return now.replace(tzinfo=timezone.utc)
    return now.astimezone(timezone.utc)


def pick_data_source(
    now: datetime | None = None,
    *,
    is_holiday: Callable[[datetime], bool] | None = None,
) -> DataRoute:
    current = _ensure_now(now)
    state = get_market_state(current, is_holiday=is_holiday)
    if state == "open":
        return DataRoute(mode="live", as_of=current, planning_context="live")
    close_dt = most_recent_regular_close(current, is_holiday=is_holiday)
    return DataRoute(mode="lkg", as_of=close_dt, planning_context="frozen")
