from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable, Literal

from .data_route import DataRoute
from .market_clock import route_for_request

DataMode = Literal["live", "lkg"]


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
    return route_for_request(False, now=current, is_holiday=is_holiday)


__all__ = ["DataRoute", "DataMode", "pick_data_source"]
