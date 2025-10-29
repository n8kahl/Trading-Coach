"""Real‑time trade follower for live coaching.

The `TradeFollower` class maintains the state of an in‑progress trade and
generates management instructions as price evolves.  It computes dynamic
stops and targets using ATR‑based rules and tracks scaling events.  In a
production system, instances of this class would be created for each trade
and fed real‑time price updates via WebSocket.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class TradeState(enum.Enum):
    PLANNED = "planned"
    ENTERED = "entered"
    SCALED = "scaled"
    TRAILING = "trailing"
    EXITED = "exited"


@dataclass
@dataclass
class FollowerUpdate:
    plan_id: str
    symbol: str
    state: TradeState
    note: str
    trailing_stop: float | None = None
    scaled: bool = False
    last_price: float | None = None
    event: str | None = None
    timestamp: str | None = None
    exit_reason: str | None = None


@dataclass
class TradeFollower:
    plan_id: str
    symbol: str
    direction: str  # "long" or "short"
    entry_price: float
    stop_price: float
    tp_price: float
    atr_value: float | None = None
    trail_multiple: float = 1.0
    stop_breakeven: bool = True
    state: TradeState = TradeState.PLANNED
    trailing_stop: float | None = None
    scaled: bool = False
    auto_replan_triggered: bool = False
    stop_warning_active: bool = False
    last_stop_warning_price: float | None = None

    def refresh_plan(self, *, entry: float, stop: float, target: float | None, atr: float | None) -> None:
        self.entry_price = entry
        self.stop_price = stop
        self.tp_price = target if target is not None else entry + (entry - stop if self.direction == "long" else stop - entry)
        self.atr_value = atr
        self.state = TradeState.PLANNED
        self.trailing_stop = stop
        self.scaled = False
        self.auto_replan_triggered = False
        self.stop_warning_active = False
        self.last_stop_warning_price = None

    def _entered(self, price: float) -> bool:
        if self.direction == "long":
            return price >= self.entry_price
        return price <= self.entry_price

    def _build_update(self, *, note: str, state: TradeState, price: float, event: str | None = None, exit_reason: str | None = None) -> FollowerUpdate:
        return FollowerUpdate(
            plan_id=self.plan_id,
            symbol=self.symbol,
            state=state,
            note=note,
            trailing_stop=self.trailing_stop,
            scaled=self.scaled,
            last_price=price,
            event=event,
            exit_reason=exit_reason,
            timestamp=_utc_now(),
        )

    def _distance_to_stop(self, price: float) -> float:
        if self.direction == "long":
            return price - self.stop_price
        return self.stop_price - price

    def _stop_warning_threshold(self) -> float | None:
        if self.atr_value and self.atr_value > 0:
            return float(self.atr_value) * 0.3
        distance = abs(self.entry_price - self.stop_price)
        if distance > 0:
            return distance * 0.2
        return None

    def update_from_price(self, price: float, atr_value: Optional[float] = None) -> Optional[FollowerUpdate]:
        if atr_value is None:
            atr_value = self.atr_value if isinstance(self.atr_value, (int, float)) else None
        if atr_value is not None and atr_value > 0:
            self.atr_value = float(atr_value)

        # Transition from planned → entered once price touches entry level.
        if self.state == TradeState.PLANNED and self._entered(price):
            if self.direction == "long" and price <= self.stop_price:
                # Avoid immediate stop-out; keep stop slightly below entry.
                self.stop_price = min(self.stop_price, self.entry_price - (self.atr_value or (self.entry_price * 0.002)))
            elif self.direction == "short" and price >= self.stop_price:
                self.stop_price = max(self.stop_price, self.entry_price + (self.atr_value or (self.entry_price * 0.002)))
            if self.atr_value and self.atr_value > 0:
                offset = self.atr_value * self.trail_multiple
                if self.direction == "long":
                    self.tp_price = self.entry_price + offset
                else:
                    self.tp_price = self.entry_price - offset
            self.state = TradeState.ENTERED
            self.trailing_stop = self.stop_price
            return self._build_update(
                note=f"Entered trade at {self.entry_price:.2f}. Stop {self.stop_price:.2f}, TP1 {self.tp_price:.2f}.",
                state=self.state,
                price=price,
                event="entered",
            )

        # If still planned and not triggered, no coaching output.
        if self.state == TradeState.PLANNED:
            return None

        # Stop loss check
        if self.direction == "long" and price <= self.stop_price:
            self.state = TradeState.EXITED
            self.stop_warning_active = False
            self.last_stop_warning_price = None
            return self._build_update(
                note=f"Stop triggered at {price:.2f}. Exiting trade.",
                state=self.state,
                price=price,
                event="stop",
                exit_reason="stop",
            )
        if self.direction == "short" and price >= self.stop_price:
            self.state = TradeState.EXITED
            self.stop_warning_active = False
            self.last_stop_warning_price = None
            return self._build_update(
                note=f"Stop triggered at {price:.2f}. Exiting trade.",
                state=self.state,
                price=price,
                event="stop",
                exit_reason="stop",
            )

        # Stop proximity warning
        if self.state in {TradeState.ENTERED, TradeState.SCALED, TradeState.TRAILING}:
            threshold = self._stop_warning_threshold()
            if threshold:
                distance = self._distance_to_stop(price)
                if distance < 0:
                    self.stop_warning_active = False
                    self.last_stop_warning_price = None
                elif distance <= threshold and not self.stop_warning_active:
                    self.stop_warning_active = True
                    self.last_stop_warning_price = price
                    return self._build_update(
                        note=f"Stop approaching — price within {distance:.2f} of stop.",
                        state=self.state,
                        price=price,
                        event="stop_warning",
                    )
                elif distance > threshold * 1.5 and self.stop_warning_active:
                    self.stop_warning_active = False
                    self.last_stop_warning_price = None

        # Scaling check
        if not self.scaled:
            hit_target = (self.direction == "long" and price >= self.tp_price) or (
                self.direction == "short" and price <= self.tp_price
            )
            if hit_target:
                self.scaled = True
                self.state = TradeState.SCALED
                if self.stop_breakeven:
                    self.stop_price = self.entry_price
                self.stop_warning_active = False
                self.last_stop_warning_price = None
                if self.atr_value and self.atr_value > 0:
                    self.trailing_stop = price - self.atr_value if self.direction == "long" else price + self.atr_value
                else:
                    offset = abs(self.entry_price - self.stop_price)
                    self.trailing_stop = price - offset if self.direction == "long" else price + offset
                return self._build_update(
                    note=f"TP1 reached at {price:.2f}. Scaled position and moved stop to {self.stop_price:.2f}. Trail now {self.trailing_stop:.2f}.",
                    state=self.state,
                    price=price,
                    event="tp_hit",
                )

        # Trailing stop adjustments after scaling
        if self.scaled and self.state != TradeState.EXITED and self.trailing_stop is not None:
            if self.direction == "long":
                trail = price - (self.atr_value if self.atr_value else abs(self.entry_price - self.stop_price))
                if trail > self.trailing_stop:
                    self.trailing_stop = trail
                    return self._build_update(
                        note=f"Trail stop raised to {self.trailing_stop:.2f}.",
                        state=TradeState.TRAILING,
                        price=price,
                        event="trail_adjusted",
                    )
                if price <= self.trailing_stop:
                    self.state = TradeState.EXITED
                    self.stop_warning_active = False
                    self.last_stop_warning_price = None
                    return self._build_update(
                        note=f"Trail stop {self.trailing_stop:.2f} hit. Closing trade.",
                        state=self.state,
                        price=price,
                        event="trail_stop",
                        exit_reason="trail",
                    )
            else:
                trail = price + (self.atr_value if self.atr_value else abs(self.entry_price - self.stop_price))
                if trail < self.trailing_stop:
                    self.trailing_stop = trail
                    return self._build_update(
                        note=f"Trail stop lowered to {self.trailing_stop:.2f}.",
                        state=TradeState.TRAILING,
                        price=price,
                        event="trail_adjusted",
                    )
                if price >= self.trailing_stop:
                    self.state = TradeState.EXITED
                    self.stop_warning_active = False
                    self.last_stop_warning_price = None
                    return self._build_update(
                        note=f"Trail stop {self.trailing_stop:.2f} hit. Closing trade.",
                        state=self.state,
                        price=price,
                        event="trail_stop",
                        exit_reason="trail",
                    )

        return None
