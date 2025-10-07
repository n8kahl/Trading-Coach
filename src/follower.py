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
from typing import Dict, Optional, Callable

import pandas as pd

from .calculations import atr, ema


class TradeState(enum.Enum):
    PLANNED = "planned"
    ENTERED = "entered"
    SCALED = "scaled"
    TRAILING = "trailing"
    EXITED = "exited"


@dataclass
class TradeFollower:
    symbol: str
    entry_price: float
    direction: str  # "long" or "short"
    risk_multiple: float = 1.0
    atr_period: int = 14
    stop_distance_multiplier: float = 1.0
    tp_multiple: float = 1.0
    state: TradeState = TradeState.PLANNED
    stop_price: float = field(init=False)
    tp_price: float = field(init=False)
    trailing_stop: float = field(init=False)
    scaled: bool = field(default=False)

    def __post_init__(self) -> None:
        # Initialize stop and target based on ATR at entry.  In practice you
        # should compute ATR from intraday data.  Here we set placeholders and
        # expect the caller to update them once data is available.
        self.stop_price = self.entry_price - self.stop_distance_multiplier
        self.tp_price = self.entry_price + self.tp_multiple
        self.trailing_stop = self.stop_price

    def update_from_price(self, price: float, atr_value: float) -> Optional[str]:
        """Update trade state based on the latest price and ATR.

        Args:
            price: The most recent traded price of the underlying.
            atr_value: The current Average True Range value.

        Returns:
            A message with coaching instructions, or `None` if no action is
            required.
        """
        message = None
        # Initialize stop and TP on first update after entering
        if self.state == TradeState.PLANNED:
            if self.direction == "long":
                self.stop_price = self.entry_price - self.stop_distance_multiplier * atr_value
                self.tp_price = self.entry_price + self.tp_multiple * atr_value
            else:
                self.stop_price = self.entry_price + self.stop_distance_multiplier * atr_value
                self.tp_price = self.entry_price - self.tp_multiple * atr_value
            self.state = TradeState.ENTERED
            message = f"Entered trade at {self.entry_price:.2f}. Stop: {self.stop_price:.2f}, TP1: {self.tp_price:.2f}."
            return message

        # Check for stop loss
        if self.direction == "long" and price <= self.stop_price:
            self.state = TradeState.EXITED
            return f"Price hit stop loss at {price:.2f}. Exiting trade."
        if self.direction == "short" and price >= self.stop_price:
            self.state = TradeState.EXITED
            return f"Price hit stop loss at {price:.2f}. Exiting trade."

        # Check for take profit level and scaling
        if not self.scaled:
            if (self.direction == "long" and price >= self.tp_price) or (self.direction == "short" and price <= self.tp_price):
                self.scaled = True
                # Move stop to breakeven after scaling
                self.stop_price = self.entry_price
                # Set a new trailing stop target at 1× ATR beyond the current price
                if self.direction == "long":
                    self.trailing_stop = price - atr_value
                else:
                    self.trailing_stop = price + atr_value
                message = (
                    f"Reached TP1 at {price:.2f}. Scaled out half. Stop moved to breakeven. "
                    f"New trail stop: {self.trailing_stop:.2f}."
                )
                self.state = TradeState.SCALED
                return message

        # After scaling, trail the stop using a simple chandelier exit (ATR based)
        if self.scaled and self.state != TradeState.EXITED:
            if self.direction == "long":
                new_trail = price - atr_value
                if new_trail > self.trailing_stop:
                    self.trailing_stop = new_trail
                    message = f"Trail stop raised to {self.trailing_stop:.2f}."
                # Exit if price falls below trailing stop
                if price <= self.trailing_stop:
                    self.state = TradeState.EXITED
                    message = f"Price fell below trailing stop {self.trailing_stop:.2f}. Exiting trade."
                    return message
            else:
                new_trail = price + atr_value
                if new_trail < self.trailing_stop:
                    self.trailing_stop = new_trail
                    message = f"Trail stop lowered to {self.trailing_stop:.2f}."
                if price >= self.trailing_stop:
                    self.state = TradeState.EXITED
                    message = f"Price rose above trailing stop {self.trailing_stop:.2f}. Exiting trade."
                    return message

        return message