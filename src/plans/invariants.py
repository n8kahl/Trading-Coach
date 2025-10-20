"""Shared invariant checks for generated plans."""

from __future__ import annotations

from typing import Sequence


class GeometryInvariantError(ValueError):
    """Raised when generated price geometry violates directional invariants."""


def assert_invariants(direction: str, entry: float, stop: float, targets: Sequence[float], rr_min: float) -> None:
    if not targets:
        raise GeometryInvariantError("missing_targets")

    direction = (direction or "").lower()
    entry_val = float(entry)
    stop_val = float(stop)
    rr_min = float(rr_min or 0.0)

    if direction not in {"long", "short"}:
        raise GeometryInvariantError("invalid_direction")

    primary_target = float(targets[0])
    if direction == "long":
        if stop_val >= entry_val:
            raise GeometryInvariantError("stop_not_below_entry")
        if primary_target <= entry_val:
            raise GeometryInvariantError("tp1_not_above_entry")
        previous = entry_val
        for idx, price in enumerate(targets, start=1):
            price_val = float(price)
            if price_val <= previous:
                raise GeometryInvariantError(f"tp{idx}_not_above_previous")
            previous = price_val
        risk = entry_val - stop_val
        reward = primary_target - entry_val
    else:
        if stop_val <= entry_val:
            raise GeometryInvariantError("stop_not_above_entry")
        if primary_target >= entry_val:
            raise GeometryInvariantError("tp1_not_below_entry")
        previous = entry_val
        for idx, price in enumerate(targets, start=1):
            price_val = float(price)
            if price_val >= previous:
                raise GeometryInvariantError(f"tp{idx}_not_below_previous")
            previous = price_val
        risk = stop_val - entry_val
        reward = entry_val - primary_target

    if risk <= 0 or reward <= 0:
        raise GeometryInvariantError("invalid_risk_reward")
    rr = reward / risk
    if rr_min > 0 and rr < rr_min:
        raise GeometryInvariantError("rr_too_low")


__all__ = ["assert_invariants", "GeometryInvariantError"]
