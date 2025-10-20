"""Helpers for injecting style-aware higher timeframe levels into geometry maps."""

from __future__ import annotations

import math
from typing import Any, Mapping, MutableMapping, Sequence, Tuple


def _safe_assign(levels: MutableMapping[str, float], key: str, value: Any) -> None:
    """Assign a float price into the levels map if it is finite and not already set."""
    if key in levels:
        return
    if isinstance(value, (int, float)) and math.isfinite(value):
        levels[key] = float(value)


def inject_style_levels(
    levels: MutableMapping[str, float],
    context: Mapping[str, Any],
    style: str,
) -> None:
    """
    Extend the base level map with higher timeframe structure relevant to the plan style.

    Args:
        levels: Mutable mapping of level tags to prices (modified in place).
        context: Mapping containing optional `levels_daily`, `levels_weekly`,
            `vol_profile_daily`, `vol_profile_weekly` sequences/dicts with price data.
        style: Normalised plan style token.
    """

    style_norm = (style or "intraday").strip().lower()
    include_daily = style_norm in {"intraday", "swing", "leaps"}
    include_weekly = style_norm in {"swing", "leaps"}

    if include_daily:
        for tag, price in _ensure_sequence(context.get("levels_daily")):
            tag_norm = str(tag or "").upper()
            if tag_norm.endswith("HIGH"):
                _safe_assign(levels, "daily_high", price)
            elif tag_norm.endswith("LOW"):
                _safe_assign(levels, "daily_low", price)
            elif tag_norm.endswith("CLOSE"):
                _safe_assign(levels, "daily_close", price)
        profile = context.get("vol_profile_daily") or {}
        if isinstance(profile, Mapping):
            _safe_assign(levels, "daily_vah", profile.get("VAH"))
            _safe_assign(levels, "daily_val", profile.get("VAL"))
            _safe_assign(levels, "daily_poc", profile.get("POC"))

    if include_weekly:
        for tag, price in _ensure_sequence(context.get("levels_weekly")):
            tag_norm = str(tag or "").upper()
            if tag_norm.endswith("HIGH"):
                _safe_assign(levels, "weekly_high", price)
            elif tag_norm.endswith("LOW"):
                _safe_assign(levels, "weekly_low", price)
            elif tag_norm.endswith("CLOSE"):
                _safe_assign(levels, "weekly_close", price)
        profile = context.get("vol_profile_weekly") or {}
        if isinstance(profile, Mapping):
            _safe_assign(levels, "weekly_vah", profile.get("VAH"))
            _safe_assign(levels, "weekly_val", profile.get("VAL"))
            _safe_assign(levels, "weekly_poc", profile.get("POC"))


def _ensure_sequence(value: Any) -> Sequence[Tuple[Any, Any]]:
    if isinstance(value, Sequence):
        return value  # type: ignore[return-value]
    return []

