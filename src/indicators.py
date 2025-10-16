"""Cached indicator utilities used by scan feature extraction."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from .calculations import atr, ema, vwap

_TTL_SECONDS = 60.0
_CACHE: Dict[Tuple[str, float], Tuple[float, Dict[str, Any]]] = {}


def get_indicator_bundle(symbol: str, history: pd.DataFrame) -> Dict[str, Any]:
    """Return a cached indicator snapshot for the latest bar."""
    if history is None or history.empty:
        return {}
    last_ts = history.index[-1]
    if isinstance(last_ts, pd.Timestamp):
        stamp = float(last_ts.timestamp())
    else:
        stamp = float(pd.Timestamp(last_ts).timestamp())
    key = (symbol.upper(), stamp)
    now = time.monotonic()
    cached = _CACHE.get(key)
    if cached and now - cached[0] < _TTL_SECONDS:
        return dict(cached[1])
    bundle = _build_bundle(history)
    _CACHE[key] = (now, bundle)
    return dict(bundle)


def _build_bundle(history: pd.DataFrame) -> Dict[str, Any]:
    closes = history["close"]
    highs = history.get("high", closes)
    lows = history.get("low", closes)
    bundle: Dict[str, Any] = {}

    ema_fast = _last_value(ema(closes, 9))
    ema_mid = _last_value(ema(closes, 21))
    ema_slow = _last_value(ema(closes, 55))
    ema_long = _last_value(ema(closes, 200))
    ordering = _ema_ordering(ema_fast, ema_mid, ema_slow, ema_long)
    bundle["ema_stack"] = {"ordering": ordering}
    bundle["ema_alignment"] = _ema_alignment(ordering)

    atr_series = atr(highs, lows, closes, period=14)
    atr_val = _last_value(atr_series)
    bundle["atr"] = atr_val
    last_close = float(closes.iloc[-1]) if not closes.empty else 0.0
    bundle["atr_norm"] = _safe_div(atr_val, last_close)
    last_range = float(highs.iloc[-1] - lows.iloc[-1]) if not history.empty else 0.0
    bundle["atr_multiple"] = _safe_div(last_range, atr_val)

    bundle["vwap"] = _last_value(vwap(history))
    bundle["impulse"] = _impulse(closes)
    bundle["htf"] = _htf_hint(history)
    return bundle


def _last_value(series: pd.Series) -> float:
    if series is None or series.empty:
        return 0.0
    value = float(series.iloc[-1])
    if np.isnan(value):
        return 0.0
    return value


def _ema_ordering(ema_fast: float, ema_mid: float, ema_slow: float, ema_long: float) -> list[str]:
    stack = [
        ("ema9", ema_fast),
        ("ema21", ema_mid),
        ("ema55", ema_slow),
        ("ema200", ema_long),
    ]
    stack.sort(key=lambda item: item[1], reverse=True)
    return [item[0] for item in stack]


def _ema_alignment(ordering: list[str]) -> float:
    desired = ["ema9", "ema21", "ema55", "ema200"]
    if ordering[: len(desired)] == desired:
        return 1.0
    if ordering[0:2] == ["ema9", "ema21"]:
        return 0.7
    return 0.4


def _impulse(closes: pd.Series) -> float:
    if closes.empty or len(closes) < 3:
        return 0.5
    recent = closes.iloc[-3:]
    delta = float(recent.iloc[-1] - recent.iloc[0])
    base = _safe_div(abs(delta), abs(float(recent.iloc[0])) + 1e-6)
    return max(0.0, min(base * 5.0, 1.0))


def _htf_hint(history: pd.DataFrame) -> Dict[str, Any]:
    closes = history["close"]
    if closes.empty:
        return {}
    window = closes.tail(60)
    slope = np.polyfit(range(len(window)), window.values, 1)[0] if len(window) >= 5 else 0.0
    slope_norm = _safe_div(slope, float(window.iloc[-1]))
    structure = 0.7 if slope_norm > 0 else 0.4 if slope_norm < 0 else 0.5
    momentum = max(0.0, min(abs(slope_norm) * 80, 1.0))
    return {
        "structure_d1": structure,
        "confluence_d1": 0.6,
        "momentum": momentum,
        "confluence": 0.55,
        "structure_w1": structure * 0.9 + 0.05,
    }


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


__all__ = ["get_indicator_bundle"]
