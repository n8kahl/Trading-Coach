"""Deterministic estimation of expected trade duration."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from ..strategy_library import normalize_style_input


_STYLE_DEFAULT_INTERVAL_MINUTES: Dict[str, int] = {
    "scalp": 5,
    "intraday": 15,
    "swing": 60,
    "leaps": 390,
}

_STYLE_DEFAULT_BARS: Dict[str, int] = {
    "scalp": 3,
    "intraday": 12,
    "swing": 24,
    "leaps": 120,
}

_STYLE_MINUTES_BOUNDS: Dict[str, Tuple[int, int]] = {
    "scalp": (10, 90),
    "intraday": (45, 360),
    "swing": (390, 390 * 10),
    "leaps": (390 * 10, 390 * 60),
}

_ATR_DIVISORS: Dict[str, float] = {
    "scalp": 4.0,
    "intraday": 6.0,
    "swing": 12.0,
    "leaps": 18.0,
}

_REALIZED_SCALE: Dict[str, float] = {
    "scalp": 0.65,
    "intraday": 0.6,
    "swing": 0.55,
    "leaps": 0.5,
}

_EM_FRACTION: Dict[str, float] = {
    "scalp": 0.25,
    "intraday": 0.2,
    "swing": 0.15,
    "leaps": 0.1,
}

_DISTANCE_FRACTION: Dict[str, float] = {
    "scalp": 0.35,
    "intraday": 0.25,
    "swing": 0.18,
    "leaps": 0.12,
}

_STYLE_MIN_PER_BAR: Dict[str, float] = {
    "scalp": 0.02,
    "intraday": 0.05,
    "swing": 0.1,
    "leaps": 0.15,
}


def _normalize_style(style: Optional[str]) -> str:
    token = normalize_style_input(style) or "intraday"
    if token == "leap":
        return "leaps"
    return token


def _parse_interval_minutes(interval_hint: Optional[str]) -> Optional[int]:
    if not interval_hint:
        return None
    token = str(interval_hint).strip().lower()
    if not token:
        return None
    if token.endswith("min"):
        token = token[:-3]
    if token.endswith("m"):
        token = token[:-1]
        if token.isdigit():
            return max(int(token), 1)
        return None
    if token.endswith("h"):
        token = token[:-1]
        if token.isdigit():
            return max(int(token) * 60, 1)
        return None
    if token in {"d", "1d", "day"}:
        return 390
    if token.isdigit():
        return max(int(token), 1)
    return None


def _format_interval(minutes: int) -> str:
    if minutes >= 390:
        return "1D"
    if minutes % 60 == 0 and minutes >= 60:
        return f"{minutes // 60}h"
    return f"{minutes}m"


def _pick_bars(
    minutes: int,
    bars_5m: Optional[pd.DataFrame],
    bars_15m: Optional[pd.DataFrame],
    bars_60m: Optional[pd.DataFrame],
) -> Tuple[Optional[pd.DataFrame], Optional[int]]:
    if minutes <= 7 and bars_5m is not None and not getattr(bars_5m, "empty", True):
        return bars_5m, 5
    if minutes <= 20 and bars_15m is not None and not getattr(bars_15m, "empty", True):
        return bars_15m, 15
    if bars_60m is not None and not getattr(bars_60m, "empty", True):
        return bars_60m, 60
    if bars_15m is not None and not getattr(bars_15m, "empty", True):
        return bars_15m, 15
    if bars_5m is not None and not getattr(bars_5m, "empty", True):
        return bars_5m, 5
    return None, None


def _safe_distance(entry: Optional[float], tp1: Optional[float]) -> Optional[float]:
    if entry is None or tp1 is None:
        return None
    try:
        a = float(entry)
        b = float(tp1)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(a) or not math.isfinite(b):
        return None
    return abs(b - a)


def _realized_component(
    style: str,
    interval_minutes: int,
    bars: Optional[pd.DataFrame],
    source_minutes: Optional[int],
) -> Optional[float]:
    if bars is None or getattr(bars, "empty", True):
        return None
    source_minutes = source_minutes or interval_minutes
    if "high" in bars.columns and "low" in bars.columns:
        series = pd.to_numeric(bars["high"], errors="coerce") - pd.to_numeric(bars["low"], errors="coerce")
    elif "close" in bars.columns:
        closes = pd.to_numeric(bars["close"], errors="coerce")
        series = closes.diff().abs()
    else:
        return None
    series = series.replace([math.inf, -math.inf], pd.NA).dropna()
    if series.empty:
        return None
    window = series.iloc[-min(len(series), 40) :]
    avg_range = float(window.mean())
    if not math.isfinite(avg_range) or avg_range <= 0:
        return None
    scale = interval_minutes / max(source_minutes, 1)
    avg_range *= max(scale, 0.5)
    component = avg_range * _REALIZED_SCALE.get(style, 0.6)
    return component if component > 0 else None


def _atr_component(style: str, atr: Optional[float]) -> Optional[float]:
    if atr is None:
        return None
    try:
        atr_val = float(atr)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(atr_val) or atr_val <= 0:
        return None
    divisor = _ATR_DIVISORS.get(style, 6.0)
    component = atr_val / divisor
    return component if component > 0 else None


def _em_cap(style: str, em: Optional[float]) -> Optional[float]:
    if em is None:
        return None
    try:
        em_val = float(em)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(em_val) or em_val <= 0:
        return None
    fraction = _EM_FRACTION.get(style, 0.2)
    return em_val * fraction


def _distance_component(style: str, distance: float) -> float:
    fraction = _DISTANCE_FRACTION.get(style, 0.25)
    floor = _STYLE_MIN_PER_BAR.get(style, 0.05)
    return max(distance * fraction, floor)


def _style_label(style: str, minutes: int) -> str:
    minutes = max(minutes, 1)
    if style == "scalp":
        if minutes <= 20:
            return "scalp ~10–20m"
        if minutes <= 45:
            return "scalp ~20–45m"
        return "scalp ~45–90m"
    if style == "intraday":
        hours = minutes / 60.0
        if hours <= 1.5:
            return "intraday ~1–2h"
        if hours <= 3.5:
            return "intraday ~2–4h"
        return "intraday ~4–6h"
    days = minutes / 390.0
    if style == "swing":
        if days <= 3:
            return "swing ~1–3d"
        if days <= 6:
            return "swing ~3–6d"
        return "swing ~6–10d"
    # leaps
    if days <= 20:
        return "leaps ~10–20 trading days"
    if days <= 40:
        return "leaps ~20–40 trading days"
    return "leaps ~40–60 trading days"


def estimate_expected_duration(
    style: str,
    interval_hint: Optional[str],
    entry: float,
    tp1: float,
    atr: Optional[float],
    em: Optional[float],
    bars_5m: Optional[pd.DataFrame],
    bars_15m: Optional[pd.DataFrame],
    bars_60m: Optional[pd.DataFrame],
) -> Dict[str, Any]:
    style_token = _normalize_style(style)
    interval_minutes = _parse_interval_minutes(interval_hint)
    if interval_minutes is None:
        interval_minutes = _STYLE_DEFAULT_INTERVAL_MINUTES.get(style_token, 15)
    interval_label = _format_interval(interval_minutes)
    bars_frame, source_minutes = _pick_bars(interval_minutes, bars_5m, bars_15m, bars_60m)
    distance = _safe_distance(entry, tp1)
    if distance is None or distance <= 0:
        default_minutes = _STYLE_DEFAULT_INTERVAL_MINUTES.get(style_token, 15) * _STYLE_DEFAULT_BARS.get(style_token, 8)
        minutes_default = max(default_minutes, _STYLE_MINUTES_BOUNDS.get(style_token, (15, 240))[0])
        label = _style_label(style_token, minutes_default)
        return {
            "minutes": int(minutes_default),
            "label": label,
            "basis": ["Default"],
            "inputs": {
                "interval": interval_label,
                "atr": float(atr) if isinstance(atr, (int, float)) and math.isfinite(float(atr)) else None,
                "em": float(em) if isinstance(em, (int, float)) and math.isfinite(float(em)) else None,
                "distance": None,
            },
        }

    components: list[float] = []
    basis: list[str] = []

    atr_component = _atr_component(style_token, atr)
    if atr_component:
        components.append(atr_component)
        basis.append("ATR")

    realized_component = _realized_component(style_token, interval_minutes, bars_frame, source_minutes)
    if realized_component:
        components.append(realized_component)
        basis.append("RealizedRange")

    distance_component = _distance_component(style_token, distance)
    components.append(distance_component)
    basis.append("Distance")

    per_bar_move = min(components) if components else distance_component

    em_cap = _em_cap(style_token, em)
    if em_cap and atr_component and (atr_component > em_cap):
        per_bar_move = min(per_bar_move, em_cap)
        basis.append("EM")

    per_bar_floor = _STYLE_MIN_PER_BAR.get(style_token, 0.05)
    per_bar_move = max(per_bar_move, per_bar_floor)
    bars_estimate = max(1, int(round(distance / per_bar_move)))

    if bars_estimate <= 0 or not math.isfinite(bars_estimate):
        bars_estimate = _STYLE_DEFAULT_BARS.get(style_token, 8)

    minutes_estimate = bars_estimate * interval_minutes
    if style_token in {"swing", "leaps"}:
        trading_minutes = 390
        days = minutes_estimate / trading_minutes
        if style_token == "swing":
            days = max(days, 1.0)
        else:
            days = max(days, 10.0)
        minutes_estimate = int(round(days * trading_minutes))

    min_bound, max_bound = _STYLE_MINUTES_BOUNDS.get(style_token, (15, 360))
    minutes_estimate = max(min_bound, min(minutes_estimate, max_bound))

    label = _style_label(style_token, minutes_estimate)
    basis_unique = list(dict.fromkeys(basis))

    result = {
        "minutes": int(minutes_estimate),
        "label": label,
        "basis": basis_unique,
        "inputs": {
            "interval": interval_label,
            "atr": float(atr) if isinstance(atr, (int, float)) and math.isfinite(float(atr)) else None,
            "em": float(em) if isinstance(em, (int, float)) and math.isfinite(float(em)) else None,
            "distance": round(distance, 4),
            "bars": bars_estimate,
            "per_bar": round(per_bar_move, 4),
        },
    }
    return result


__all__ = ["estimate_expected_duration"]
