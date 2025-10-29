"""Higher timeframe structural level extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd


@dataclass
class HTFLevels:
    pdh: Optional[float]
    pdl: Optional[float]
    pdc: Optional[float]
    pwh: Optional[float]
    pwl: Optional[float]
    pwc: Optional[float]
    vah: Optional[float] = None
    val: Optional[float] = None
    poc: Optional[float] = None


def _extract_previous_row(frame: Optional[pd.DataFrame]) -> Optional[pd.Series]:
    if frame is None or frame.empty:
        return None
    frame_sorted = frame.sort_index()
    if len(frame_sorted) < 2:
        return None
    return frame_sorted.iloc[-2]


def _extract_optional(frame: Optional[pd.DataFrame], column: str) -> Optional[float]:
    if frame is None or frame.empty or column not in frame.columns:
        return None
    series = frame[column].dropna()
    if series.empty:
        return None
    return float(series.iloc[-1])


def compute_htf_levels(daily_bars: Optional[pd.DataFrame], weekly_bars: Optional[pd.DataFrame]) -> HTFLevels:
    """Compute higher timeframe reference levels without external requests."""

    prev_daily = _extract_previous_row(daily_bars)
    prev_weekly = _extract_previous_row(weekly_bars)

    pdh = float(prev_daily["high"]) if prev_daily is not None and "high" in prev_daily else None
    pdl = float(prev_daily["low"]) if prev_daily is not None and "low" in prev_daily else None
    pdc = float(prev_daily["close"]) if prev_daily is not None and "close" in prev_daily else None

    pwh = float(prev_weekly["high"]) if prev_weekly is not None and "high" in prev_weekly else None
    pwl = float(prev_weekly["low"]) if prev_weekly is not None and "low" in prev_weekly else None
    pwc = float(prev_weekly["close"]) if prev_weekly is not None and "close" in prev_weekly else None

    vah = _extract_optional(daily_bars, "vah")
    val = _extract_optional(daily_bars, "val")
    poc = _extract_optional(daily_bars, "poc")

    return HTFLevels(
        pdh=pdh,
        pdl=pdl,
        pdc=pdc,
        pwh=pwh,
        pwl=pwl,
        pwc=pwc,
        vah=vah,
        val=val,
        poc=poc,
    )


def _last_completed_row(frame: Optional[pd.DataFrame]) -> Optional[pd.Series]:
    if frame is None or frame.empty:
        return None
    frame_sorted = frame.sort_index()
    if len(frame_sorted) < 2:
        return None
    return frame_sorted.iloc[-2]


def compute_intraday_htf_levels(
    bars_60m: Optional[pd.DataFrame],
    bars_240m: Optional[pd.DataFrame],
) -> Dict[str, float]:
    """Return H1/H4 highs, lows, and floor pivots for last completed bars."""

    out: Dict[str, float] = {}

    def _hlc(row: Optional[pd.Series]) -> tuple[Optional[float], Optional[float], Optional[float]]:
        if row is None:
            return None, None, None
        try:
            return float(row["high"]), float(row["low"]), float(row["close"])
        except Exception:
            return None, None, None

    def _pivot_triplet(
        high: Optional[float], low: Optional[float], close: Optional[float]
    ) -> tuple[Optional[float], Optional[float], Optional[float]]:
        if high is None or low is None or close is None:
            return None, None, None
        pivot = (high + low + close) / 3.0
        r1 = 2 * pivot - low
        s1 = 2 * pivot - high
        return pivot, r1, s1

    h1_row = _last_completed_row(bars_60m)
    h1_high, h1_low, h1_close = _hlc(h1_row)
    if h1_high is not None:
        out["h1_high"] = h1_high
    if h1_low is not None:
        out["h1_low"] = h1_low
    pivot, r1, s1 = _pivot_triplet(h1_high, h1_low, h1_close)
    if pivot is not None:
        out["h1_pivot"] = pivot
    if r1 is not None:
        out["h1_r1"] = r1
    if s1 is not None:
        out["h1_s1"] = s1

    h4_row = _last_completed_row(bars_240m)
    h4_high, h4_low, h4_close = _hlc(h4_row)
    if h4_high is not None:
        out["h4_high"] = h4_high
    if h4_low is not None:
        out["h4_low"] = h4_low
    pivot, r1, s1 = _pivot_triplet(h4_high, h4_low, h4_close)
    if pivot is not None:
        out["h4_pivot"] = pivot
    if r1 is not None:
        out["h4_r1"] = r1
    if s1 is not None:
        out["h4_s1"] = s1

    return out


__all__ = ["HTFLevels", "compute_htf_levels", "compute_intraday_htf_levels"]
