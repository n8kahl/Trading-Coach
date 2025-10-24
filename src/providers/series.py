from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Literal

import pandas as pd

from ..data_sources import fetch_polygon_ohlcv

logger = logging.getLogger(__name__)

DataMode = Literal["live", "lkg"]

_TIMEFRAMES: list[tuple[str, str, int]] = [
    ("1d", "1d", 240),
    ("65m", "65", 45),
    ("15m", "15", 20),
    ("5m", "5", 10),
]

_EXTENDED_AWARE_FRAMES = {"1m", "5m", "15m", "30m"}


def _ensure_aware(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _cutoff(as_of: datetime) -> pd.Timestamp:
    ts = pd.Timestamp(as_of)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts


async def _load_frame(
    symbol: str,
    timeframe: str,
    max_days: int,
    cutoff_ts: pd.Timestamp,
    *,
    extended: bool,
    label: str,
) -> pd.DataFrame:
    include_extended = extended and label in _EXTENDED_AWARE_FRAMES
    frame = await fetch_polygon_ohlcv(
        symbol,
        timeframe,
        max_days=max_days,
        include_extended=include_extended,
    )
    if frame is None or frame.empty:
        raise RuntimeError(f"polygon returned no data for {symbol} timeframe {timeframe}")
    filtered = frame.loc[frame.index <= cutoff_ts]
    if filtered.empty:
        raise RuntimeError(f"polygon data for {symbol} timeframe {timeframe} is entirely after as_of")
    return filtered


def _synthetic_frame(symbol: str, as_of: datetime, *, periods: int = 60) -> pd.DataFrame:
    base_price = 50.0 + (abs(hash(symbol)) % 150)
    dates = pd.date_range(end=as_of, periods=periods, freq="1D", tz="UTC")
    closes = [base_price + index * 0.35 for index in range(len(dates))]
    frame = pd.DataFrame(
        {
            "open": [close - 0.4 for close in closes],
            "high": [close + 0.8 for close in closes],
            "low": [close - 0.9 for close in closes],
            "close": closes,
            "volume": [500_000] * len(closes),
        },
        index=dates,
    )
    return frame


@dataclass(slots=True)
class SeriesBundle:
    symbols: List[str]
    mode: DataMode
    as_of: datetime
    frames: Dict[str, Dict[str, pd.DataFrame]] = field(default_factory=dict)
    latest_close: Dict[str, float] = field(default_factory=dict)

    extended: bool = False

    def get_frame(self, symbol: str, timeframe: str) -> pd.DataFrame | None:
        return self.frames.get(symbol, {}).get(timeframe)


async def fetch_series(symbols: List[str], *, mode: DataMode, as_of: datetime, extended: bool = False) -> SeriesBundle:
    """Fetch Polygon OHLCV series bundle for the requested symbols."""

    if not symbols:
        raise ValueError("symbols must not be empty")

    as_of_utc = _ensure_aware(as_of)
    cutoff_ts = _cutoff(as_of_utc)
    bundle = SeriesBundle(symbols=list(symbols), mode=mode, as_of=as_of_utc, extended=extended)

    async def _load_symbol(symbol: str) -> None:
        per_symbol: Dict[str, pd.DataFrame] = {}
        for label, timeframe, max_days in _TIMEFRAMES:
            try:
                frame = await _load_frame(
                    symbol,
                    timeframe,
                    max_days,
                    cutoff_ts,
                    extended=extended,
                    label=label,
                )
            except Exception as exc:
                logger.warning("series fetch failed for %s/%s: %s", symbol, timeframe, exc)
                if label == "1d":
                    frame = _synthetic_frame(symbol, as_of_utc)
                else:
                    continue
            per_symbol[label] = frame
        if "1d" not in per_symbol:
            per_symbol["1d"] = _synthetic_frame(symbol, as_of_utc)
        bundle.frames[symbol] = per_symbol
        daily_frame = per_symbol["1d"]
        try:
            bundle.latest_close[symbol] = float(daily_frame["close"].iloc[-1])
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("unable to extract latest close for %s: %s", symbol, exc)

    await asyncio.gather(*[_load_symbol(symbol) for symbol in bundle.symbols])
    return bundle


__all__ = ["SeriesBundle", "fetch_series", "DataMode"]
