from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Literal

import pandas as pd

from ..data_sources import fetch_polygon_ohlcv
from ..app.engine.index_common import (
    DEFAULT_INDEX_RATIOS,
    is_index_symbol,
    resolve_polygon_symbol,
    resolve_proxy_symbol,
    scale_threshold,
)

logger = logging.getLogger(__name__)

DataMode = Literal["live", "lkg"]

_TIMEFRAMES: list[tuple[str, str, int]] = [
    ("1d", "1d", 240),
    ("65m", "65", 45),
    ("15m", "15", 20),
    ("5m", "5", 10),
]

_EXTENDED_AWARE_FRAMES = {"1m", "5m", "15m", "30m"}


class ScaleMismatchError(RuntimeError):
    """Raised when index data fails scale validation."""

    def __init__(self, symbol: str, close_value: float, threshold: float, path: str) -> None:
        message = (
            f"{symbol} close {close_value:.2f} below threshold {threshold:.2f} "
            f"during {path} fetch"
        )
        super().__init__(message)
        self.symbol = symbol
        self.close_value = float(close_value)
        self.threshold = float(threshold)
        self.path = path


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


def _valid_index_close(symbol: str, close_value: float) -> bool:
    threshold = scale_threshold(symbol)
    if threshold is None:
        return True
    try:
        return float(close_value) >= float(threshold)
    except (TypeError, ValueError):
        return False


async def _load_equity_frame(
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


def _apply_ratio(frame: pd.DataFrame, ratio: float) -> pd.DataFrame:
    scaled = frame.copy()
    for column in ("open", "high", "low", "close"):
        if column in scaled.columns:
            scaled[column] = scaled[column].astype(float) * ratio
    if "volume" in scaled.columns:
        try:
            scaled["volume"] = scaled["volume"].astype(float)
        except Exception:
            scaled["volume"] = 0.0
    else:
        scaled["volume"] = 0.0
    return scaled


async def _load_index_frame(
    symbol: str,
    timeframe: str,
    max_days: int,
    cutoff_ts: pd.Timestamp,
    *,
    extended: bool,
    label: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    include_extended = extended and label in _EXTENDED_AWARE_FRAMES
    base = symbol.upper()
    polygon_symbol = resolve_polygon_symbol(base)
    polygon_close: float | None = None
    if polygon_symbol:
        frame = await fetch_polygon_ohlcv(
            polygon_symbol,
            timeframe,
            max_days=max_days,
            include_extended=include_extended,
        )
        if frame is not None and not frame.empty:
            filtered = frame.loc[frame.index <= cutoff_ts]
            if not filtered.empty:
                try:
                    polygon_close = float(filtered["close"].iloc[-1])
                except Exception:
                    polygon_close = None
                if polygon_close is not None and _valid_index_close(base, polygon_close):
                    return filtered, {"path": "index_polygon"}

    proxy_symbol = resolve_proxy_symbol(base)
    if proxy_symbol:
        proxy_frame = await fetch_polygon_ohlcv(
            proxy_symbol,
            timeframe,
            max_days=max_days,
            include_extended=include_extended,
        )
        if proxy_frame is not None and not proxy_frame.empty:
            filtered = proxy_frame.loc[proxy_frame.index <= cutoff_ts]
            if not filtered.empty:
                ratio = DEFAULT_INDEX_RATIOS.get(base)
                if ratio:
                    scaled = _apply_ratio(filtered, float(ratio))
                    try:
                        proxy_close = float(scaled["close"].iloc[-1])
                    except Exception:
                        proxy_close = None
                    if proxy_close is not None and _valid_index_close(base, proxy_close):
                        return scaled, {"path": "index_synthetic_from_proxy", "ratio": float(ratio), "proxy": proxy_symbol}
                    if proxy_close is not None:
                        raise ScaleMismatchError(base, proxy_close, scale_threshold(base) or 0.0, "index_synthetic_from_proxy")

    if polygon_close is not None:
        raise ScaleMismatchError(base, polygon_close, scale_threshold(base) or 0.0, "index_polygon")
    raise RuntimeError(f"index data unavailable for {base}")


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
        base = symbol.upper()
        is_index = is_index_symbol(base)
        path_used: set[str] = set()
        proxy_meta: dict[str, object] = {}
        for label, timeframe, max_days in _TIMEFRAMES:
            try:
                if is_index:
                    frame, meta = await _load_index_frame(
                        base,
                        timeframe,
                        max_days,
                        cutoff_ts,
                        extended=extended,
                        label=label,
                    )
                    path_used.add(str(meta.get("path", "index_unknown")))
                    proxy_meta.update(meta)
                else:
                    frame = await _load_equity_frame(
                        symbol,
                        timeframe,
                        max_days,
                        cutoff_ts,
                        extended=extended,
                        label=label,
                    )
                    path_used.add("equity_polygon")
            except Exception as exc:
                logger.warning("series fetch failed for %s/%s: %s", symbol, timeframe, exc)
                if is_index:
                    raise
                if label == "1d":
                    frame = _synthetic_frame(symbol, as_of_utc)
                    path_used.add("equity_path_fallback")
                else:
                    continue
            per_symbol[label] = frame
        if not per_symbol:
            raise RuntimeError(f"no frames loaded for {symbol}")
        if "1d" not in per_symbol:
            if is_index:
                raise RuntimeError(f"missing daily frame for index symbol {symbol}")
            per_symbol["1d"] = _synthetic_frame(symbol, as_of_utc)
        bundle.frames[symbol] = per_symbol
        daily_frame = per_symbol["1d"]
        try:
            latest_close = float(daily_frame["close"].iloc[-1])
            if is_index:
                threshold = scale_threshold(base)
                if threshold and latest_close < threshold:
                    path_label = ",".join(sorted(path_used)) or "unknown"
                    raise ScaleMismatchError(base, latest_close, threshold, path_label)
            bundle.latest_close[symbol] = latest_close
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("unable to extract latest close for %s: %s", symbol, exc)
        if is_index:
            log_payload = {
                "symbol": base,
                "paths": sorted(path_used),
                "proxy": proxy_meta.get("proxy"),
                "ratio": proxy_meta.get("ratio"),
            }
            logger.info("index_series_path", extra=log_payload)

    await asyncio.gather(*[_load_symbol(symbol) for symbol in bundle.symbols])
    return bundle


__all__ = ["SeriesBundle", "fetch_series", "DataMode", "ScaleMismatchError"]
