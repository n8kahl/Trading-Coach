"""Quantitative statistics (MFE/MAE, expected move) used for target sizing."""

from __future__ import annotations

import asyncio
import io
import math
import time
import zlib
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .data_sources import fetch_polygon_ohlcv
from .calculations import atr


STYLE_CONFIG: Dict[str, Dict[str, float | int | str]] = {
    "scalp": {
        "timeframe": "1",        # minutes
        "horizon_minutes": 60,
        "lookback_days": 5,
        "min_samples": 150,
    },
    "intraday": {
        "timeframe": "5",
        "horizon_minutes": 360,  # full session
        "lookback_days": 15,
        "min_samples": 180,
    },
    "swing": {
        "timeframe": "60",
        "horizon_minutes": 60 * 24 * 5,  # ~5 trading days
        "lookback_days": 120,
        "min_samples": 180,
    },
    "leaps": {
        "timeframe": "240",
        "horizon_minutes": 60 * 24 * 20,
        "lookback_days": 365,
        "min_samples": 160,
    },
}

_CACHE: Dict[Tuple[str, str, str], Tuple[float, Dict[str, object]]] = {}
_CACHE_TTL = 30 * 60  # seconds
_LOCK = asyncio.Lock()


def _compress_array(values: np.ndarray) -> bytes:
    if values is None or values.size == 0:
        return b""
    buffer = io.BytesIO()
    np.save(buffer, values.astype(np.float32, copy=False), allow_pickle=False)
    return zlib.compress(buffer.getvalue(), level=3)


def _decompress_array(blob: bytes) -> np.ndarray:
    if not blob:
        return np.array([], dtype=np.float32)
    try:
        payload = zlib.decompress(blob)
    except zlib.error:  # pragma: no cover - defensive
        return np.array([], dtype=np.float32)
    buffer = io.BytesIO(payload)
    return np.load(buffer, allow_pickle=False)


def _pack_stats_for_cache(stats: Dict[str, object]) -> Dict[str, object]:
    packed = dict(stats)
    for side in ("long", "short"):
        segment = dict(packed.get(side) or {})
        array = segment.pop("mfe", None)
        if array is None:
            normalized = np.array([], dtype=np.float32)
        else:
            normalized = np.asarray(array, dtype=np.float32)
        segment["mfe_bytes"] = _compress_array(normalized)
        segment["mfe_count"] = int(normalized.size)
        packed[side] = segment
    return packed


def _unpack_stats_from_cache(payload: Dict[str, object]) -> Dict[str, object]:
    restored = dict(payload)
    for side in ("long", "short"):
        segment = dict(restored.get(side) or {})
        blob = segment.pop("mfe_bytes", b"")
        segment["mfe"] = _decompress_array(blob)
        restored[side] = segment
    return restored


def _tf_minutes(token: str) -> int:
    token = token.upper()
    if token == "D":
        return 390  # RTH minutes
    try:
        return int(token)
    except ValueError:
        return 5


def _compute_mfe(frame: pd.DataFrame, horizon_bars: int) -> Tuple[np.ndarray, np.ndarray]:
    closes = frame["close"].to_numpy()
    highs = frame["high"].to_numpy()
    lows = frame["low"].to_numpy()
    total = closes.shape[0]
    if total <= horizon_bars + 1:
        return np.array([]), np.array([])

    long_mfe: list[float] = []
    short_mfe: list[float] = []
    for idx in range(total - horizon_bars):
        entry = closes[idx]
        if entry <= 0:
            continue
        horizon_slice = slice(idx, idx + horizon_bars + 1)
        high = float(np.max(highs[horizon_slice]))
        low = float(np.min(lows[horizon_slice]))
        if math.isfinite(high):
            long_mfe.append((high - entry) / entry)
        if math.isfinite(low):
            short_mfe.append((entry - low) / entry)
    return np.array(long_mfe, dtype=float), np.array(short_mfe, dtype=float)


def _quantiles(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {}
    percentiles = [0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
    output: Dict[str, float] = {}
    for pct in percentiles:
        output[f"q{int(pct * 100)}"] = float(np.quantile(values, pct))
    return output


def _expected_move_fallback(atr_value: float, horizon_minutes: float, tf_minutes: float) -> Optional[float]:
    if not atr_value or atr_value <= 0 or horizon_minutes <= 0 or tf_minutes <= 0:
        return None
    steps = max(horizon_minutes / tf_minutes, 1.0)
    return float(atr_value * math.sqrt(steps))


async def _compute_stats(symbol: str, style: str) -> Optional[Dict[str, object]]:
    cfg = STYLE_CONFIG.get(style)
    if not cfg:
        return None

    timeframe = str(cfg["timeframe"])
    horizon_minutes = float(cfg["horizon_minutes"])
    lookback_days = int(cfg["lookback_days"])
    min_samples = int(cfg["min_samples"])
    tf_minutes = _tf_minutes(timeframe)
    horizon_bars = max(int(round(horizon_minutes / max(tf_minutes, 1))), 1)

    frame = await fetch_polygon_ohlcv(symbol, timeframe, max_days=lookback_days)
    if frame is None or frame.empty:
        return None
    frame = frame.sort_index()
    frame = frame.dropna(subset=["open", "high", "low", "close"])

    long_mfe, short_mfe = _compute_mfe(frame, horizon_bars)
    if long_mfe.size < min_samples or short_mfe.size < min_samples:
        return None
    long_mfe = long_mfe.astype(np.float32, copy=False)
    short_mfe = short_mfe.astype(np.float32, copy=False)

    atr_series = atr(frame["high"], frame["low"], frame["close"], 14)
    atr_mean = float(atr_series.dropna().tail(100).mean()) if not atr_series.empty else None
    em_fallback = _expected_move_fallback(atr_mean, horizon_minutes, tf_minutes)

    stats = {
        "style": style,
        "timeframe": timeframe,
        "horizon_minutes": horizon_minutes,
        "long": {
            "mfe": long_mfe,
            "quantiles": _quantiles(long_mfe),
        },
        "short": {
            "mfe": short_mfe,
            "quantiles": _quantiles(short_mfe),
        },
        "atr": atr_mean,
        "expected_move": em_fallback,
    }
    return stats


async def get_style_stats(symbol: str, style: str, *, as_of: str | None = None) -> Optional[Dict[str, object]]:
    """Return cached statistical profile for symbol/style."""
    token = as_of or "live"
    key = (symbol.upper(), style, token)
    async with _LOCK:
        entry = _CACHE.get(key)
        now = time.monotonic()
        if entry and now - entry[0] < _CACHE_TTL:
            cached_payload = entry[1]
            return _unpack_stats_from_cache(cached_payload)

    stats = await _compute_stats(symbol, style)
    async with _LOCK:
        if stats:
            _CACHE[key] = (time.monotonic(), _pack_stats_for_cache(stats))
        else:
            _CACHE.pop(key, None)
    return stats


def estimate_probability(mfe_values: np.ndarray, distance: float) -> Optional[float]:
    """Return empirical probability that MFE exceeds `distance` (distance expressed as % in decimals)."""
    if distance is None or distance <= 0 or mfe_values.size == 0:
        return None
    hits = np.count_nonzero(mfe_values >= distance)
    return hits / float(mfe_values.size)
