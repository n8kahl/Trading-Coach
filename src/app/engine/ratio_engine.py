"""Gamma ratio engine bridging index levels to ETF proxies."""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ...data_sources import fetch_polygon_ohlcv
from .index_common import ETF_PROXIES, POLYGON_INDEX_TICKERS


@dataclass(slots=True)
class RatioSnapshot:
    index_symbol: str
    proxy_symbol: str
    gamma_current: float
    gamma_mean: float
    spot_ratio: float
    drift: float
    samples: int
    updated_at: pd.Timestamp
    window_minutes: int

    def translate_level(self, level: float) -> float:
        """Translate an index level to the ETF proxy using the current gamma."""
        ratio = self.spot_ratio
        if self.gamma_mean and math.isfinite(self.gamma_mean):
            adjustment = self.gamma_current / self.gamma_mean if self.gamma_mean else 1.0
            ratio *= adjustment if math.isfinite(adjustment) else 1.0
        return float(level) * ratio


class RatioEngine:
    """Maintain rolling gamma between index and ETF proxies."""

    def __init__(self, *, lookback_minutes: int = 60, refresh_seconds: int = 120) -> None:
        self.lookback_minutes = max(lookback_minutes, 15)
        self.refresh_seconds = max(refresh_seconds, 30)
        self._snapshot_cache: Dict[Tuple[str, str], RatioSnapshot] = {}
        self._locks: Dict[Tuple[str, str], asyncio.Lock] = {}

    async def snapshot(self, index_symbol: str) -> Optional[RatioSnapshot]:
        base = index_symbol.upper()
        proxy = ETF_PROXIES.get(base)
        polygon_symbol = POLYGON_INDEX_TICKERS.get(base)
        if not proxy or not polygon_symbol:
            return None

        cache_key = (base, proxy)
        now = pd.Timestamp.utcnow()
        existing = self._snapshot_cache.get(cache_key)
        if existing and (now - existing.updated_at).total_seconds() < self.refresh_seconds:
            return existing

        lock = self._locks.setdefault(cache_key, asyncio.Lock())
        async with lock:
            # Re-check cache after acquiring the lock.
            existing = self._snapshot_cache.get(cache_key)
            if existing and (now - existing.updated_at).total_seconds() < self.refresh_seconds:
                return existing

            index_frame, proxy_frame = await asyncio.gather(
                fetch_polygon_ohlcv(polygon_symbol, "1"),
                fetch_polygon_ohlcv(proxy, "1"),
            )
            if index_frame is None or proxy_frame is None or index_frame.empty or proxy_frame.empty:
                return existing

            aligned = _align_bars(index_frame, proxy_frame, minutes=self.lookback_minutes)
            if aligned is None or aligned.empty or len(aligned) < 15:
                return existing

            gamma_current, gamma_mean, drift, samples, spot_ratio = _compute_gamma(aligned)
            if not math.isfinite(gamma_current) or samples < 10:
                return existing

            snapshot = RatioSnapshot(
                index_symbol=base,
                proxy_symbol=proxy,
                gamma_current=float(gamma_current),
                gamma_mean=float(gamma_mean),
                spot_ratio=float(spot_ratio),
                drift=float(drift),
                samples=samples,
                updated_at=now,
                window_minutes=self.lookback_minutes,
            )
            self._snapshot_cache[cache_key] = snapshot
            return snapshot

    def translate_level(self, index_price: float, snapshot: RatioSnapshot | None) -> float:
        if snapshot is None:
            return index_price
        return snapshot.translate_level(index_price)


def _align_bars(index_frame: pd.DataFrame, proxy_frame: pd.DataFrame, *, minutes: int) -> pd.DataFrame | None:
    window = minutes * 2  # grab double window then trim
    index_tail = index_frame.sort_index().tail(window)
    proxy_tail = proxy_frame.sort_index().tail(window)
    if index_tail.empty or proxy_tail.empty:
        return None

    joined = (
        pd.concat(
            [
                index_tail["close"].rename("index_close"),
                proxy_tail["close"].rename("proxy_close"),
            ],
            axis=1,
            join="inner",
        )
        .dropna()
        .tail(minutes)
    )
    if joined.empty:
        return None
    return joined


def _rolling_gamma(index_returns: pd.Series, proxy_returns: pd.Series, window: int) -> np.ndarray:
    values: list[float] = []
    idx = index_returns.values
    proxy = proxy_returns.values
    length = len(idx)
    for start in range(length - window + 1):
        seg_idx = idx[start : start + window]
        seg_proxy = proxy[start : start + window]
        denom = float(np.dot(seg_idx, seg_idx))
        if denom == 0:
            continue
        values.append(float(np.dot(seg_idx, seg_proxy) / denom))
    return np.array(values, dtype=float) if values else np.array([], dtype=float)


def _compute_gamma(aligned: pd.DataFrame) -> Tuple[float, float, float, int, float]:
    returns = aligned.pct_change().dropna()
    if returns.empty or returns["index_close"].abs().sum() == 0:
        return 0.0, 0.0, 0.0, 0, 0.0

    idx = returns["index_close"].values
    proxy = returns["proxy_close"].values
    denom = float(np.dot(idx, idx))
    if denom == 0:
        return 0.0, 0.0, 0.0, len(returns), float(aligned["proxy_close"].iloc[-1] / aligned["index_close"].iloc[-1])

    gamma_current = float(np.dot(idx, proxy) / denom)
    window = max(10, min(len(returns), 30))
    gamma_series = _rolling_gamma(returns["index_close"], returns["proxy_close"], window)
    gamma_mean = float(np.nanmean(gamma_series)) if gamma_series.size else gamma_current
    drift = gamma_current - gamma_mean
    spot_ratio = float(aligned["proxy_close"].iloc[-1] / aligned["index_close"].iloc[-1])
    return gamma_current, gamma_mean, drift, len(returns), spot_ratio


__all__ = ["RatioEngine", "RatioSnapshot"]
