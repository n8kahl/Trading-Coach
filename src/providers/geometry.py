from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

import pandas as pd

from .series import SeriesBundle


def _ema(series: pd.Series, period: int) -> Optional[float]:
    if period <= 0 or len(series) < period:
        return None
    return float(series.ewm(span=period, adjust=False).mean().iloc[-1])


def _atr(frame: pd.DataFrame, period: int = 14) -> Optional[float]:
    if len(frame) < period + 1:
        return None
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    close = frame["close"].astype(float)
    prev_close = close.shift(1)
    tr_components = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    )
    tr = tr_components.max(axis=1)
    atr_series = tr.rolling(period).mean()
    value = atr_series.iloc[-1]
    return float(value) if pd.notna(value) else None


def _clean_levels(levels: Dict[str, Dict[str, Optional[float]]]) -> Dict[str, Dict[str, float]]:
    cleaned: Dict[str, Dict[str, float]] = {}
    for namespace, mapping in levels.items():
        filtered = {key: float(value) for key, value in mapping.items() if value is not None}
        cleaned[namespace] = filtered
    return cleaned


@dataclass(slots=True)
class GeometryDetail:
    symbol: str
    expected_move: Optional[float]
    remaining_atr: Optional[float]
    em_used: bool
    snap_trace: List[str]
    key_levels_used: Dict[str, Dict[str, float]]
    bias: Literal["long", "short"] | None
    last_close: Optional[float]
    ema_fast: Optional[float]
    ema_slow: Optional[float]


@dataclass(slots=True)
class GeometryBundle:
    symbols: List[str]
    series: SeriesBundle
    details: Dict[str, GeometryDetail] = field(default_factory=dict)

    def get(self, symbol: str) -> GeometryDetail | None:
        return self.details.get(symbol)

    @property
    def expected_move(self) -> Optional[float]:
        if not self.details:
            return None
        return next(iter(self.details.values())).expected_move

    @property
    def remaining_atr(self) -> Optional[float]:
        if not self.details:
            return None
        return next(iter(self.details.values())).remaining_atr

    @property
    def em_used(self) -> Optional[bool]:
        if not self.details:
            return None
        return next(iter(self.details.values())).em_used

    @property
    def snap_trace(self) -> List[str] | None:
        if not self.details:
            return None
        return next(iter(self.details.values())).snap_trace

    @property
    def key_levels(self) -> Dict[str, Dict[str, float]] | None:
        first = next(iter(self.details.values()), None)
        return first.key_levels_used if first else None


async def build_geometry(symbols: List[str], series: SeriesBundle) -> GeometryBundle:
    """Derive ATR/EMA driven geometry metrics for each symbol."""

    bundle = GeometryBundle(symbols=list(symbols), series=series)
    for symbol in symbols:
        frames = series.frames.get(symbol)
        if not frames:
            continue
        daily = frames.get("1d")
        if daily is None or daily.empty:
            continue
        ordered = daily.sort_index()
        expected_move = _atr(ordered)
        remaining_atr = None
        if expected_move is not None:
            try:
                today_range = float((ordered["high"].iloc[-1] - ordered["low"].iloc[-1]))
            except Exception:
                today_range = None
            if today_range is not None:
                remaining_atr = max(expected_move - today_range, 0.0)
        last_close = float(ordered["close"].iloc[-1])
        ema_fast = _ema(ordered["close"], 21)
        ema_slow = _ema(ordered["close"], 55)
        bias: Literal["long", "short"] | None = None
        if ema_fast is not None and ema_slow is not None:
            bias = "long" if ema_fast >= ema_slow else "short"
        elif ema_fast is not None:
            bias = "long" if last_close >= ema_fast else "short"

        key_levels = _clean_levels(
            {
                "session": {
                    "prev_close": float(ordered["close"].iloc[-2]) if len(ordered) > 1 else None,
                    "prev_high": float(ordered["high"].iloc[-2]) if len(ordered) > 1 else None,
                    "prev_low": float(ordered["low"].iloc[-2]) if len(ordered) > 1 else None,
                },
                "structural": {
                    "ema21": ema_fast,
                    "ema55": ema_slow,
                },
            }
        )
        snap_trace = ["geometry:atr14"]
        if expected_move is not None:
            snap_trace.append(f"atr={expected_move:.4f}")
        if ema_fast is not None:
            snap_trace.append(f"ema21={ema_fast:.4f}")
        if ema_slow is not None:
            snap_trace.append(f"ema55={ema_slow:.4f}")

        bundle.details[symbol] = GeometryDetail(
            symbol=symbol,
            expected_move=expected_move,
            remaining_atr=remaining_atr,
            em_used=expected_move is not None,
            snap_trace=snap_trace,
            key_levels_used=key_levels,
            bias=bias,
            last_close=last_close,
            ema_fast=ema_fast,
            ema_slow=ema_slow,
        )

    if not bundle.details:
        raise RuntimeError("unable to derive geometry metrics for requested symbols")

    return bundle


__all__ = ["GeometryBundle", "GeometryDetail", "build_geometry"]
