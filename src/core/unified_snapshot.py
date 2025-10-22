"""Unified market data snapshot utilities."""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from datetime import datetime, date, timezone
from typing import Any, Dict, Iterable, Optional
from zoneinfo import ZoneInfo

import pandas as pd

from ..calculations import atr, ema, vwap
from ..config import SNAPSHOT_LOOKBACK, SNAPSHOT_MAX_CONCURRENCY
from ..data_sources import fetch_polygon_ohlcv

_ET_TZ = ZoneInfo("America/New_York")
_SUPPORTED_TIMEFRAMES = {"1m", "5m"}
_INDEX_SYMBOLS: tuple[str, ...] = ("SPY", "QQQ")
_VOLATILITY_SYMBOL = "CBOE:VIX"


def _normalize_symbol(symbol: str) -> str:
    return (symbol or "").strip().upper()


def _normalize_timeframe(token: str | None) -> str:
    raw = (token or "").strip().lower()
    if raw.endswith("m"):
        raw = raw[:-1]
    if raw in {"1", "1m"}:
        return "1m"
    if raw in {"5", "5m"}:
        return "5m"
    return raw or "1m"


def _trim_frame(frame: pd.DataFrame | None, lookback: int) -> pd.DataFrame | None:
    if frame is None or frame.empty:
        return None
    if lookback and len(frame) > lookback:
        frame = frame.tail(lookback)
    return frame


def _session_stats(frame: pd.DataFrame | None) -> Dict[str, float | None]:
    payload = {
        "session_high": None,
        "session_low": None,
        "prev_high": None,
        "prev_low": None,
        "prev_close": None,
    }
    if frame is None or frame.empty:
        return payload
    if frame.index.tz is None:
        localized = frame.tz_localize("UTC")
    else:
        localized = frame.tz_convert("UTC")
    local = localized.tz_convert(_ET_TZ)
    dates = local.index.date
    if len(dates) == 0:
        return payload
    today = dates[-1]
    today_mask = dates == today
    if today_mask.any():
        payload["session_high"] = float(local.loc[today_mask, "high"].max())
        payload["session_low"] = float(local.loc[today_mask, "low"].min())
    prev_date: Optional[date] = None
    for item in reversed(dates[:-1]):
        if item != today:
            prev_date = item
            break
    if prev_date:
        prev_mask = dates == prev_date
        segment = local.loc[prev_mask]
        if not segment.empty:
            payload["prev_high"] = float(segment["high"].max())
            payload["prev_low"] = float(segment["low"].min())
            payload["prev_close"] = float(segment["close"].iloc[-1])
    return payload


def _expected_move_from_atr(atr_value: float | None, *, tf_minutes: float = 1.0, horizon_minutes: float = 390.0) -> float | None:
    if atr_value is None or not math.isfinite(atr_value):
        return None
    if atr_value <= 0 or tf_minutes <= 0 or horizon_minutes <= 0:
        return None
    steps = max(horizon_minutes / tf_minutes, 1.0)
    return float(atr_value * math.sqrt(steps))


@dataclass
class SymbolSnapshot:
    symbol: str
    last_price: float | None
    bars: Dict[str, pd.DataFrame] = field(default_factory=dict)
    atr14: float | None = None
    expected_move: float | None = None
    vwap_value: float | None = None
    ema_values: Dict[int, float] = field(default_factory=dict)
    session_high: float | None = None
    session_low: float | None = None
    prev_high: float | None = None
    prev_low: float | None = None
    prev_close: float | None = None

    def get_bars(self, timeframe: str) -> pd.DataFrame | None:
        key = _normalize_timeframe(timeframe)
        return self.bars.get(key)

    def change_pct(self) -> float | None:
        if self.prev_close is None or self.prev_close == 0 or self.last_price is None:
            return None
        try:
            return float((self.last_price - self.prev_close) / self.prev_close * 100.0)
        except ZeroDivisionError:
            return None


@dataclass
class UnifiedSnapshot:
    symbols: Dict[str, SymbolSnapshot]
    indices: Dict[str, Dict[str, float | None]]
    volatility: Dict[str, float | None]
    generated_at: datetime

    def get_symbol(self, symbol: str) -> SymbolSnapshot | None:
        return self.symbols.get(_normalize_symbol(symbol))

    def summary(self) -> Dict[str, Any]:
        return {
            "generated_at": self.generated_at.isoformat(),
            "symbol_count": len(self.symbols),
        }


async def _fetch_symbol_frames(symbol: str, *, semaphore: asyncio.Semaphore, include_5m: bool, lookback: int) -> Dict[str, pd.DataFrame | None]:
    frames: Dict[str, pd.DataFrame | None] = {"1m": None, "5m": None}

    async def _fetch(timeframe: str) -> pd.DataFrame | None:
        async with semaphore:
            return await fetch_polygon_ohlcv(symbol, timeframe)

    one_minute = await _fetch("1")
    frames["1m"] = _trim_frame(one_minute, lookback)

    if include_5m:
        five_minute = await _fetch("5")
        frames["5m"] = _trim_frame(five_minute, max(lookback // 5, 0))
    return frames


def _compute_indicators(frame: pd.DataFrame | None) -> tuple[float | None, float | None, Dict[int, float], float | None]:
    if frame is None or frame.empty:
        return None, None, {}, None
    closes = frame["close"]
    highs = frame["high"]
    lows = frame["low"]
    volumes = frame.get("volume")
    atr_series = atr(highs, lows, closes, 14)
    atr_value = float(atr_series.dropna().iloc[-1]) if not atr_series.dropna().empty else None
    em_map: Dict[int, float] = {}
    for period in (9, 20, 50):
        ema_series = ema(closes, period)
        if not ema_series.dropna().empty:
            em_map[period] = float(ema_series.dropna().iloc[-1])
    vwap_series = None
    if volumes is not None:
        typical_price = (highs + lows + closes) / 3.0
        vwap_series = vwap(typical_price, volumes)
    vwap_value = float(vwap_series.dropna().iloc[-1]) if vwap_series is not None and not vwap_series.dropna().empty else None
    expected_move_val = _expected_move_from_atr(atr_value)
    return atr_value, expected_move_val, em_map, vwap_value


async def get_unified_snapshot(
    symbols: Iterable[str],
    *,
    interval: str = "1m",
    lookback: int = SNAPSHOT_LOOKBACK,
    live: bool = True,
) -> UnifiedSnapshot:
    requested = {_normalize_symbol(symbol) for symbol in symbols if symbol}
    requested.discard("")
    fetch_symbols = set(requested)
    fetch_symbols.update(_INDEX_SYMBOLS)
    fetch_symbols.add(_VOLATILITY_SYMBOL)

    semaphore = asyncio.Semaphore(max(1, SNAPSHOT_MAX_CONCURRENCY))
    include_5m = True
    fetch_tasks = {
        symbol: asyncio.create_task(_fetch_symbol_frames(symbol, semaphore=semaphore, include_5m=include_5m, lookback=lookback))
        for symbol in fetch_symbols
    }
    await asyncio.gather(*fetch_tasks.values())

    symbol_snapshots: Dict[str, SymbolSnapshot] = {}
    for symbol, task in fetch_tasks.items():
        frames = task.result()
        frame_1m = frames.get("1m")
        frame_5m = frames.get("5m")
        last_price = None
        if frame_1m is not None and not frame_1m.empty:
            last_price = float(frame_1m["close"].iloc[-1])
        atr_value, expected_move, ema_map, vwap_value = _compute_indicators(frame_1m)
        session_meta = _session_stats(frame_1m)

        bars_map: Dict[str, pd.DataFrame] = {}
        for key, frame in frames.items():
            if frame is None or frame.empty:
                continue
            frame = frame.copy()
            frame.attrs["source"] = "unified_snapshot"
            bars_map[key] = frame

        snapshot = SymbolSnapshot(
            symbol=symbol,
            last_price=last_price,
            bars=bars_map,
            atr14=atr_value,
            expected_move=expected_move,
            vwap_value=vwap_value,
            ema_values=ema_map,
            session_high=session_meta.get("session_high"),
            session_low=session_meta.get("session_low"),
            prev_high=session_meta.get("prev_high"),
            prev_low=session_meta.get("prev_low"),
            prev_close=session_meta.get("prev_close"),
        )
        if frame_5m is not None and not frame_5m.empty and snapshot.atr14 is None:
            atr_value_5m, expected_move_5m, _, _ = _compute_indicators(frame_5m)
            if snapshot.atr14 is None:
                snapshot.atr14 = atr_value_5m
            if snapshot.expected_move is None:
                snapshot.expected_move = expected_move_5m
        symbol_snapshots[symbol] = snapshot

    indices_context: Dict[str, Dict[str, float | None]] = {}
    for index_symbol in _INDEX_SYMBOLS:
        snap = symbol_snapshots.get(index_symbol)
        if snap:
            indices_context[index_symbol] = {
                "close": snap.last_price,
                "change_pct": snap.change_pct(),
            }

    volatility_value = None
    vol_snapshot = symbol_snapshots.get(_VOLATILITY_SYMBOL)
    if vol_snapshot:
        volatility_value = vol_snapshot.last_price

    unified = UnifiedSnapshot(
        symbols={symbol: snapshot for symbol, snapshot in symbol_snapshots.items() if symbol in requested},
        indices=indices_context,
        volatility={"symbol": _VOLATILITY_SYMBOL, "value": volatility_value},
        generated_at=datetime.now(timezone.utc),
    )
    return unified


__all__ = ["SymbolSnapshot", "UnifiedSnapshot", "get_unified_snapshot"]
