"""
Polygon real-time bar streamer that forwards 1-minute aggregates to the stream bus.

This module connects to Polygon's indices websocket feed and emits ``bar`` events for
each tracked symbol.  The rest of the application can treat these events like live
ticks — updating the price header immediately while the heavier plan recalculations
continue to run on their existing schedule.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, Iterable, List, Optional

import websockets

logger = logging.getLogger(__name__)

# Polygon's indices websocket endpoint plus aggregate channel.
POLYGON_INDICES_WS_URL = "wss://socket.polygon.io/indices"
AGGREGATE_CHANNEL_PREFIX = "XA"


def _stream_symbol_for(symbol: str) -> str:
    base = (symbol or "").strip().upper()
    if base.startswith("I:"):
        return base
    return f"I:{base}"


def _base_symbol(stream_symbol: str) -> Optional[str]:
    if not stream_symbol:
        return None
    if ":" in stream_symbol:
        parts = stream_symbol.split(":", 1)
        if len(parts) == 2:
            return parts[1].upper()
    return stream_symbol.upper()


@dataclass(slots=True)
class PolygonBarEvent:
    symbol: str
    timestamp: int  # epoch seconds
    open: float
    high: float
    low: float
    close: float
    volume: float

    def to_stream_payload(self) -> Dict[str, float | int | str]:
        return {
            "t": "bar",
            "symbol": self.symbol,
            "ts": self.timestamp,
            "time": self.timestamp,
            "p": self.close,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }


class PolygonRealtimeBarStreamer:
    """Connect to Polygon's indices websocket and emit minute bars."""

    def __init__(
        self,
        api_key: str,
        symbols: Iterable[str],
        *,
        on_event: Callable[[str, Dict[str, object]], Awaitable[None]],
        reconnect_delay: float = 5.0,
    ) -> None:
        self._api_key = (api_key or "").strip()
        self._symbols = sorted({symbol.strip().upper() for symbol in symbols if symbol})
        self._on_event = on_event
        self._reconnect_delay = max(reconnect_delay, 1.0)
        self._task: Optional[asyncio.Task[None]] = None
        self._stop_requested = asyncio.Event()

    def start(self) -> None:
        if not self._api_key or not self._symbols:
            logger.info("Polygon realtime bars disabled (api_key=%s, symbols=%s)", bool(self._api_key), self._symbols)
            return
        if self._task is not None:
            return
        logger.info("Starting Polygon realtime bar streamer for %s", ",".join(self._symbols))
        self._stop_requested.clear()
        self._task = asyncio.create_task(self._run(), name="polygon-realtime-bars")

    async def stop(self) -> None:
        if self._task is None:
            return
        logger.info("Stopping Polygon realtime bar streamer")
        self._stop_requested.set()
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        finally:
            self._task = None
            self._stop_requested.clear()

    async def _run(self) -> None:
        backoff = self._reconnect_delay
        while not self._stop_requested.is_set():
            try:
                async with websockets.connect(
                    POLYGON_INDICES_WS_URL,
                    ping_interval=20,
                    ping_timeout=20,
                    close_timeout=5,
                ) as ws:
                    await self._authenticate(ws)
                    await self._subscribe(ws)
                    backoff = self._reconnect_delay  # reset backoff on success
                    async for raw in ws:
                        if self._stop_requested.is_set():
                            break
                        await self._handle_message(raw)
            except asyncio.CancelledError:
                break
            except Exception as exc:  # pragma: no cover - network resilience
                logger.warning("Polygon realtime bar stream error: %s", exc, exc_info=True)
            if self._stop_requested.is_set():
                break
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60.0)

    async def _authenticate(self, ws: websockets.WebSocketClientProtocol) -> None:
        payload = json.dumps({"action": "auth", "params": self._api_key})
        await ws.send(payload)

    async def _subscribe(self, ws: websockets.WebSocketClientProtocol) -> None:
        stream_tokens = ",".join(f"{AGGREGATE_CHANNEL_PREFIX}.{_stream_symbol_for(sym)}" for sym in self._symbols)
        payload = json.dumps({"action": "subscribe", "params": stream_tokens})
        await ws.send(payload)

    async def _handle_message(self, raw: str) -> None:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.debug("Failed to decode Polygon realtime payload: %s", raw)
            return

        payloads: List[Dict[str, object]]
        if isinstance(data, list):
            payloads = data
        elif isinstance(data, dict):
            payloads = [data]
        else:
            return

        for chunk in payloads:
            if not isinstance(chunk, dict):
                continue
            event_type = chunk.get("ev")
            if event_type == "status":
                logger.debug("Polygon status event: %s", chunk)
                continue
            if event_type != AGGREGATE_CHANNEL_PREFIX:
                continue
            event = self._parse_bar_event(chunk)
            if event is None:
                continue
            try:
                await self._on_event(event.symbol, event.to_stream_payload())
            except Exception:  # pragma: no cover - defensive fan-out
                logger.exception("Failed to dispatch realtime bar event")

    def _parse_bar_event(self, payload: Dict[str, object]) -> Optional[PolygonBarEvent]:
        stream_symbol = payload.get("sym")
        base_symbol = _base_symbol(stream_symbol if isinstance(stream_symbol, str) else "")
        if not base_symbol:
            return None
        ts_ms = payload.get("t")
        if not isinstance(ts_ms, (int, float)):
            return None
        open_price = self._coerce_number(payload.get("o"))
        high_price = self._coerce_number(payload.get("h"))
        low_price = self._coerce_number(payload.get("l"))
        close_price = self._coerce_number(payload.get("c"))
        if None in (open_price, high_price, low_price, close_price):
            return None
        volume_val = self._coerce_number(payload.get("v"), default=0.0)
        return PolygonBarEvent(
            symbol=base_symbol,
            timestamp=int(ts_ms / 1000),
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=volume_val if volume_val is not None else 0.0,
        )

    @staticmethod
    def _coerce_number(value: object, default: Optional[float] = None) -> Optional[float]:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return default
        return default


__all__ = ["PolygonRealtimeBarStreamer"]
