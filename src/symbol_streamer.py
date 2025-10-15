from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, Optional
from zoneinfo import ZoneInfo

import httpx
from .config import get_settings
from .data_sources import fetch_polygon_ohlcv

logger = logging.getLogger(__name__)


@dataclass
class QuoteResult:
    price: Optional[float]
    timestamp: Optional[str]
    source: str
    error: Optional[str] = None


async def _fetch_polygon_last_trade(symbol: str, api_key: str) -> QuoteResult:
    url = f"https://api.polygon.io/v2/last/trade/{symbol.upper()}"
    params = {"apiKey": api_key}
    timeout = httpx.Timeout(4.0, connect=2.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            payload = resp.json()
        except httpx.HTTPError as exc:
            logger.debug("polygon last trade request failed", exc_info=exc, extra={"symbol": symbol})
            return QuoteResult(price=None, timestamp=None, source="polygon", error="polygon_error")
    trade = payload.get("results") or {}
    price = trade.get("p")
    timestamp_ns = trade.get("t")
    timestamp = None
    if isinstance(timestamp_ns, (int, float)) and timestamp_ns > 0:
        timestamp = datetime.fromtimestamp(timestamp_ns / 1_000_000_000, tz=timezone.utc).isoformat()
    return QuoteResult(
        price=float(price) if isinstance(price, (int, float)) else None,
        timestamp=timestamp,
        source="polygon",
        error=None if isinstance(price, (int, float)) else "no_price",
    )


async def _fetch_finnhub_quote(symbol: str, api_key: str) -> QuoteResult:
    url = "https://finnhub.io/api/v1/quote"
    params = {"symbol": symbol.upper(), "token": api_key}
    timeout = httpx.Timeout(4.0, connect=2.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            payload = resp.json()
        except httpx.HTTPError as exc:
            logger.debug("finnhub quote request failed", exc_info=exc, extra={"symbol": symbol})
            return QuoteResult(price=None, timestamp=None, source="finnhub", error="finnhub_error")
    price = payload.get("c")
    timestamp_sec = payload.get("t")
    timestamp = None
    if isinstance(timestamp_sec, (int, float)) and timestamp_sec > 0:
        timestamp = datetime.fromtimestamp(timestamp_sec, tz=timezone.utc).isoformat()
    return QuoteResult(
        price=float(price) if isinstance(price, (int, float)) else None,
        timestamp=timestamp,
        source="finnhub",
        error=None if isinstance(price, (int, float)) else "no_price",
    )


async def _fetch_yahoo_last_trade(symbol: str) -> QuoteResult:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol.upper()}"
    params = {"interval": "1m", "range": "1d", "includePrePost": "false"}
    timeout = httpx.Timeout(5.0, connect=2.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            payload = resp.json()
        except httpx.HTTPError as exc:
            logger.debug("yahoo last trade request failed", exc_info=exc, extra={"symbol": symbol})
            return QuoteResult(price=None, timestamp=None, source="yahoo", error="yahoo_error")
    try:
        result = payload["chart"]["result"][0]
        timestamps = result.get("timestamp") or []
        quote = result["indicators"]["quote"][0]
        closes = quote.get("close") or []
    except (KeyError, IndexError, TypeError) as exc:
        logger.debug("yahoo payload parse failed", exc_info=exc, extra={"symbol": symbol})
        return QuoteResult(price=None, timestamp=None, source="yahoo", error="yahoo_error")
    if not timestamps or not closes:
        return QuoteResult(price=None, timestamp=None, source="yahoo", error="yahoo_empty")
    price = closes[-1]
    ts = timestamps[-1]
    if price is None:
        return QuoteResult(price=None, timestamp=None, source="yahoo", error="yahoo_no_price")
    timestamp = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    return QuoteResult(price=float(price), timestamp=timestamp, source="yahoo", error=None)


async def fetch_live_quote(symbol: str) -> QuoteResult:
    """Attempt to fetch the latest trade from available providers."""

    settings = get_settings()
    errors = []
    if settings.polygon_api_key:
        quote = await _fetch_polygon_last_trade(symbol, settings.polygon_api_key)
        if quote.price is not None:
            return quote
        errors.append(quote.error or "polygon_error")
    if settings.finnhub_api_key:
        quote = await _fetch_finnhub_quote(symbol, settings.finnhub_api_key)
        if quote.price is not None:
            return quote
        errors.append(quote.error or "finnhub_error")
    yahoo_quote = await _fetch_yahoo_last_trade(symbol)
    if yahoo_quote.price is not None:
        return yahoo_quote
    errors.append(yahoo_quote.error or "yahoo_error")
    try:
        polygon_frame = await fetch_polygon_ohlcv(symbol.upper(), "1")
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("polygon bars fallback failed", exc_info=exc, extra={"symbol": symbol})
        polygon_frame = None
    if polygon_frame is not None and not polygon_frame.empty:
        last = polygon_frame.iloc[-1]
        price = float(last.get("close"))
        ts = last.name
        if isinstance(ts, datetime):
            ts = ts.astimezone(timezone.utc).isoformat()
        else:
            ts = datetime.utcnow().isoformat()
        return QuoteResult(price=price, timestamp=ts, source="polygon_bars", error=None)
    if not settings.polygon_api_key and not settings.finnhub_api_key:
        errors.append("missing_credentials")
    return QuoteResult(price=None, timestamp=None, source="none", error=";".join(errors) or "no_data")


def determine_market_phase(now: Optional[datetime] = None) -> str:
    """Return the current U.S. equities market phase."""

    dt = (now or datetime.now(timezone.utc)).astimezone(ZoneInfo("America/New_York"))
    if dt.weekday() >= 5:
        return "closed"
    minutes = dt.hour * 60 + dt.minute
    if 4 * 60 <= minutes < 9 * 60 + 30:
        return "premarket"
    if 9 * 60 + 30 <= minutes < 16 * 60:
        return "regular"
    if 16 * 60 <= minutes < 20 * 60:
        return "afterhours"
    return "closed"


class SymbolStreamCoordinator:
    """Background poller that keeps symbols supplied with live data events."""

    def __init__(
        self,
        send_event: Callable[[str, Dict[str, Any]], Awaitable[None]],
        *,
        regular_interval: float = 5.0,
        after_hours_interval: float = 20.0,
    ) -> None:
        self._send_event = send_event
        self._regular_interval = regular_interval
        self._after_hours_interval = after_hours_interval
        self._tasks: Dict[str, asyncio.Task[None]] = {}
        self._lock = asyncio.Lock()

    async def ensure_symbol(self, symbol: str) -> None:
        uppercase = (symbol or "").upper()
        if not uppercase:
            return
        async with self._lock:
            task = self._tasks.get(uppercase)
            if task and not task.done():
                return
            task = asyncio.create_task(self._runner(uppercase))
            self._tasks[uppercase] = task
            task.add_done_callback(lambda _: self._tasks.pop(uppercase, None))

    async def _runner(self, symbol: str) -> None:
        last_phase: Optional[str] = None
        last_note: Optional[str] = None
        failure_streak = 0
        try:
            while True:
                phase = determine_market_phase()
                note: Optional[str] = None
                try:
                    quote = await fetch_live_quote(symbol)
                except Exception as exc:
                    logger.exception("live quote fetch failed", extra={"symbol": symbol})
                    quote = QuoteResult(price=None, timestamp=None, source="unknown", error="exception")
                if quote.price is not None:
                    failure_streak = 0
                    event = {
                        "t": "tick",
                        "p": quote.price,
                        "ts": quote.timestamp or datetime.now(timezone.utc).isoformat(),
                        "source": quote.source,
                    }
                    await self._send_event(symbol, event)
                    if last_note and "Live data unavailable" in last_note:
                        recovery = "Live data connection restored."
                        await self._send_event(symbol, {"t": "market_status", "phase": phase, "note": recovery})
                        last_note = recovery
                else:
                    failure_streak += 1
                    if quote.error == "missing_credentials":
                        note = "Live data unavailable: add Polygon or Finnhub API credentials."
                    elif quote.error:
                        note = f"Live data unavailable ({quote.error})."

                if phase != "regular":
                    phase_note = f"Market {phase.replace('_', ' ')} — live updates may be limited."
                    note = f"{phase_note} {note}" if note else phase_note

                if note and note != last_note:
                    await self._send_event(symbol, {"t": "market_status", "phase": phase, "note": note})
                    last_note = note
                elif phase != last_phase:
                    default_note = "Market open." if phase == "regular" else f"Market {phase.replace('_', ' ')} — live updates may be limited."
                    await self._send_event(symbol, {"t": "market_status", "phase": phase, "note": default_note})
                    last_note = default_note

                last_phase = phase
                interval = self._regular_interval if phase == "regular" else self._after_hours_interval
                if quote.error == "missing_credentials":
                    interval = max(interval, 60.0)
                elif quote.price is None:
                    interval = min(max(interval * (1 + failure_streak * 0.5), interval), 60.0)
                await asyncio.sleep(interval)
        except asyncio.CancelledError:
            logger.info("symbol stream cancelled", extra={"symbol": symbol})
            raise
        except Exception:
            logger.exception("symbol stream crashed", extra={"symbol": symbol})
