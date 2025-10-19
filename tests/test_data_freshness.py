from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pandas as pd
import pytest

from src import agent_server
from src import symbol_streamer
from src.symbol_streamer import QUOTES_MAX_AGE_S, QuoteResult


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio("asyncio")
async def test_quote_freshness_gate_allows_recent_quotes(monkeypatch):
    now = datetime.now(timezone.utc)

    async def fake_polygon(symbol: str, api_key: str) -> QuoteResult:
        return QuoteResult(price=101.25, timestamp=now.isoformat(), source="polygon")

    async def fake_finnhub(symbol: str, api_key: str) -> QuoteResult:
        return QuoteResult(price=None, timestamp=None, source="finnhub", error="no_price")

    def fake_settings():
        return SimpleNamespace(polygon_api_key="key", finnhub_api_key="other")

    async def fake_polygon_bars(symbol: str, interval: str):
        raise AssertionError("fallback should not be used when quote fresh")

    monkeypatch.setattr(symbol_streamer, "_fetch_polygon_last_trade", fake_polygon)
    monkeypatch.setattr(symbol_streamer, "_fetch_finnhub_quote", fake_finnhub)
    monkeypatch.setattr(symbol_streamer, "fetch_polygon_ohlcv", fake_polygon_bars)
    monkeypatch.setattr(symbol_streamer, "get_settings", fake_settings)

    quote = await symbol_streamer.fetch_live_quote("AAPL")
    assert quote.price == pytest.approx(101.25)
    assert quote.age_seconds is not None
    assert quote.age_seconds <= QUOTES_MAX_AGE_S


@pytest.mark.anyio("asyncio")
async def test_quote_freshness_gate_rejects_stale_quotes(monkeypatch):
    ts = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()

    async def stale_polygon(symbol: str, api_key: str) -> QuoteResult:
        return QuoteResult(price=404.0, timestamp=ts, source="polygon")

    async def fake_polygon_bars(symbol: str, interval: str):
        return None

    def fake_settings():
        return SimpleNamespace(polygon_api_key="token", finnhub_api_key=None)

    monkeypatch.setattr(symbol_streamer, "_fetch_polygon_last_trade", stale_polygon)
    monkeypatch.setattr(symbol_streamer, "_fetch_finnhub_quote", stale_polygon)
    monkeypatch.setattr(symbol_streamer, "fetch_polygon_ohlcv", fake_polygon_bars)
    monkeypatch.setattr(symbol_streamer, "get_settings", fake_settings)

    quote = await symbol_streamer.fetch_live_quote("MSFT")
    assert quote.price is None
    assert quote.error is not None and "stale_quote" in quote.error


@pytest.fixture
def _stubbed_context(monkeypatch):
    def fake_settings():
        return SimpleNamespace(polygon_api_key="token", finnhub_api_key=None)

    monkeypatch.setattr(agent_server, "get_settings", fake_settings)
    monkeypatch.setattr(agent_server, "normalize_interval", lambda interval: interval)

    base_time = datetime.now(timezone.utc)
    frame = pd.DataFrame(
        {
            "time": [
                (base_time - timedelta(minutes=idx)).isoformat()
                for idx in range(3)
            ],
            "open": [100.0, 100.5, 101.0],
            "high": [101.0, 101.5, 102.0],
            "low": [99.5, 100.0, 100.5],
            "close": [100.5, 101.0, 101.5],
            "volume": [1_000, 1_050, 1_100],
        }
    )

    def fake_get_candles(symbol: str, interval: str, lookback: int):
        return frame

    monkeypatch.setattr(agent_server, "get_candles", fake_get_candles)
    monkeypatch.setattr(agent_server, "compute_context_overlays", lambda *a, **k: {})
    monkeypatch.setattr(agent_server, "ema", lambda series, length: pd.Series([series.iloc[-1]]))
    monkeypatch.setattr(agent_server, "vwap", lambda close, volume: pd.Series([float(close.iloc[-1])]))
    monkeypatch.setattr(agent_server, "atr", lambda *a, **k: pd.Series([1.0]))
    monkeypatch.setattr(agent_server, "adx", lambda *a, **k: pd.Series([20.0]))

    async def fake_chain(symbol: str):
        return pd.DataFrame({"dummy": [1]})

    monkeypatch.setattr(agent_server, "fetch_polygon_option_chain", fake_chain)
    monkeypatch.setattr(agent_server, "summarize_polygon_chain", lambda chain, rules=None, top_n=3: {
        "underlying": {"symbol": "AAPL", "price": 100.0},
        "best": {}
    })

    return frame


@pytest.mark.anyio("asyncio")
async def test_context_options_hold_on_underlying_mismatch(monkeypatch, _stubbed_context):
    async def fake_quote(symbol: str):
        return QuoteResult(price=103.0, timestamp=datetime.now(timezone.utc).isoformat(), source="polygon", age_seconds=1.0)

    monkeypatch.setattr(agent_server, "fetch_live_quote", fake_quote)

    response = await agent_server.gpt_context("AAPL")
    options = response.get("options")
    assert options is not None
    assert options.get("hold_refresh") is True
    pct = options["consistency"].get("underlying_vs_quote_pct")
    assert pct is not None and pct > 0.002


@pytest.mark.anyio("asyncio")
async def test_context_options_pass_when_within_threshold(monkeypatch, _stubbed_context):
    async def fresh_quote(symbol: str):
        return QuoteResult(price=100.1, timestamp=datetime.now(timezone.utc).isoformat(), source="polygon", age_seconds=2.0)

    monkeypatch.setattr(agent_server, "fetch_live_quote", fresh_quote)

    response = await agent_server.gpt_context("AAPL")
    options = response.get("options")
    assert options is not None
    assert options.get("hold_refresh") is not True
    pct = options["consistency"].get("underlying_vs_quote_pct")
    assert pct is not None and pct <= 0.002 + 1e-9
    assert options["consistency"]["quote_age_s"] == pytest.approx(2.0)
