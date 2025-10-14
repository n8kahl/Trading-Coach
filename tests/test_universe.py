import types

import pytest

from src.app.providers import universe


@pytest.mark.asyncio
async def test_load_universe_sector_filter(monkeypatch):
    reference = [
        {"ticker": "HCA", "market_cap": 50_000_000_000, "sector": "Health Care", "primary_exchange": "XNYS"},
        {"ticker": "TECH", "market_cap": 100_000_000_000, "sector": "Information Technology", "primary_exchange": "XNAS"},
        {"ticker": "HLTH", "market_cap": 30_000_000_000, "sector": "Health Care", "primary_exchange": "XNYS"},
        {"ticker": "SMALL", "market_cap": 1_000_000_000, "sector": "Health Care", "primary_exchange": "XNYS"},
    ]

    biases = {"health care": 0.1, "information technology": -0.05}

    async def fake_fetch(api_key):  # noqa: ARG001
        return reference

    async def fake_bias(api_key):  # noqa: ARG001
        return biases

    monkeypatch.setattr(universe, "_fetch_top_list", fake_fetch)
    monkeypatch.setattr(universe, "_sector_biases", fake_bias)

    settings = types.SimpleNamespace(polygon_api_key="test")
    monkeypatch.setattr(universe, "get_settings", lambda: settings)

    tickers = await universe.load_universe(style="intraday", sector="healthcare", limit=3)

    assert tickers[:2] == ["HCA", "HLTH"]
    assert "TECH" not in tickers


@pytest.mark.asyncio
async def test_load_universe_without_polygon_key(monkeypatch):
    settings = types.SimpleNamespace(polygon_api_key=None)
    monkeypatch.setattr(universe, "get_settings", lambda: settings)

    tickers = await universe.load_universe(style=None, sector=None, limit=15)

    assert len(tickers) == 15
    assert tickers[0] == "SPY"
