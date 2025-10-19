import asyncio
from datetime import datetime

import pytest

from src.services.universe import UniverseProvider, UniverseSnapshot


class DummyPolygonClient:
    def __init__(self) -> None:
        self.index_calls = 0
        self.top_calls = 0

    async def fetch_index_constituents(self, index_ticker: str):
        self.index_calls += 1
        if index_ticker == "I:SPX":
            return ["AAPL", "MSFT", "NVDA"]
        if index_ticker == "I:NDX":
            return ["QQQ", "ADBE", "NFLX"]
        return []

    async def fetch_top_market_cap(self, limit: int):
        self.top_calls += 1
        return ["TSLA", "NVDA", "META"][:limit]


class NoOpPersistence:
    async def record_universe_snapshot(self, snapshot: UniverseSnapshot):
        self.last_snapshot = snapshot


@pytest.mark.asyncio
async def test_universe_provider_fetches_index_and_caches(monkeypatch):
    client = DummyPolygonClient()
    persistence = NoOpPersistence()
    provider = UniverseProvider(client, persistence=persistence)

    snapshot = await provider.get_universe("sp500", style="intraday", limit=2)
    assert snapshot.symbols == ["AAPL", "MSFT"]
    assert persistence.last_snapshot.name == "sp500"
    assert client.index_calls == 1

    # Subsequent call should use cache
    snapshot_cached = await provider.get_universe("sp500", style="intraday", limit=2)
    assert snapshot_cached.symbols == snapshot.symbols
    assert client.index_calls == 1

    # Unknown universe should fall back to top market cap list
    snapshot_top = await provider.get_universe("unknown", style="intraday", limit=2)
    assert snapshot_top.symbols == ["TSLA", "NVDA"]
    assert client.top_calls >= 1
