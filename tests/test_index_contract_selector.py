import pandas as pd
import pytest

from src.app.engine.index_health import FeedStatus
from src.app.engine.index_mode import GammaSnapshot, IndexDataHealth
from src.app.engine.index_selector import IndexOptionSelector


class StubPlanner:
    contract_preference = ("INDEX_POLYGON", "INDEX_TRADIER", "ETF_PROXY")

    def __init__(self, polygon_df, tradier_df, proxy_df, snapshot, health):
        self._polygon = polygon_df
        self._tradier = tradier_df
        self._proxy = proxy_df
        self._snapshot = snapshot
        self._health = health

    async def polygon_index_chain(self, symbol):  # noqa: ARG002
        return self._polygon

    async def tradier_index_chain(self, symbol):  # noqa: ARG002
        return self._tradier

    async def gamma_snapshot(self, symbol):  # noqa: ARG002
        return self._snapshot

    async def feed_health(self, symbol, expiration=None):  # noqa: ARG002
        return self._health

    def proxy_symbol(self, symbol):  # noqa: ARG002
        return "SPY"


def _basic_health(status_polygon: str, status_tradier: str) -> IndexDataHealth:
    return IndexDataHealth(
        polygon={"SPX": FeedStatus(source="polygon", symbol="SPX", status=status_polygon)},
        tradier={"SPX": FeedStatus(source="tradier", symbol="SPX", status=status_tradier)},
        notes=[],
    )


@pytest.mark.asyncio
async def test_index_selector_prefers_polygon_chain(monkeypatch):
    polygon_df = pd.DataFrame(
        [
            {
                "symbol": "SPX240621C05000000",
                "option_type": "call",
                "bid": 10.0,
                "ask": 11.0,
                "spread_pct": 0.05,
                "dte": 5,
                "delta": 0.52,
                "open_interest": 250,
                "volume": 120,
            }
        ]
    )

    planner = StubPlanner(
        polygon_df=polygon_df,
        tradier_df=pd.DataFrame(),
        proxy_df=pd.DataFrame(),
        snapshot=None,
        health=_basic_health("healthy", "healthy"),
    )

    selector = IndexOptionSelector(planner=planner)

    contract, decision = await selector.contract_decision("SPX", prefer_delta=0.5, style="intraday")

    assert contract is not None
    assert decision.source == "INDEX_POLYGON"
    assert contract["symbol"] == "SPX240621C05000000"


@pytest.mark.asyncio
async def test_index_selector_falls_back_to_etf(monkeypatch):
    now = pd.Timestamp.utcnow()
    snapshot = GammaSnapshot(
        index_symbol="SPX",
        proxy_symbol="SPY",
        gamma_current=0.1,
        gamma_mean=0.1,
        spot_ratio=0.1,
        drift=0.0,
        samples=40,
        updated_at=now,
        window_minutes=60,
    )

    proxy_df = pd.DataFrame(
        [
            {
                "symbol": "SPY240621C00450000",
                "option_type": "call",
                "bid": 1.0,
                "ask": 1.1,
                "spread_pct": 0.02,
                "dte": 5,
                "delta": 0.48,
                "open_interest": 12000,
                "volume": 5000,
            }
        ]
    )

    planner = StubPlanner(
        polygon_df=pd.DataFrame(),
        tradier_df=pd.DataFrame(),
        proxy_df=proxy_df,
        snapshot=snapshot,
        health=_basic_health("failed", "degraded"),
    )

    async def fake_fetch(symbol: str, expiration: str | None = None):  # noqa: ARG001
        return proxy_df.copy()

    monkeypatch.setattr("src.app.engine.index_selector.fetch_option_chain_cached", fake_fetch)

    selector = IndexOptionSelector(planner=planner)
    contract, decision = await selector.contract_decision("SPX", prefer_delta=0.5, style="intraday")

    assert contract is not None
    assert decision.source == "ETF_PROXY"
    assert contract["proxy_for"] == "SPX"
    assert decision.execution_proxy["symbol"] == "SPY"
    assert "SPY" in decision.fallback_note
