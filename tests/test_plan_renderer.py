import pandas as pd
import pytest
from starlette.requests import Request

from src.agent_server import ScanUniverse, AuthedUser, gpt_scan
from src.scanner import Plan, Signal
from src.app.engine.index_selector import ContractsDecision
from src.app.engine.index_mode import GammaSnapshot, IndexDataHealth
from src.app.engine.index_health import FeedStatus


class StubIndexMode:
    contract_preference = ("INDEX_POLYGON", "INDEX_TRADIER", "ETF_PROXY")

    def __init__(self, decision):
        self.decision = decision
        self.planner = self

    def applies(self, symbol: str) -> bool:
        return symbol.upper() == "SPX"

    def liquidity_proxy(self, symbol: str) -> str:  # noqa: ARG002
        return "SPY"

    async def select_contract(self, symbol: str, *, prefer_delta: float, style: str | None = None):  # noqa: ARG002
        contract = {
            "symbol": "SPY240621C00450000",
            "proxy_for": "SPX",
            "bid": 1.0,
            "ask": 1.1,
            "spread_pct": 0.02,
        }
        return contract, self.decision

    async def ratio_snapshot(self, symbol: str):  # noqa: ARG002
        return self.decision.proxy_snapshot

    def translate_level(self, value: float, snapshot: GammaSnapshot) -> float:
        return value * snapshot.gamma_current

    def translate_targets(self, targets, snapshot: GammaSnapshot):
        return [t * snapshot.gamma_current for t in targets]


@pytest.mark.asyncio
async def test_plan_renderer_adds_index_fallback_metadata(monkeypatch):
    now = pd.Timestamp.utcnow()
    snapshot = GammaSnapshot(
        index_symbol="SPX",
        proxy_symbol="SPY",
        gamma_current=0.1,
        gamma_mean=0.1,
        spot_ratio=0.1,
        drift=0.0,
        samples=30,
        updated_at=now,
        window_minutes=60,
    )
    decision = ContractsDecision(
        source="ETF_PROXY",
        chain=None,
        proxy_snapshot=snapshot,
        fallback_note="Index chain degraded",
        execution_proxy={"symbol": "SPY", "gamma": 0.1, "ratio": 0.1, "note": "fallback"},
        health=IndexDataHealth(
            polygon={"SPX": FeedStatus(source="polygon", symbol="SPX", status="failed")},
            tradier={"SPX": FeedStatus(source="tradier", symbol="SPX", status="degraded")},
            notes=["degraded"],
        ),
    )

    stub_mode = StubIndexMode(decision)

    async def fake_collect_market_data(tickers, timeframe, as_of=None):  # noqa: ARG001
        frame = pd.DataFrame(
            {
                "open": [4000.0, 4005.0],
                "high": [4005.0, 4010.0],
                "low": [3995.0, 4000.0],
                "close": [4002.0, 4008.0],
                "volume": [1_000_000, 1_200_000],
            },
            index=pd.date_range("2024-01-01 14:30", periods=2, freq="5T", tz="UTC"),
        )
        return {"SPX": frame}, {"SPX": "polygon"}, None

    plan = Plan(
        direction="long",
        entry=4005.0,
        stop=3995.0,
        targets=[4015.0],
        confidence=0.6,
        risk_reward=1.5,
    )
    signal = Signal(
        symbol="SPX",
        strategy_id="test_strategy",
        description="Test",
        score=0.8,
        plan=plan,
        features={"direction_bias": "long"},
    )

    async def fake_scan_market(tickers, market_data, **kwargs):  # noqa: ARG001
        return [signal]

    monkeypatch.setattr("src.agent_server._get_index_mode", lambda: stub_mode)
    monkeypatch.setattr("src.agent_server._collect_market_data", fake_collect_market_data)
    monkeypatch.setattr("src.agent_server.scan_market", fake_scan_market)
    monkeypatch.setattr("src.agent_server.compute_context_overlays", lambda *args, **kwargs: {})
    monkeypatch.setattr("src.agent_server._fetch_option_chain_with_aliases", lambda symbol, as_of: pd.DataFrame())
    monkeypatch.setattr("src.agent_server.select_tradier_contract", lambda symbol: None)
    monkeypatch.setattr("src.agent_server.fetch_polygon_option_chain", lambda symbol: pd.DataFrame())

    async def fake_option_chain_cached(symbol, expiration=None):  # noqa: ARG001
        return pd.DataFrame()

    monkeypatch.setattr("src.agent_server.fetch_option_chain_cached", fake_option_chain_cached)
    monkeypatch.setattr("src.tradier.fetch_option_chain_cached", fake_option_chain_cached)
    monkeypatch.setattr("src.agent_server._load_remote_ohlcv", lambda symbol, timeframe: pd.DataFrame())

    universe = ScanUniverse(tickers=["SPX"], style="intraday")
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": [],
        "scheme": "https",
        "server": ("example.com", 443),
    }
    request = Request(scope)
    results = await gpt_scan(universe, request=request, user=AuthedUser(user_id="tester"))

    assert results[0]["fallback_banner"].startswith("Index chain degraded")
    assert results[0]["settlement_note"].startswith("ETF options")
    assert results[0]["execution_proxy"]["symbol"] == "SPY"
