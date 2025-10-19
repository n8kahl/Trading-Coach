from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.services.contract_rules import ContractRuleBook
from src.services.scan_engine import PlanningScanEngine
from src.services.persist import PlanningPersistence
from src.services.polygon_client import AggregatesResult
from src.services.universe import UniverseSnapshot


class StubPolygonClient:
    def __init__(self, frames):
        self.frames = frames

    async def fetch_many(self, symbols, timeframes):
        return {symbol: self.frames[symbol] for symbol in symbols if symbol in self.frames}


def make_frame(prices):
    ts = [datetime.now(timezone.utc) - timedelta(days=len(prices) - idx) for idx in range(len(prices))]
    data = {
        "open": prices,
        "high": [p * 1.01 for p in prices],
        "low": [p * 0.99 for p in prices],
        "close": prices,
        "volume": [1_000_000 for _ in prices],
    }
    frame = pd.DataFrame(data, index=pd.to_datetime(ts))
    frame.sort_index(inplace=True)
    return frame


@pytest.mark.asyncio
async def test_planning_scan_engine_scores_candidates(monkeypatch):
    prices = [150 + i for i in range(60)]
    daily_frame = make_frame(prices)
    intraday_60 = make_frame(prices[-60:])
    intraday_30 = make_frame(prices[-30:])
    frames = {
        "AAPL": AggregatesResult(symbol="AAPL", windows={"1d": daily_frame, "60": intraday_60, "30": intraday_30}),
        "I:SPX": AggregatesResult(symbol="I:SPX", windows={"1d": daily_frame}),
        "I:NDX": AggregatesResult(symbol="I:NDX", windows={"1d": daily_frame}),
        "I:RUT": AggregatesResult(symbol="I:RUT", windows={"1d": daily_frame}),
        "I:VIX": AggregatesResult(symbol="I:VIX", windows={"1d": daily_frame}),
    }
    stub_client = StubPolygonClient(frames)
    persistence = PlanningPersistence()  # operates in no-op mode without DB
    engine = PlanningScanEngine(stub_client, persistence, rulebook=ContractRuleBook(), min_readiness=0.2)
    universe = UniverseSnapshot(
        name="sp500",
        source="test",
        as_of_utc=datetime.now(timezone.utc),
        symbols=["AAPL"],
        metadata={},
    )
    result = await engine.run(universe, style="intraday")
    assert result.candidates, "Expected at least one planning candidate"
    candidate = result.candidates[0]
    assert candidate.symbol == "AAPL"
    assert 0.0 <= candidate.readiness_score <= 1.0
