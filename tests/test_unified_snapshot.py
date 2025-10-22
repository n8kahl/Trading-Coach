import pandas as pd
import pytest

from src.core.unified_snapshot import get_unified_snapshot


async def _fake_fetch(symbol: str, timeframe: str, *, max_days=None):
    base = 100.0
    if "QQQ" in symbol.upper():
        base = 300.0
    elif "SPY" in symbol.upper():
        base = 400.0
    elif "VIX" in symbol.upper():
        base = 20.0

    if timeframe in {"5", "5m"}:
        freq = "5min"
        periods = 20
    else:
        freq = "1min"
        periods = 60

    index = pd.date_range("2024-01-01", periods=periods, freq=freq, tz="UTC")
    data = {
        "open": [base + i * 0.1 for i in range(periods)],
        "high": [base + 0.5 + i * 0.1 for i in range(periods)],
        "low": [base - 0.5 + i * 0.1 for i in range(periods)],
        "close": [base + 0.2 + i * 0.1 for i in range(periods)],
        "volume": [1_000_000 + i * 1_000 for i in range(periods)],
    }
    frame = pd.DataFrame(data, index=index)
    frame.attrs["source"] = "polygon"
    return frame


@pytest.mark.asyncio
async def test_unified_snapshot_basic(monkeypatch):
    monkeypatch.setattr("src.core.unified_snapshot.fetch_polygon_ohlcv", _fake_fetch)
    snapshot = await get_unified_snapshot(["AAPL"])
    assert "AAPL" in snapshot.symbols
    snap = snapshot.symbols["AAPL"]
    assert snap.last_price is not None
    assert snap.atr14 is not None
    assert snap.expected_move is not None
    bars = snap.get_bars("1m")
    assert isinstance(bars, pd.DataFrame)
    assert not bars.empty
    assert snapshot.indices
    assert snapshot.volatility["value"] is not None
    summary = snapshot.summary()
    assert summary["symbol_count"] == 1
