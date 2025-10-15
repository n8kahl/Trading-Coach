import pandas as pd
import pytest

from src.app.engine.ratio_engine import RatioEngine, RatioSnapshot


@pytest.mark.asyncio
async def test_ratio_engine_computes_gamma_with_translation(monkeypatch):
    index_times = pd.date_range("2024-01-01 14:30", periods=60, freq="T", tz="UTC")
    index_prices = pd.Series(range(4000, 4060), index=index_times, dtype=float)
    proxy_prices = index_prices * 0.1
    index_df = pd.DataFrame({"close": index_prices})
    proxy_df = pd.DataFrame({"close": proxy_prices})

    async def fake_fetch(symbol: str, timeframe: str):  # noqa: ARG001
        if symbol == "I:SPX":
            return index_df
        if symbol == "SPY":
            return proxy_df
        raise AssertionError(f"Unexpected symbol {symbol}")

    monkeypatch.setattr("src.app.engine.ratio_engine.fetch_polygon_ohlcv", fake_fetch)

    engine = RatioEngine(lookback_minutes=60, refresh_seconds=10)
    snapshot = await engine.snapshot("SPX")

    assert isinstance(snapshot, RatioSnapshot)
    assert pytest.approx(snapshot.spot_ratio, rel=1e-3) == 0.1
    assert pytest.approx(snapshot.gamma_current, rel=1e-3) == 1.0
    assert pytest.approx(snapshot.gamma_mean, rel=1e-3) == 1.0
    translated = engine.translate_level(4010.0, snapshot)
    assert pytest.approx(translated, rel=1e-3) == 401.0


@pytest.mark.asyncio
async def test_ratio_engine_detects_gamma_drift(monkeypatch):
    index_times = pd.date_range("2024-01-01 14:30", periods=60, freq="T", tz="UTC")
    base_prices = pd.Series(range(1200, 1260), index=index_times, dtype=float)
    index_df = pd.DataFrame({"close": base_prices})
    # Introduce drift on the final 10 samples
    proxy_series = base_prices * 0.1
    proxy_series.iloc[-10:] = proxy_series.iloc[-10:] * 1.05
    proxy_df = pd.DataFrame({"close": proxy_series})

    async def fake_fetch(symbol: str, timeframe: str):  # noqa: ARG001
        if symbol == "I:NDX":
            return index_df
        if symbol == "QQQ":
            return proxy_df
        raise AssertionError(f"Unexpected symbol {symbol}")

    monkeypatch.setattr("src.app.engine.ratio_engine.fetch_polygon_ohlcv", fake_fetch)

    engine = RatioEngine(lookback_minutes=60, refresh_seconds=10)
    snapshot = await engine.snapshot("NDX")

    assert snapshot is not None
    assert snapshot.drift != 0.0
    # Drift should be positive because proxy outran the index
    assert snapshot.drift > 0
