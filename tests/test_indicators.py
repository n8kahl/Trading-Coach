import pandas as pd
import pytest

from src.indicators import get_indicator_bundle


def test_indicator_bundle_uses_typical_price_for_vwap():
    index = pd.date_range("2024-01-01 09:30", periods=3, freq="1min", tz="UTC")
    data = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.5],
            "high": [101.0, 102.0, 103.0],
            "low": [99.5, 100.5, 101.8],
            "close": [100.5, 101.8, 102.7],
            "volume": [1000, 1500, 2000],
        },
        index=index,
    )

    bundle = get_indicator_bundle("TEST", data)

    typical = ((data["high"] + data["low"] + data["close"]) / 3.0).to_numpy()
    volumes = data["volume"].to_numpy()
    expected_vwap = float(((typical * volumes).cumsum() / volumes.cumsum())[-1])

    assert bundle["vwap"] == pytest.approx(expected_vwap, rel=1e-6)
