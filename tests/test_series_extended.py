from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

import src.providers.series as series_module
from src.providers.series import SeriesBundle


def _make_frame(index: pd.DatetimeIndex, base: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "open": base - 0.3,
            "high": base + 0.6,
            "low": base - 0.5,
            "close": base,
            "volume": 100_000,
        },
        index=index,
    )


@pytest.mark.asyncio
async def test_fetch_series_extended_marks_bundle(monkeypatch: pytest.MonkeyPatch) -> None:
    as_of = datetime(2024, 6, 10, 20, 0, tzinfo=timezone.utc)
    symbol = "SPY"

    daily_index = pd.date_range(end=as_of, periods=15, freq="1D", tz="UTC")
    sixtyfive_index = pd.date_range(end=as_of, periods=30, freq="65min", tz="UTC")
    fifteen_index = pd.date_range(end=as_of, periods=48, freq="15min", tz="UTC")
    five_index = pd.date_range(end=as_of, periods=144, freq="5min", tz="UTC")

    frames = {
        "1d": _make_frame(daily_index, 440.0),
        "65": _make_frame(sixtyfive_index, 442.0),
        "15": _make_frame(fifteen_index, 441.0),
        "5": _make_frame(five_index, 441.5),
    }

    captured: list[tuple[str, bool]] = []

    async def fake_fetch_polygon(symbol_input: str, timeframe: str, *, max_days=None, include_extended=False):  # type: ignore[override]
        captured.append((timeframe, include_extended))
        frame = frames.get(timeframe)
        if frame is None:
            raise AssertionError(f"unexpected timeframe requested: {timeframe}")
        return frame.copy()

    monkeypatch.setattr(series_module, "fetch_polygon_ohlcv", fake_fetch_polygon)

    bundle = await series_module.fetch_series([symbol], mode="live", as_of=as_of, extended=True)

    assert bundle.extended is True
    assert ("5", True) in captured
    assert ("15", True) in captured
    assert ("1d", False) in captured

    frame_5m = bundle.get_frame(symbol, "5m")
    assert frame_5m is not None
    earliest = frame_5m.index.min()
    assert earliest < pd.Timestamp("2024-06-10T13:30:00Z")
    assert frame_5m.index.max() <= pd.Timestamp(as_of)
