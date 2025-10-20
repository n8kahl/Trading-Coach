from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from src.features.mtf import MTFBundle, compute_mtf_bundle


def _build_trending_frame(
    *,
    start: datetime,
    periods: int,
    freq: str,
    price_start: float,
    step: float,
) -> pd.DataFrame:
    index = pd.date_range(start=start, periods=periods, freq=freq, tz="UTC")
    closes = [price_start + step * idx for idx in range(periods)]
    highs = [close + 0.4 for close in closes]
    lows = [close - 0.4 for close in closes]
    opens = closes[:-1] + [closes[-1]]
    volume = [1_000_000 + 5_000 * idx for idx in range(periods)]
    return pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volume,
        },
        index=index,
    )


def test_compute_mtf_bundle_trending_up() -> None:
    start = datetime(2024, 3, 1, 14, 0, tzinfo=timezone.utc)
    bars_5m = _build_trending_frame(start=start, periods=120, freq="5min", price_start=100.0, step=0.05)
    bars_15m = _build_trending_frame(start=start, periods=80, freq="15min", price_start=99.0, step=0.12)
    bars_60m = _build_trending_frame(start=start, periods=80, freq="h", price_start=98.0, step=0.4)
    bars_d = _build_trending_frame(start=start - timedelta(days=80), periods=80, freq="D", price_start=95.0, step=0.8)

    bundle = compute_mtf_bundle(
        "AAPL",
        bars_5m,
        bars_15m,
        bars_60m,
        bars_d,
        vwap_5m=float(bars_5m["close"].tail(10).mean()),
    )
    assert isinstance(bundle, MTFBundle)
    assert bundle.bias_htf == "long"
    assert pytest.approx(bundle.agreement, rel=1e-3) == 0.5  # default until amplified
    assert bundle.by_tf["5m"].ema_up is True
    assert bundle.by_tf["60m"].ema_up is True
    assert bundle.by_tf["D"].ema_up is True
    assert bundle.by_tf["5m"].vwap_rel in {"above", "near"}
    assert "5m↑" in bundle.notes[0]
    assert bundle.notes[1] in {"VWAP>", "VWAP≈"}


def test_compute_mtf_bundle_trending_down() -> None:
    start = datetime(2024, 3, 1, 14, 0, tzinfo=timezone.utc)
    bars_5m = _build_trending_frame(start=start, periods=120, freq="5min", price_start=100.0, step=-0.05)
    bars_15m = _build_trending_frame(start=start, periods=80, freq="15min", price_start=101.0, step=-0.12)
    bars_60m = _build_trending_frame(start=start, periods=80, freq="h", price_start=102.0, step=-0.32)
    bars_d = _build_trending_frame(start=start - timedelta(days=80), periods=80, freq="D", price_start=105.0, step=-0.7)

    bundle = compute_mtf_bundle(
        "MSFT",
        bars_5m,
        bars_15m,
        bars_60m,
        bars_d,
        vwap_5m=float(bars_5m["close"].tail(10).mean()),
    )
    assert isinstance(bundle, MTFBundle)
    assert bundle.bias_htf == "short"
    assert bundle.by_tf["5m"].ema_down is True
    assert bundle.by_tf["60m"].ema_down is True
    assert bundle.by_tf["D"].ema_down is True
    assert bundle.by_tf["5m"].vwap_rel in {"below", "near"}
    assert "5m↓" in bundle.notes[0]
    assert bundle.notes[1] in {"VWAP<", "VWAP≈"}
