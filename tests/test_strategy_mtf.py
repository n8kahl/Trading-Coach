from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from src.features.htf_levels import HTFLevels
from src.features.mtf import MTFBundle, TFState
from src.strategy.engine import infer_strategy


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])


def test_infer_strategy_prefers_power_hour_with_supportive_mtf() -> None:
    bundle = MTFBundle(
        by_tf={
            "5m": TFState(tf="5m", ema_up=True, ema_down=False, adx_slope=0.4, vwap_rel="above", atr=0.7),
            "15m": TFState(tf="15m", ema_up=True, ema_down=False, adx_slope=0.3, vwap_rel="above", atr=1.2),
            "60m": TFState(tf="60m", ema_up=True, ema_down=False, adx_slope=0.25, vwap_rel="near", atr=2.4),
            "D": TFState(tf="D", ema_up=True, ema_down=False, adx_slope=0.2, vwap_rel="above", atr=3.6),
        },
        bias_htf="long",
        agreement=0.75,
        notes=["D↑, 60m↑, 15m↑", "VWAP>"],
    )
    ctx = {
        "symbol": "AAPL",
        "timestamp": datetime(2024, 3, 5, 20, 5, tzinfo=timezone.utc),  # 3:05pm ET
        "mtf": bundle,
        "htf_levels": HTFLevels(pdh=181.2, pdl=176.4, pdc=178.5, pwh=None, pwl=None, pwc=None),
        "price": 179.3,
        "vwap": 178.9,
        "opening_range_high": 180.0,
        "opening_range_low": 177.8,
        "bars_5m": _empty_frame(),
        "bars_15m": _empty_frame(),
        "bars_60m": _empty_frame(),
    }

    selected_id, profile = infer_strategy("long", ctx)

    assert selected_id == "power_hour_trend"
    assert profile.waiting_for == "5m close above VWAP + acceptance over ORH"
    assert profile.mtf is not None
    assert profile.mtf["bias"] == "long"
    assert profile.mtf["agreement"] == pytest.approx(1.0)
    assert profile.mtf["notes"] == ["D↑, 60m↑, 15m↑", "VWAP>"]
    assert any(rule.id == "power_hour_trend" and rule.score > 0.6 for rule in profile.matched_rules)


def test_infer_strategy_falls_back_when_mtf_conflicts() -> None:
    conflicting_bundle = MTFBundle(
        by_tf={
            "5m": TFState(tf="5m", ema_up=False, ema_down=True, adx_slope=-0.4, vwap_rel="below", atr=0.7),
            "15m": TFState(tf="15m", ema_up=False, ema_down=True, adx_slope=-0.3, vwap_rel="below", atr=1.1),
            "60m": TFState(tf="60m", ema_up=False, ema_down=True, adx_slope=-0.2, vwap_rel="below", atr=2.2),
            "D": TFState(tf="D", ema_up=False, ema_down=True, adx_slope=-0.15, vwap_rel="below", atr=3.1),
        },
        bias_htf="short",
        agreement=0.2,
        notes=["D↓, 60m↓, 15m↓", "VWAP<"],
    )
    ctx = {
        "symbol": "AAPL",
        "timestamp": datetime(2024, 3, 5, 20, 5, tzinfo=timezone.utc),
        "mtf": conflicting_bundle,
        "htf_levels": HTFLevels(pdh=181.2, pdl=176.4, pdc=178.5, pwh=None, pwl=None, pwc=None),
        "price": 179.3,
        "vwap": 178.9,
        "opening_range_high": 180.0,
        "opening_range_low": 177.8,
        "bars_5m": _empty_frame(),
        "bars_15m": _empty_frame(),
        "bars_60m": _empty_frame(),
    }

    selected_id, profile = infer_strategy("long", ctx)

    assert selected_id == "baseline_auto"
    assert profile.name == "Baseline Geometry"
    assert profile.mtf is not None
    assert profile.mtf["bias"] == "short"
    assert profile.mtf["agreement"] == pytest.approx(0.0)
    assert profile.mtf["notes"] == ["D↓, 60m↓, 15m↓", "VWAP<"]
