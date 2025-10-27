from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from src.plans.duration import estimate_expected_duration
from src.config import get_settings
from src import agent_server

from test_trade_detail import _run_fallback_plan


def _make_bars(minutes: int, span: float = 0.6, periods: int = 40) -> pd.DataFrame:
    base = 100.0
    idx = pd.date_range(datetime(2024, 1, 2, 14, 30), periods=periods, freq=f"{minutes}min", tz="UTC")
    high = [base + span / 2] * periods
    low = [base - span / 2] * periods
    close = [base + 0.01 * i for i in range(periods)]
    return pd.DataFrame(
        {
            "open": close,
            "high": high,
            "low": low,
            "close": close,
        },
        index=idx,
    )


def test_estimate_expected_duration_intraday_with_atr() -> None:
    bars_5m = _make_bars(5, span=0.8)
    result = estimate_expected_duration(
        style="intraday",
        interval_hint="5m",
        entry=100.0,
        tp1=102.0,
        atr=1.2,
        em=3.0,
        bars_5m=bars_5m,
        bars_15m=None,
        bars_60m=None,
    )
    assert result["minutes"] == 50
    assert result["label"] == "intraday ~1–2h"
    assert "ATR" in result["basis"]
    assert result["inputs"]["interval"] == "5m"


def test_estimate_expected_duration_scalp_with_em_cap() -> None:
    bars_5m = _make_bars(5, span=0.4)
    result = estimate_expected_duration(
        style="scalp",
        interval_hint="5m",
        entry=50.0,
        tp1=50.4,
        atr=0.8,
        em=0.3,
        bars_5m=bars_5m,
        bars_15m=None,
        bars_60m=None,
    )
    assert result["label"].startswith("scalp")
    assert "EM" in result["basis"]
    assert 10 <= result["minutes"] <= 90


def test_estimate_expected_duration_swing_uses_hourly_bars() -> None:
    bars_60m = _make_bars(60, span=2.4)
    result = estimate_expected_duration(
        style="swing",
        interval_hint="60m",
        entry=120.0,
        tp1=125.0,
        atr=4.0,
        em=None,
        bars_5m=None,
        bars_15m=None,
        bars_60m=bars_60m,
    )
    assert result["label"].startswith("swing")
    assert result["minutes"] >= 390
    assert result["minutes"] <= 390 * 10


def test_estimate_expected_duration_em_cap_increases_minutes() -> None:
    bars_5m = _make_bars(5, span=0.6)
    base = estimate_expected_duration(
        style="intraday",
        interval_hint="5m",
        entry=200.0,
        tp1=202.0,
        atr=1.2,
        em=None,
        bars_5m=bars_5m,
        bars_15m=None,
        bars_60m=None,
    )
    capped = estimate_expected_duration(
        style="intraday",
        interval_hint="5m",
        entry=200.0,
        tp1=202.0,
        atr=1.2,
        em=0.5,
        bars_5m=bars_5m,
        bars_15m=None,
        bars_60m=None,
    )
    assert capped["minutes"] > base["minutes"]
    assert "EM" in capped["basis"]


@pytest.mark.asyncio
async def test_fallback_plan_response_includes_expected_duration(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "gpt_backend_v2_enabled", False, raising=False)
    monkeypatch.setattr(agent_server, "_max_entry_distance_pct", lambda style: 0.1)
    monkeypatch.setattr(agent_server, "is_actionable_soon", lambda *args, **kwargs: True)
    plan_response = await _run_fallback_plan(
        monkeypatch,
        symbol="AAPL",
        expected_move=2.5,
        atr=1.1,
        min_actionability=0.0,
        ema_bias="long",
    )
    assert plan_response is not None
    assert plan_response.targets
    assert isinstance(plan_response.expected_duration, dict)
    assert plan_response.expected_duration["minutes"] >= 10
    assert isinstance(plan_response.plan, dict)
    assert plan_response.plan["expected_duration"] == plan_response.expected_duration
    assert isinstance(plan_response.structured_plan, dict)
    assert plan_response.structured_plan["expected_duration"] == plan_response.expected_duration
    assert isinstance(plan_response.meta, dict)
    assert plan_response.meta["expected_duration"] == plan_response.expected_duration
