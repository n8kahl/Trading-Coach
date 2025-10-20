from datetime import datetime, timezone

from src.plans.geometry import build_plan_geometry


def test_build_plan_geometry_long():
    geometry = build_plan_geometry(
        entry=100.0,
        side="long",
        style="intraday",
        strategy=None,
        atr_tf=2.0,
        atr_daily=3.0,
        iv_expected_move=8.0,
        realized_range=0.5,
        levels={"pdh": 104.0, "swing_low": 99.5},
        timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )
    assert geometry.stop.price < geometry.entry
    assert geometry.targets[0].price > geometry.entry
    assert geometry.runner.fraction > 0
    assert geometry.targets[0].rr_multiple >= 1.6


def test_power_hour_modifier_limits_targets():
    geometry = build_plan_geometry(
        entry=200.0,
        side="long",
        style="intraday",
        strategy="power_hour_trend",
        atr_tf=2.0,
        atr_daily=4.0,
        iv_expected_move=10.0,
        realized_range=0.6,
        levels={"pdh": 206.0, "swing_low": 199.2},
        timestamp=datetime(2025, 1, 1, 20, 30, tzinfo=timezone.utc),
    )
    assert len(geometry.targets) == 2
    assert geometry.runner.atr_trail_mult < 1.0
    assert geometry.runner.fraction > 0
    assert geometry.targets[0].rr_multiple >= 1.6


def test_gap_fill_prefers_gap_level():
    geometry = build_plan_geometry(
        entry=150.0,
        side="short",
        style="intraday",
        strategy="gap_fill_open",
        atr_tf=2.0,
        atr_daily=4.0,
        iv_expected_move=10.0,
        realized_range=0.5,
        levels={"gap_fill": 145.0, "session_low": 144.0, "session_high": 156.0},
        timestamp=datetime(2025, 1, 1, 14, 0, tzinfo=timezone.utc),
    )
    assert round(geometry.targets[0].price, 2) == 145.0
    assert geometry.targets[0].reason == "gap_fill"
    assert geometry.targets[1].price < geometry.targets[0].price
