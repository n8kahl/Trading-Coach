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
        iv_expected_move=6.0,
        realized_range=1.0,
        levels={"pdl": 98.5, "pdc": 99.2, "session_low": 99.0, "session_high": 103.0},
        timestamp=datetime(2025, 1, 1, tzinfo=timezone.utc),
    )
    assert geometry.stop.price < geometry.entry
    assert geometry.targets[0].price > geometry.entry
    assert geometry.runner.fraction > 0
