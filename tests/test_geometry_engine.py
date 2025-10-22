import pytest

from src.core.geometry_engine import summarize_plan_geometry
from src.plans.geometry import build_plan_geometry


def test_summarize_plan_geometry_returns_expected_metrics():
    levels = {
        "vah": 101.2,
        "val": 98.8,
        "orh": 100.5,
        "orl": 99.1,
        "poc": 100.0,
    }
    geometry = build_plan_geometry(
        entry=100.0,
        side="long",
        style="intraday",
        strategy=None,
        atr_tf=1.5,
        atr_daily=2.0,
        iv_expected_move=None,
        realized_range=0.8,
        levels=levels,
        timestamp=None,
        em_points=2.4,
    )

    summary = summarize_plan_geometry(
        geometry,
        entry=geometry.entry,
        atr_value=1.5,
        expected_move=2.4,
        key_levels_used={"structural": [{"role": "ORL", "price": 99.1}]},
    )

    assert summary.entry == pytest.approx(geometry.entry)
    assert summary.stop == pytest.approx(geometry.stop.price)
    assert summary.atr_used == pytest.approx(1.5)
    assert summary.expected_move == pytest.approx(2.4)
    assert summary.targets, "targets should not be empty"
    assert isinstance(summary.snap_trace, list)
    assert "structural" in summary.key_levels_used
