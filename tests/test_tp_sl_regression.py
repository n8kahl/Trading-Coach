from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.plans.pipeline import build_structured_geometry


def test_geometry_stable_without_htf_levels() -> None:
    levels = {
        "prev_high": 105.0,
        "prev_low": 95.0,
        "opening_range_high": 101.0,
        "opening_range_low": 99.0,
        "session_high": 104.2,
        "session_low": 97.4,
        "vah": 103.6,
        "val": 98.7,
        "poc": 100.5,
    }

    geometry = build_structured_geometry(
        symbol="AAPL",
        style="intraday",
        direction="long",
        entry=100.0,
        levels=levels,
        atr_value=1.8,
        plan_time=datetime(2024, 3, 5, 15, 30, tzinfo=timezone.utc),
        raw_targets=[103.0, 104.5, 106.0],
        rr_floor=1.2,
        em_hint=4.0,
    )

    assert geometry.stop == pytest.approx(97.13, rel=0, abs=1e-2)
    targets = [round(tp, 2) for tp in geometry.targets]
    assert targets[0] == pytest.approx(103.6, rel=0, abs=1e-2)
    assert len(targets) >= 2
    assert targets[-1] <= 104.4 + 1e-2
    assert geometry.runner_policy["fraction"] == pytest.approx(0.2)
    assert geometry.runner_policy["atr_trail_mult"] == pytest.approx(1.0)
    assert "VAH" in geometry.snap_tags


def test_htf_levels_snap_targets_when_near_em_cap() -> None:
    base_levels = {
        "opening_range_high": 100.3,
        "opening_range_low": 99.7,
        "session_high": 100.5,
        "session_low": 99.2,
    }

    geometry_without_htf = build_structured_geometry(
        symbol="QQQ",
        style="intraday",
        direction="long",
        entry=100.0,
        levels=base_levels,
        atr_value=1.0,
        plan_time=datetime(2024, 3, 7, 15, 15, tzinfo=timezone.utc),
        raw_targets=[100.9, 102.0, 103.0],
        rr_floor=1.1,
        em_hint=3.0,
    )

    levels_with_close_htf = dict(base_levels)
    levels_with_close_htf["pwh"] = 101.0
    geometry_with_close_htf = build_structured_geometry(
        symbol="QQQ",
        style="intraday",
        direction="long",
        entry=100.0,
        levels=levels_with_close_htf,
        atr_value=1.0,
        plan_time=datetime(2024, 3, 7, 15, 15, tzinfo=timezone.utc),
        raw_targets=[100.9, 102.0, 103.0],
        rr_floor=1.1,
        em_hint=3.0,
    )

    levels_with_far_htf = dict(base_levels)
    levels_with_far_htf["pwh"] = 101.5
    geometry_with_far_htf = build_structured_geometry(
        symbol="QQQ",
        style="intraday",
        direction="long",
        entry=100.0,
        levels=levels_with_far_htf,
        atr_value=1.0,
        plan_time=datetime(2024, 3, 7, 15, 15, tzinfo=timezone.utc),
        raw_targets=[100.9, 102.0, 103.0],
        rr_floor=1.1,
        em_hint=3.0,
    )

    base_targets = [round(tp, 2) for tp in geometry_without_htf.targets]
    close_targets = [round(tp, 2) for tp in geometry_with_close_htf.targets]
    far_targets = [round(tp, 2) for tp in geometry_with_far_htf.targets]

    assert close_targets[0] == pytest.approx(levels_with_close_htf["pwh"])
    assert far_targets[0] == base_targets[0]
