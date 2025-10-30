from datetime import datetime, timezone

import pytest

from src.plans.pipeline import build_structured_geometry


def test_structured_geometry_reason_shapes_and_stop_meta():
    levels = {
        "opening_range_high": 401.5,
        "opening_range_low": 398.2,
        "session_high": 402.3,
        "session_low": 397.4,
        "val": 399.1,
        "vah": 401.8,
        "poc": 400.4,
    }
    geometry = build_structured_geometry(
        symbol="SPY",
        style="intraday",
        direction="long",
        entry=400.0,
        levels=levels,
        atr_value=1.2,
        plan_time=datetime(2024, 4, 12, 15, 30, tzinfo=timezone.utc),
        raw_targets=[401.2, 402.6, 404.5],
        rr_floor=1.6,
        em_hint=4.5,
    )

    assert isinstance(geometry.stop_meta, dict)
    assert set(geometry.stop_meta.keys()) >= {"anchor", "wick_buffer", "atr_floor", "final", "source"}
    assert geometry.stop_meta["final"] == pytest.approx(geometry.stop, rel=0, abs=1e-6)

    allowed_reason_keys = {
        "label",
        "reason",
        "snap_tag",
        "ideal_price",
        "ideal_distance",
        "ideal_fraction",
        "snap_price",
        "snap_distance",
        "snap_fraction",
        "snap_deviation",
        "candidate_nodes",
        "synthetic",
        "selected_node",
        "rr_multiple",
        "fraction",
        "em_cap_relaxed",
        "outside_ideal_band",
        "no_structural",
        "modifiers",
        "rr_floor",
        "distance",
        "synthetic_meta",
        "watch_plan",
        "raw_rr_multiple",
        "snap_rr_multiple",
    }

    for reason in geometry.tp_reasons:
        assert set(reason.keys()) <= allowed_reason_keys
        assert "label" in reason and "reason" in reason
        modifiers = reason.get("modifiers")
        if modifiers:
            assert isinstance(modifiers, list)
            for modifier in modifiers:
                assert set(modifier.keys()) <= {"reason", "meta"}
        candidate_nodes = reason.get("candidate_nodes")
        if candidate_nodes:
            assert isinstance(candidate_nodes, list)
            for node in candidate_nodes:
                assert set(node.keys()) >= {"label", "price", "distance", "picked"}
                decisions = node.get("decisions")
                if decisions:
                    assert all(isinstance(entry, dict) and "reason" in entry for entry in decisions)

    tp1 = geometry.tp_reasons[0]
    assert tp1.get("candidate_nodes") is None or isinstance(tp1["candidate_nodes"], list)
