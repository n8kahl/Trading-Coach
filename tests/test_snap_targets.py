from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.plans.pipeline import build_structured_geometry
from src.plans.snap import MIN_ATR_MULT_TP1, snap_targets


def test_tp1_prefers_nearest_structure_over_extreme_when_rr_ok() -> None:
    entry = 100.0
    stop = 101.0  # 1.0R risk for short bias
    expected_move = 3.0
    levels = {
        "orl": 99.1,
        "val": 99.05,
        "session_low": 98.0,
    }

    snapped, reasons, _ = snap_targets(
        entry=entry,
        direction="short",
        raw_tps=[98.6, 97.5, 96.0],
        levels=levels,
        atr_value=1.2,
        style="intraday",
        expected_move=expected_move,
        max_em_fraction=None,
        stop_price=stop,
        rr_floor=0.6,
    )

    assert snapped, "TP1 should be available"
    assert snapped[0] == pytest.approx(99.1, rel=1e-6)

    reason = reasons[0]
    assert reason["snap_tag"] == "ORL"
    assert reason["selected_node"] == "ORL"
    assert reason["reason"].startswith("Nearest structure clearing RR")
    assert reason["rr_multiple"] >= 0.9

    nodes = {node["label"]: node["picked"] for node in reason["candidate_nodes"]}
    assert nodes["ORL"] is True
    assert nodes["SESSION_LOW"] is False


@pytest.mark.parametrize(
    "style, expected_move, fail_distance, pass_distance",
    (
        ("scalp", 4.0, 0.4, 1.1),
        ("intraday", 4.0, 0.7, 1.2),
        ("swing", 4.0, 0.8, 1.5),
        ("leaps", 4.0, 0.9, 1.8),
    ),
)
def test_tp1_rr_floor_applied_per_style(
    style: str,
    expected_move: float,
    fail_distance: float,
    pass_distance: float,
) -> None:
    entry = 100.0
    stop = entry + 1.0
    levels = {
        "val": entry - fail_distance,
        "session_low": entry - pass_distance,
    }

    snapped, reasons, _ = snap_targets(
        entry=entry,
        direction="short",
        raw_tps=[entry - pass_distance],
        levels=levels,
        atr_value=1.5,
        style=style,
        expected_move=expected_move,
        stop_price=stop,
        rr_floor=0.3,
    )

    assert snapped, "TP1 should be populated"
    expected_price = round(entry - pass_distance, 2)
    assert snapped[0] == pytest.approx(expected_price, rel=1e-6)

    reason = reasons[0]
    assert reason["snap_tag"] == "SESSION_LOW"
    assert reason["rr_multiple"] >= MIN_ATR_MULT_TP1[style]

    nodes = {node["label"]: node["picked"] for node in reason["candidate_nodes"]}
    assert nodes["SESSION_LOW"] is True
    assert nodes["VAL"] is False


def test_tp1_respects_em_cap_and_max_fraction() -> None:
    geometry = build_structured_geometry(
        symbol="TEST",
        style="scalp",
        direction="short",
        entry=100.0,
        levels={"session_low": 96.0, "orh": 101.5},
        atr_value=1.0,
        plan_time=datetime(2024, 1, 2, tzinfo=timezone.utc),
        raw_targets=[98.6],
        rr_floor=0.4,
        em_hint=4.0,
    )

    assert geometry.targets[0] == pytest.approx(98.6, rel=1e-6)
    assert geometry.tp_reasons[0]["reason"].endswith("Snap skipped")
