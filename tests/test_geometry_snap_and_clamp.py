import math
from datetime import datetime, timezone

import pytest

from src.plans.clamp import clamp_targets_to_em, ensure_monotonic
from src.plans.snap import snap_targets, stop_from_structure


def test_clamp_targets_strong_snap_allows_extended_em():
    entry = 100.0
    raw_targets = [112.0, 115.0, 118.0]
    em_points = 10.0
    snap_tags = {"VAH"}

    clamped = clamp_targets_to_em(entry, "long", raw_targets, em_points, snap_tags, "intraday")

    assert clamped == pytest.approx([110.8, 110.9, 111.0], rel=0, abs=1e-6)


def test_clamp_targets_without_strong_snap_strict_em():
    entry = 100.0
    raw_targets = [115.0, 118.0]
    em_points = 10.0

    clamped = clamp_targets_to_em(entry, "long", raw_targets, em_points, {"SWING"}, "intraday")

    assert clamped == pytest.approx([109.9, 110.0], rel=0, abs=1e-6)


def test_stop_from_structure_short_prefers_orh_buffer():
    entry = 118.5
    levels = {"orh": 120.0}
    stop_price, label = stop_from_structure(entry, "short", levels, atr_value=0.8, style="intraday")

    assert label == "ORH"
    assert stop_price == pytest.approx(120.15, rel=0, abs=1e-2)
    assert stop_price >= entry + 0.6 * 0.8


def test_stop_from_structure_long_prefers_orl_buffer():
    entry = 102.0
    levels = {"orl": 99.75}
    stop_price, label = stop_from_structure(entry, "long", levels, atr_value=0.5, style="intraday")

    assert label == "ORL"
    assert stop_price == pytest.approx(99.60, rel=0, abs=1e-2)
    assert stop_price <= entry - 0.6 * 0.5


def test_snap_targets_uses_structural_nodes():
    entry = 215.25
    levels = {
        "orl": 214.89,
        "session_low": 214.56,
        "vah": 218.0,
        "val": 212.4,
    }
    raw_targets = [214.7, 214.4, 213.9]

    snapped, reasons, tags = snap_targets(entry, "short", raw_targets, levels, atr_value=0.42, style="intraday")

    assert snapped[0] == pytest.approx(214.89, rel=0, abs=1e-2)
    assert snapped[1] == pytest.approx(214.56, rel=0, abs=1e-2)
    assert "ORL" in reasons[0]["reason"].upper()
    assert "SESSION_LOW" in reasons[1]["reason"].upper()
    assert {"ORL", "SESSION_LOW"} <= {tag.upper() for tag in tags}
