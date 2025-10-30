import pytest

from src.plans.snap import snap_targets


def _modifier_reason(reason_entry: dict[str, object], label: str) -> bool:
    modifiers = reason_entry.get("modifiers") or []
    if not isinstance(modifiers, list):
        return False
    for item in modifiers:
        if isinstance(item, dict) and item.get("reason") == label:
            return True
    return False


def _candidate_decision(reason_entry: dict[str, object], label: str, code: str) -> bool:
    candidates = reason_entry.get("candidate_nodes")
    if not isinstance(candidates, list):
        return False
    for node in candidates:
        if not isinstance(node, dict):
            continue
        if str(node.get("label")).upper() != label:
            continue
        decisions = node.get("decisions")
        if not isinstance(decisions, list):
            continue
        if any(isinstance(entry, dict) and entry.get("reason") == code for entry in decisions):
            return True
    return False


def test_tp1_prefers_interior_over_session_extreme():
    entry = 640.0
    expected_move = 6.0
    levels = {
        "session_low": 637.3,
        "val": 637.1,
    }
    raw_targets = [637.4, 635.5, 633.0]

    snapped, reasons, snap_tags = snap_targets(
        entry=entry,
        direction="short",
        raw_tps=raw_targets,
        levels=levels,
        atr_value=1.2,
        style="intraday",
        expected_move=expected_move,
        max_em_fraction=0.60,
        stop_price=641.0,
        rr_floor=1.8,
    )

    assert snapped[0] == pytest.approx(637.1, rel=0, abs=1e-2)
    reason = reasons[0]
    assert reason.get("selected_node") == "VAL"
    assert reason.get("rr_multiple") and float(reason["rr_multiple"]) >= 1.8
    assert reason.get("distance") and float(reason["distance"]) <= expected_move * 0.75 + 1e-9
    assert _modifier_reason(reason, "PREFER_INTERIOR_OVER_EXTREME")
    assert _candidate_decision(reason, "SESSION_LOW", "REPLACED_BY_INTERIOR")
    assert "VAL" in {tag.upper() for tag in snap_tags}
