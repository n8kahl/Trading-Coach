import pytest

from src.plans.snap import snap_targets


def _candidate_has_decision(reason_entry: dict[str, object], label: str, code: str) -> bool:
    nodes = reason_entry.get("candidate_nodes")
    if not isinstance(nodes, list):
        return False
    for node in nodes:
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


def test_tp1_respects_rr_floor_rejects_low_reward_extreme():
    entry = 200.0
    expected_move = 8.0
    levels = {
        "session_high": 202.5,
        "vah": 203.8,
    }
    raw_targets = [202.4, 203.6, 205.5]

    snapped, reasons, snap_tags = snap_targets(
        entry=entry,
        direction="long",
        raw_tps=raw_targets,
        levels=levels,
        atr_value=1.4,
        style="intraday",
        expected_move=expected_move,
        max_em_fraction=0.60,
        stop_price=198.0,
        rr_floor=1.8,
    )

    assert snapped[0] == pytest.approx(203.8, rel=0, abs=1e-2)
    reason = reasons[0]
    assert reason.get("selected_node") == "VAH"
    assert reason.get("rr_multiple") and float(reason["rr_multiple"]) >= 1.8
    assert _candidate_has_decision(reason, "SESSION_HIGH", "REJECT_RR_FLOOR")
    assert reason.get("snap_tag") == "VAH"
    assert "VAH" in {tag.upper() for tag in snap_tags}
