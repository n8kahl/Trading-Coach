from datetime import datetime, timezone

import pytest

from src.plans.pipeline import build_structured_geometry


def test_ba_structured_geometry_snaps_to_session_levels():
    entry = 215.25
    style = "intraday"
    direction = "short"
    atr = 0.4258
    em_hint = 0.7238
    levels = {
        "orh": 217.38,
        "orl": 214.89,
        "session_low": 214.56,
        "prev_low": 208.20,
        "prev_close": 215.25,
    }
    raw_targets = [214.7, 214.4, 214.1]

    structured = build_structured_geometry(
        symbol="BA",
        style=style,
        direction=direction,
        entry=entry,
        levels=levels,
        atr_value=atr,
        plan_time=datetime(2025, 10, 20, 15, 56, 58, tzinfo=timezone.utc),
        raw_targets=raw_targets,
        rr_floor=1.6,
        em_hint=em_hint,
    )

    assert structured.stop == pytest.approx(217.44, rel=0, abs=1e-2)
    assert structured.targets[0] == pytest.approx(215.0, rel=0, abs=1e-2)
    assert structured.targets[1] == pytest.approx(214.75, rel=0, abs=1e-2)
    assert structured.targets[2] == pytest.approx(214.55, rel=0, abs=1e-2)
    session_roles = [item["label"] for item in structured.key_levels_used.get("session", [])]
    assert "ORH" in session_roles
    reason_tp1 = structured.tp_reasons[0]
    assert str(reason_tp1.get("reason", "")).startswith("SYNTHETIC_EM_BUCKET")
    candidate_nodes = reason_tp1.get("candidate_nodes") or []
    assert any(
        node.get("label") == "ORL" and any(dec.get("reason") == "REJECT_RR_FLOOR" for dec in node.get("decisions", []))
        for node in candidate_nodes
    )
    assert {"WATCH_PLAN", "TP1 RR below floor"} <= set(structured.warnings)
    assert structured.tp_reasons[0].get("watch_plan") == "true"
