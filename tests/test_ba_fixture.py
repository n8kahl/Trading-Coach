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

    assert structured.stop == pytest.approx(217.53, rel=0, abs=1e-2)
    assert structured.targets[0] == pytest.approx(214.89, rel=0, abs=1e-2)
    assert structured.targets[1] == pytest.approx(214.56, rel=0, abs=1e-2)
    assert abs(structured.targets[2] - entry) <= 1.20 * em_hint + 1e-6
    session_roles = [item["label"] for item in structured.key_levels_used.get("session", [])]
    assert "ORH" in session_roles
    assert any("ORL" in reason["reason"].upper() for reason in structured.tp_reasons)
