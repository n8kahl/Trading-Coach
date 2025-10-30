import pytest

from src.plans.snap import snap_targets


def _modifier_meta(reason_entry: dict[str, object], label: str) -> dict[str, object] | None:
    modifiers = reason_entry.get("modifiers")
    if not isinstance(modifiers, list):
        return None
    for item in modifiers:
        if not isinstance(item, dict):
            continue
        if item.get("reason") == label:
            return item.get("meta") if isinstance(item.get("meta"), dict) else {}
    return None


def test_tp1_respects_em_cap_and_synthesizes_when_needed():
    entry = 500.0
    expected_move = 4.0
    levels = {
        "vah": 503.4,
        "session_high": 503.8,
        "pwh": 504.1,
    }
    raw_targets = [503.5, 504.5, 505.0]

    snapped, reasons, snap_tags = snap_targets(
        entry=entry,
        direction="long",
        raw_tps=raw_targets,
        levels=levels,
        atr_value=1.1,
        style="intraday",
        expected_move=expected_move,
        max_em_fraction=0.60,
        stop_price=498.2,
        rr_floor=1.6,
    )

    reason = reasons[0]
    assert str(reason.get("reason", "")).startswith("SYNTHETIC_EM_BUCKET")
    assert reason.get("synthetic") is True
    assert reason.get("watch_plan") == "true"
    meta = _modifier_meta(reason, "SYNTHETIC_EM_BUCKET")
    assert meta is not None
    assert meta.get("cap") == "≤0.40xEM"
    distance = float(reason.get("distance"))
    assert distance <= expected_move * 0.40 + 1e-9
    assert abs(snapped[0] - entry) <= expected_move * 0.40 + 1e-6
    assert all(tag.upper() not in {"VAH", "SESSION_HIGH", "PWH"} for tag in snap_tags)
