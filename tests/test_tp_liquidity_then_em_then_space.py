from datetime import datetime, timezone

from src.plans.pipeline import build_structured_geometry


def test_targets_snap_then_clamp_then_respace():
    entry = 100.0
    levels = {
        "vah": 100.8,
        "poc": 101.15,
        "orh": 101.4,
        "prev_high": 101.75,
        "session_high": 101.9,
    }
    geometry = build_structured_geometry(
        symbol="TEST",
        style="intraday",
        direction="long",
        entry=entry,
        levels=levels,
        atr_value=0.6,
        plan_time=datetime.now(timezone.utc),
        raw_targets=[100.7, 101.2, 101.8],
        rr_floor=1.4,
        em_hint=1.0,
    )

    targets = geometry.targets
    assert len(targets) >= 3
    # monotonic increasing with minimum spacing enforced
    gaps = [targets[i + 1] - targets[i] for i in range(len(targets) - 1)]
    assert all(gap > 0 for gap in gaps)
    assert all(gap >= 0.15 for gap in gaps)
    # respect expected-move clamp
    assert all(target <= entry + 1.0 + 1e-6 for target in targets)
    # tp reasons retained per target
    assert len(geometry.tp_reasons) >= len(targets)
