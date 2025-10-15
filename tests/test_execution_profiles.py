import math

from src.app.engine.execution_profiles import ExecutionContext, refine_plan
from src.scanner import Plan


def test_refine_scalp_prefers_reclaim_entry():
    original_plan = Plan(
        direction="long",
        entry=670.23,
        stop=669.54,
        targets=[671.40, 672.10, 672.85],
        confidence=0.62,
        risk_reward=1.20,
        notes="Baseline plan",
        atr=0.82,
    )
    ctx = ExecutionContext(
        symbol="NVDA",
        style="scalp",
        direction="long",
        price=670.05,
        key_levels={
            "opening_range_high": 669.80,
            "session_high": 670.05,
            "prev_high": 670.30,
        },
        atr14=0.82,
        expected_move=1.65,
        vwap=669.92,
        ema_stack="bullish",
        session_phase="morning",
        minutes_to_close=360,
        data_mode=None,
    )
    entry_before = original_plan.entry
    rr_before = original_plan.risk_reward

    refined_plan, adjustment = refine_plan(original_plan, ctx)

    assert refined_plan.entry < entry_before  # moved closer to reclaim level
    assert math.isclose(refined_plan.stop, round(refined_plan.stop, 4))
    assert refined_plan.entry - refined_plan.stop >= 0.2  # respect minimum stop distance
    assert refined_plan.risk_reward >= rr_before
    assert adjustment.entry_type == "reclaim"
    assert "Sniper entry" in refined_plan.notes


def test_refine_intraday_clamps_stop():
    plan = Plan(
        direction="short",
        entry=420.5,
        stop=424.5,  # overly wide stop
        targets=[415.0, 410.0],
        confidence=0.55,
        risk_reward=0.9,
        notes=None,
        atr=2.2,
    )
    ctx = ExecutionContext(
        symbol="SPY",
        style="intraday",
        direction="short",
        price=421.0,
        key_levels={"opening_range_low": 422.0},
        atr14=2.2,
        expected_move=3.0,
        vwap=421.5,
        ema_stack="bearish",
        session_phase="midday",
        minutes_to_close=200,
        data_mode=None,
    )

    refined_plan, _ = refine_plan(plan, ctx)

    assert refined_plan.stop < 424.5  # tightened stop
    assert refined_plan.stop > refined_plan.entry  # still above entry because short
    assert refined_plan.risk_reward >= plan.risk_reward
