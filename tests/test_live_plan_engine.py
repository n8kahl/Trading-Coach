import pytest

from src.live_plan_engine import LivePlanEngine, PlanStatus


@pytest.mark.asyncio
async def test_live_plan_engine_status_transitions():
    engine = LivePlanEngine()
    snapshot = {
        "plan": {
            "plan_id": "abc123",
            "version": 1,
            "symbol": "TGT",
            "bias": "long",
            "entry": 100.0,
            "stop": 95.0,
            "targets": [110.0, 115.0],
            "rr_to_t1": 2.0,
            "confidence": 0.6,
        }
    }
    baseline_event = await engine.register_snapshot(snapshot)
    assert baseline_event is not None
    assert baseline_event["changes"]["status"] == PlanStatus.INTACT.value

    # Move price close to stop to trigger at_risk
    events = await engine.handle_market_event("TGT", {"t": "tick", "p": 95.2})
    assert events, "expected at least one event when approaching stop"
    at_risk = events[-1]
    assert at_risk["changes"]["status"] == PlanStatus.AT_RISK.value
    assert at_risk["changes"]["next_step"] == "tighten_stop"

    # Breach stop should invalidate
    events = await engine.handle_market_event("TGT", {"t": "tick", "p": 94.8})
    assert events, "expected event on stop breach"
    invalidated = events[-1]
    assert invalidated["changes"]["status"] == PlanStatus.INVALIDATED.value
    assert invalidated["changes"]["next_step"] == "plan_invalidated"
