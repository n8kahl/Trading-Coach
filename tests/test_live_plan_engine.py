import asyncio

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


@pytest.mark.asyncio
async def test_live_plan_engine_follower_and_replan():
    triggered: list[tuple[str, str | None, str, str | None]] = []

    async def fake_replan(symbol: str, style: str | None, origin: str, reason: str | None) -> None:
        triggered.append((symbol, style, origin, reason))

    engine = LivePlanEngine()
    engine.set_replan_callback(fake_replan)

    snapshot = {
        "plan": {
            "plan_id": "tsla-plan",
            "version": 1,
            "symbol": "TSLA",
            "bias": "long",
            "style": "intraday",
            "entry": 100.0,
            "stop": 97.5,
            "targets": [104.0],
            "rr_to_t1": 2.0,
            "confidence": 0.7,
            "atr": 1.5,
        }
    }

    await engine.register_snapshot(snapshot)

    events = await engine.handle_market_event("TSLA", {"t": "tick", "p": 100.2})
    assert any(evt["changes"].get("status") == "entered" for evt in events)

    events = await engine.handle_market_event("TSLA", {"t": "tick", "p": 104.2})
    assert any(evt["changes"].get("status") == "scaled" for evt in events)

    # Move below trail
    exit_events = await engine.handle_market_event("TSLA", {"t": "tick", "p": 102.5})
    assert any(evt["changes"].get("status") == "exited" for evt in exit_events)

    await asyncio.sleep(0)
    assert triggered, "expected auto replan callback"
    symbol, style, origin, reason = triggered[0]
    assert symbol == "TSLA"
    assert style == "intraday"
    assert origin == "tsla-plan"
