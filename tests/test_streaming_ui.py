import pytest

from src.agent_server import _LIVE_PLAN_ENGINE, _ingest_stream_event


@pytest.mark.asyncio
async def test_streaming_ui_emits_coach_events(monkeypatch):
    captured: list[tuple[str, dict]] = []

    async def fake_publish(symbol: str, event: dict) -> None:
        captured.append((symbol, event))

    monkeypatch.setattr("src.agent_server._publish_stream_event", fake_publish)

    plan_snapshot = {
        "plan": {
            "plan_id": "stream-plan-1",
            "version": 1,
            "symbol": "TSLA",
            "bias": "long",
            "entry": 100.0,
            "stop": 98.0,
            "targets": [104.0],
            "rr_to_t1": 2.0,
            "atr": 1.0,
        }
    }

    await _LIVE_PLAN_ENGINE.register_snapshot(plan_snapshot)
    captured.clear()

    # Enter the trade
    await _ingest_stream_event("TSLA", {"t": "tick", "p": 100.2})
    # Approach the stop to trigger warning
    await _ingest_stream_event("TSLA", {"t": "tick", "p": 98.3})
    # Hit TP to trigger coaching cue
    await _ingest_stream_event("TSLA", {"t": "tick", "p": 104.2})

    coach_events = [
        event
        for _, event in captured
        if isinstance(event, dict)
        and event.get("t") == "plan_delta"
        and event.get("changes", {}).get("coach_event")
    ]

    assert any(evt["changes"]["coach_event"] == "stop_warning" for evt in coach_events), "expected stop warning coach event"
    assert any(evt["changes"]["coach_event"] == "tp_hit" for evt in coach_events), "expected tp hit coach event"
