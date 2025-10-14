from app.engine.events import EventDecision, apply_event_gating


def test_apply_event_gating_allows_when_no_events():
    decision = apply_event_gating(None)
    assert decision.action == "allow"
    assert decision.triggered == []


def test_apply_event_gating_defined_risk():
    events = [
        {"label": "FOMC", "minutes_to_event": 45, "severity": "high"},
        {"label": "Earnings", "minutes_to_event": 200, "severity": "medium"},
    ]
    decision = apply_event_gating(events, minutes_threshold=90, severity_threshold="medium")
    assert decision.action == "defined_risk"
    assert len(decision.triggered) == 1
    assert "FOMC" in decision.reason


def test_apply_event_gating_suppress_mode():
    events = [{"label": "Jobs Data", "minutes_to_event": 10, "severity": "critical"}]
    decision = apply_event_gating(events, mode="suppress", minutes_threshold=30)
    assert decision.action == "suppress"
    assert isinstance(decision.triggered, list)
    assert decision.reason is not None
