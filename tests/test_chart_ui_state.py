from datetime import datetime, timezone

from src.services.chart_utils import (
    build_ui_state,
    infer_session_label,
    normalize_confidence,
    normalize_style_token,
)


def test_infer_session_label_classifies_day_segments():
    premkt = datetime(2024, 6, 10, 8, 0, tzinfo=timezone.utc)
    regular = datetime(2024, 6, 10, 14, 0, tzinfo=timezone.utc)
    after = datetime(2024, 6, 10, 22, 0, tzinfo=timezone.utc)

    assert infer_session_label(premkt) == "premkt"
    assert infer_session_label(regular) == "live"
    assert infer_session_label(after) == "after"


def test_normalize_style_and_confidence_and_build_ui_state():
    style = normalize_style_token("Intraday Momentum")
    assert style == "intraday"

    confidence = normalize_confidence(55)
    assert confidence == 0.55

    payload = build_ui_state(session="live", confidence=confidence, style=style)
    assert payload == '{"confidence":0.55,"session":"live","style":"intraday"}'
