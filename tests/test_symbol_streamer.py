from datetime import datetime, timezone

from src.symbol_streamer import determine_market_phase


def test_determine_market_phase_regular_hours():
    dt = datetime(2024, 7, 15, 15, 0, tzinfo=timezone.utc)  # 11:00 ET on Monday
    assert determine_market_phase(dt) == "regular"


def test_determine_market_phase_after_hours():
    dt = datetime(2024, 7, 16, 22, 0, tzinfo=timezone.utc)  # 18:00 ET on Tuesday
    assert determine_market_phase(dt) == "afterhours"


def test_determine_market_phase_weekend():
    dt = datetime(2024, 7, 14, 14, 0, tzinfo=timezone.utc)  # Sunday
    assert determine_market_phase(dt) == "closed"
