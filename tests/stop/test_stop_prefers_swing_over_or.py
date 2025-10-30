import pytest

from src.plans.snap import stop_from_structure


def test_stop_prefers_recent_swing_when_within_em_band():
    entry = 100.0
    levels = {
        "orl": 92.5,
        "swing_low": 97.2,
    }
    expected_move = 10.0

    stop_price, label, meta = stop_from_structure(
        entry=entry,
        direction="long",
        levels=levels,
        atr_value=1.4,
        style="intraday",
        expected_move=expected_move,
    )

    assert label == "SWING_LOW"
    assert meta["anchor"] == "swing"
    assert meta["final"] == stop_price
    assert pytest.approx(meta["structural"], rel=0, abs=1e-6) == levels["swing_low"] - meta["wick_buffer"]
    distance = entry - stop_price
    assert expected_move * 0.30 <= distance <= expected_move * 0.60 + 1e-9
    # Ensure OR anchor would have been further away and was not chosen
    assert stop_price > levels["orl"]  # closer to entry than ORL - wick path
