from src.plans.snap import apply_atr_floor, compute_adaptive_wick_buffer


def test_apply_atr_floor_with_wick_buffer_short_long() -> None:
    entry = 100.0
    atr = 1.2

    # Short bias: ATR floor should push stop no closer than entry + k*ATR
    structural_short = 100.3
    stop_short = apply_atr_floor(entry, structural_short, atr, direction="short", style="intraday")
    expected_short = entry + 0.9 * atr
    assert stop_short == expected_short
    assert stop_short >= structural_short

    # Long bias: ATR floor should not allow stop tighter than entry - k*ATR
    structural_long = 99.5
    stop_long = apply_atr_floor(entry, structural_long, atr, direction="long", style="intraday")
    expected_long = entry - 0.9 * atr
    assert stop_long == expected_long
    assert stop_long <= structural_long

    buffer = compute_adaptive_wick_buffer(atr=2.0, tick=0.02)
    assert 0.05 <= buffer <= 0.35
