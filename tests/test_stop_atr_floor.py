from src.plans.snap import stop_from_structure


def test_stop_respects_atr_floor_long():
    levels = {"orl": 99.6}
    stop, label = stop_from_structure(entry=100.0, direction="long", levels=levels, atr_value=2.0, style="intraday")
    # ATR floor: entry - 0.9 * 2.0 = 98.2, rounded to cents
    assert stop <= 98.2
    assert label in {"ORL", "ATR_FALLBACK"}
    assert stop < 99.6


def test_stop_respects_atr_floor_short():
    levels = {"orh": 101.2}
    stop, label = stop_from_structure(entry=100.0, direction="short", levels=levels, atr_value=1.5, style="intraday")
    floor = 100.0 + 0.9 * 1.5
    assert stop >= floor - 1e-6
    assert label in {"ORH", "ATR_FALLBACK"}
