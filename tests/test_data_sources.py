from src.data_sources import _parse_polygon_timeframe


def test_parse_polygon_timeframe_minutes():
    multiplier, timespan, default_days = _parse_polygon_timeframe("15")
    assert multiplier == 15
    assert timespan == "minute"
    assert default_days >= 5


def test_parse_polygon_timeframe_hours():
    multiplier, timespan, default_days = _parse_polygon_timeframe("4h")
    assert multiplier == 4
    assert timespan == "hour"
    assert default_days >= 5


def test_parse_polygon_timeframe_daily():
    multiplier, timespan, default_days = _parse_polygon_timeframe("1D")
    assert multiplier == 1
    assert timespan == "day"
    assert default_days == 250
