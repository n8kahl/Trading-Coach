from src.app.services.chart_layers import _build_forecast_path_generic


def _mkplan(entry=100.5, targets=None):
    return {"entry": entry, "targets": targets or [101.3]}


def test_orb_like_without_vwap_builds_path():
    anno = _build_forecast_path_generic(
        "range_break_retest",
        "long",
        "acceptance over ORH",
        {"ORH": 100.2},
        _mkplan(),
        last_time_s=1_700_000_000,
        interval_s=300,
        last_close=100.0,
    )
    assert anno and len(anno["points"]) >= 3


def test_vwap_reclaim_uses_vwap_when_present():
    anno = _build_forecast_path_generic(
        "vwap_reclaim",
        "long",
        "5m close above VWAP + acceptance over ORH",
        {"VWAP": 99.9, "ORH": 100.2},
        _mkplan(),
        last_time_s=1_700_000_000,
        interval_s=300,
        last_close=99.7,
    )
    values = {round(point["value"], 4) for point in anno["points"]}
    assert 99.9 in values and 100.2 in values


def test_baseline_fallback_when_unknown_strategy():
    anno = _build_forecast_path_generic(
        "unknown_strategy",
        "short",
        "close below Pivot",
        {"Pivot": 99.0},
        _mkplan(entry=98.6, targets=[97.8]),
        last_time_s=1_700_000_000,
        interval_s=60,
        last_close=99.5,
    )
    assert anno and len(anno["points"]) >= 3
