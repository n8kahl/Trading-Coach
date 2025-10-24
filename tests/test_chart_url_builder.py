from urllib.parse import parse_qs, urlsplit

from src.app.services.chart_url import make_chart_url
from src.app.services.instrument_precision import get_precision


def test_make_chart_url_orders_params_and_formats_precision():
    params = {
        "symbol": "spy",
        "interval": "5m",
        "direction": "long",
        "entry": 430.1234,
        "stop": 428.9876,
        "tp": [432.55, 435.0],
        "ema": [9, 21],
        "focus": "plan",
        "center_time": "latest",
        "scale_plan": "auto",
        "theme": "dark",
        "plan_id": "SPY-PLAN",
        "plan_version": 2,
    }

    url = make_chart_url(params, base_url="https://example.com/tv", precision_map={"SPY": 2})

    assert (
        url
        == "https://example.com/tv?center_time=latest&direction=long&ema=9,21&entry=430.12&focus=plan&interval=5m&plan_id=SPY-PLAN&plan_version=2&scale_plan=auto&stop=428.99&symbol=SPY&theme=dark&tp=432.55,435"
    )


def test_make_chart_url_ignores_non_allowlisted_keys():
    url = make_chart_url(
        {
            "symbol": "NDX",
            "interval": "15m",
            "direction": "short",
            "entry": 16050.25,
            "stop": 16120.75,
            "tp": [15890.5],
            "session_status": "open",
            "session_as_of": "2024-01-01T21:00:00Z",
        },
        base_url="https://example.com/tv",
        precision_map={"NDX": 2},
    )

    assert "session_status" not in url
    assert "session_as_of" not in url
    assert url.endswith("entry=16050.25&interval=15m&stop=16120.75&symbol=NDX&tp=15890.5")


def test_get_precision_resolves_aliases():
    mapping = {"SPX": 1, "DEFAULT": 3}

    assert get_precision("I:SPX", precision_map=mapping) == 1
    assert get_precision("spx", precision_map=mapping) == 1
    assert get_precision("UNKNOWN", precision_map=mapping) == 3


def test_make_chart_url_includes_ui_state_token():
    params = {
        "symbol": "AAPL",
        "interval": "5m",
        "direction": "long",
        "entry": 195.12,
        "stop": 193.5,
        "tp": [198.4],
        "ui_state": '{"session":"live","confidence":0.82,"style":"intraday"}',
    }

    url = make_chart_url(params, base_url="https://example.com/tv", precision_map={"AAPL": 2})

    parsed = urlsplit(url)
    query = parse_qs(parsed.query)
    assert query["ui_state"] == ['{"session":"live","confidence":0.82,"style":"intraday"}']
