from src.app.services.chart_url import make_chart_url


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
