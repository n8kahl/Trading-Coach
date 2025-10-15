import pytest
from starlette.requests import Request

from src.agent_server import ChartParams, gpt_chart_url


@pytest.mark.asyncio
async def test_gpt_chart_url_returns_png_and_focus_params():
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/gpt/chart-url",
        "headers": [(b"host", b"test.local")],
        "scheme": "https",
        "client": ("127.0.0.1", 0),
        "server": ("test.local", 443),
        "query_string": b"",
    }
    request = Request(scope)
    params = ChartParams(
        symbol="SPY",
        interval="5m",
        direction="long",
        entry=430.10,
        stop=428.80,
        tp="432.10,433.55",
        focus="plan",
        center_time="latest",
    )

    links = await gpt_chart_url(params, request)

    assert links.interactive.startswith("https://test.local/charts/html?")
    assert "focus=plan" in links.interactive
    assert "center_time=latest" in links.interactive
    assert links.png is not None
    assert links.png.startswith("https://test.local/charts/png?")
    assert "focus=plan" in links.png
