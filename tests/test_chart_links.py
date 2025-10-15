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


@pytest.mark.asyncio
async def test_gpt_chart_url_carries_data_metadata():
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
        symbol="NVDA",
        interval="5m",
        direction="short",
        entry=177.0,
        stop=178.0,
        tp="175.50,175.10,174.80",
        last_update="2025-10-15T20:05:00Z",
        data_source="polygon_cached",
        data_mode="degraded",
        data_age_ms="964528",
    )

    links = await gpt_chart_url(params, request)

    assert "last_update=2025-10-15T20%3A05%3A00Z" in links.interactive
    assert "data_source=polygon_cached" in links.interactive
    assert "data_mode=degraded" in links.interactive
