import pytest
from fastapi import HTTPException
from starlette.requests import Request

from src.agent_server import ChartParams, gpt_chart_url
from src.config import get_settings


@pytest.mark.asyncio
async def test_gpt_chart_url_returns_interactive_focus_params(monkeypatch):
    monkeypatch.setenv("FF_CHART_CANONICAL_V1", "1")
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://test.local")
    get_settings.cache_clear()
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

    assert (
        links.interactive
        == "https://test.local/tv?center_time=latest&direction=long&entry=430.1&focus=plan&interval=5m&stop=428.8&symbol=SPY&tp=432.1,433.55&view=6M"
    )
    assert not hasattr(links, "png")
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_gpt_chart_url_strips_non_canonical_params(monkeypatch):
    monkeypatch.setenv("FF_CHART_CANONICAL_V1", "1")
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://test.local")
    get_settings.cache_clear()
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

    assert (
        links.interactive
        == "https://test.local/tv?direction=short&entry=177&interval=5m&stop=178&symbol=NVDA&tp=175.5,175.1,174.8&view=6M"
    )
    assert "data_source" not in links.interactive
    assert "data_mode" not in links.interactive
    assert "last_update" not in links.interactive
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_gpt_chart_url_rejects_non_monotonic(monkeypatch):
    monkeypatch.setenv("FF_CHART_CANONICAL_V1", "1")
    get_settings.cache_clear()
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/gpt/chart-url",
        "headers": [(b"host", b"example.com")],
        "scheme": "https",
        "client": ("127.0.0.1", 0),
        "server": ("example.com", 443),
        "query_string": b"",
    }
    request = Request(scope)
    params = ChartParams(
        symbol="TSLA",
        interval="5m",
        direction="long",
        entry=250.0,
        stop=248.0,
        tp="251.0,250.5",
    )

    with pytest.raises(HTTPException) as exc:
        await gpt_chart_url(params, request)
    assert exc.value.status_code == 422
    assert "tp2" in exc.value.detail["error"]
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_gpt_chart_url_rejects_snapped_rr(monkeypatch):
    monkeypatch.setenv("FF_CHART_CANONICAL_V1", "1")
    get_settings.cache_clear()
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/gpt/chart-url",
        "headers": [(b"host", b"example.com")],
        "scheme": "https",
        "client": ("127.0.0.1", 0),
        "server": ("example.com", 443),
        "query_string": b"",
    }
    request = Request(scope)
    params = ChartParams(
        symbol="TSLA",
        interval="5m",
        direction="long",
        entry=250.0,
        stop=249.0,
        tp="251.2",
        levels="251.05|PDH",
    )
    with pytest.raises(HTTPException) as exc:
        await gpt_chart_url(params, request)
    assert exc.value.status_code == 422
    assert "R:R" in exc.value.detail["error"]
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_gpt_chart_url_rejects_snapped_monotonic(monkeypatch):
    monkeypatch.setenv("FF_CHART_CANONICAL_V1", "1")
    get_settings.cache_clear()
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/gpt/chart-url",
        "headers": [(b"host", b"example.com")],
        "scheme": "https",
        "client": ("127.0.0.1", 0),
        "server": ("example.com", 443),
        "query_string": b"",
    }
    request = Request(scope)
    params = ChartParams(
        symbol="AAPL",
        interval="5m",
        direction="long",
        entry=150.0,
        stop=149.5,
        tp="150.2,150.3",
        levels="150.30|VAH;150.22|VWAP",
    )
    with pytest.raises(HTTPException) as exc:
        await gpt_chart_url(params, request)
    assert exc.value.status_code == 422
    assert "R:R" in exc.value.detail["error"]
    get_settings.cache_clear()
