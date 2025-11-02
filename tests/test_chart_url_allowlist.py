from __future__ import annotations

from urllib.parse import parse_qs, urlsplit

import pytest
from httpx import ASGITransport, AsyncClient

from src.app.services.chart_url import ALLOWED_KEYS, make_chart_url
from src.agent_server import app
from src.config import get_settings


def test_make_chart_url_enforces_allowlist(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FF_CHART_CANONICAL_V1", "1")
    get_settings.cache_clear()

    params = {
        "symbol": "spy",
        "interval": "5m",
        "direction": "long",
        "entry": 430.10,
        "stop": 428.80,
        "tp": [432.1, 433.55],
        "focus": "plan",
        "center_time": "latest",
        "ui_state": {"style": "intraday"},
        "range": "5D",
        "view": "6M",
        "scale_plan": "auto",
        "unlisted": "should_not_be_included",
    }

    url = make_chart_url(params, base_url="https://example.com/chart")
    parsed = urlsplit(url)
    query = parse_qs(parsed.query)

    for key in query.keys():
        assert key in ALLOWED_KEYS

    assert "unlisted" not in query
    assert query["symbol"] == ["SPY"]
    assert query["entry"] == ["430.1"]
    assert query["stop"] == ["428.8"]
    assert query["tp"] == ["432.1,433.55"]

    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_chart_url_endpoint_strips_unknown_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FF_CHART_CANONICAL_V1", "1")
    get_settings.cache_clear()

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="https://test.local") as client:
        response = await client.post(
            "/gpt/chart-url",
            json={
                "symbol": "SPY",
                "interval": "5m",
                "direction": "long",
                "entry": 430.1,
                "stop": 428.8,
                "tp": "432.1,433.55",
                "focus": "plan",
                "center_time": "latest",
                "hijack": "1",
            },
        )

    get_settings.cache_clear()

    assert response.status_code == 422
    payload = response.json()
    assert any(
        isinstance(item, dict) and item.get("loc", [None])[-1] == "hijack"
        for item in payload.get("detail", [])
    )
