import asyncio
import json

import pandas as pd
import pytest
from starlette.requests import Request

from urllib.parse import parse_qs, urlparse

from src import agent_server


@pytest.mark.asyncio
async def test_gpt_plan_includes_trade_detail(monkeypatch):
    async def fake_scan(universe, request, user):
        return [
            {
                "symbol": "TSLA",
                "style": "swing",
                "plan": {
                    "plan_id": "TSLA-SWING-20251010",
                    "entry": 254.0,
                    "stop": 247.8,
                    "targets": [266.0, 273.5],
                    "direction": "long",
                },
                "charts": {
                    "params": {
                        "symbol": "TSLA",
                        "interval": "1D",
                        "direction": "long",
                        "entry": "254.0",
                        "stop": "247.8",
                        "tp": "266.0,273.5",
                    }
                },
                "market_snapshot": {
                    "indicators": {"atr14": 6.2},
                    "volatility": {"expected_move_horizon": 12},
                    "trend": {"direction_hint": "long"},
                },
                "key_levels": {},
                "features": {},
            }
        ]

    async def fake_chart_url(params, request):
        return agent_server.ChartLinks(
            interactive="https://example.com/chart",
            png="https://example.com/chart.png",
        )

    monkeypatch.setattr(agent_server, "gpt_scan", fake_scan)
    monkeypatch.setattr(agent_server, "gpt_chart_url", fake_chart_url)

    scope = {"type": "http", "method": "POST", "path": "/", "headers": [], "query_string": b""}
    request = Request(scope)
    user = agent_server.AuthedUser(user_id="test-user")

    response = await agent_server.gpt_plan(agent_server.PlanRequest(symbol="TSLA", style="swing"), request, user)

    assert response.trade_detail, "trade_detail should be populated"
    assert response.idea_url == response.trade_detail, "legacy idea_url alias should mirror trade_detail"
    assert response.plan["trade_detail"] == response.trade_detail, "embedded plan must carry trade_detail"
    assert response.plan["idea_url"] == response.trade_detail, "embedded plan must keep idea_url alias"
    parsed = urlparse(response.trade_detail)
    params = parse_qs(parsed.query)
    assert params.get("plan_id") == ["TSLA-SWING-20251010"]
    assert params.get("plan_version") == [str(response.version)]


@pytest.mark.asyncio
async def test_simulate_generator_serializes_timestamp(monkeypatch):
    index = pd.date_range("2024-01-01", periods=1, freq="min", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": [1.0],
            "high": [1.1],
            "low": [0.9],
            "close": [1.05],
            "volume": [1000],
        },
        index=index,
    )
    frame["time"] = index

    monkeypatch.setattr(agent_server, "get_candles", lambda symbol, interval, lookback=30: frame)

    params = {"minutes": 5, "entry": 1.0, "stop": 0.9, "tp1": 1.1, "direction": "long"}
    generator = agent_server._simulate_generator("AAPL", params)
    chunk = await generator.__anext__()
    assert chunk.startswith("data: ")
    payload = json.loads(chunk[len("data: ") :].strip())
    assert payload["time"].endswith("+00:00")
    await generator.aclose()
