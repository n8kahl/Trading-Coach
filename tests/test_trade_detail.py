import asyncio
from starlette.requests import Request

import pytest

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
        return agent_server.ChartLinks(interactive="https://example.com/chart")

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
