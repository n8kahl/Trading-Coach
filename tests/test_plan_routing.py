from __future__ import annotations

from datetime import datetime, timezone

import pytest
from httpx import ASGITransport, AsyncClient

import src.agent_server as agent_server
from src.agent_server import app
from src.config import get_settings
from src.lib import data_source as data_source_module
from src.lib.data_source import DataRoute
from src.lib.market_clock import most_recent_regular_close
import src.services.fallbacks as plan_fallbacks


@pytest.mark.asyncio()
async def test_plan_route_closed_uses_lkg(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "gpt_market_routing_enabled", True, raising=False)
    sunday = datetime(2024, 6, 9, 14, 0, 0, tzinfo=timezone.utc)  # Sunday

    def fake_pick() -> DataRoute:
        return data_source_module.pick_data_source(now=sunday)

    monkeypatch.setattr(agent_server, "pick_data_source", fake_pick)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/gpt/plan", json={"symbol": "NVDA"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["planning_context"] == "frozen"
    snapshot = payload["data_quality"]["snapshot"]
    assert snapshot["generated_at"] == most_recent_regular_close(sunday).isoformat()
    assert snapshot["symbol_count"] == 1
    assert payload["data_quality"]["expected_move"] is not None
    assert payload["data_quality"]["remaining_atr"] is not None
    assert "snap_trace" in payload and payload["snap_trace"]
    assert payload["key_levels_used"]


@pytest.mark.asyncio()
async def test_plan_route_open_uses_live(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "gpt_market_routing_enabled", True, raising=False)
    tuesday = datetime(2024, 6, 11, 14, 15, 0, tzinfo=timezone.utc)  # Tuesday 10:15 ET

    def fake_pick() -> DataRoute:
        return data_source_module.pick_data_source(now=tuesday)

    monkeypatch.setattr(agent_server, "pick_data_source", fake_pick)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/gpt/plan", json={"symbol": "AAPL"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["planning_context"] == "live"
    assert payload["data_quality"]["snapshot"]["generated_at"] == tuesday.isoformat()
    assert payload["warnings"] == []


@pytest.mark.asyncio()
async def test_plan_no_coupling_200(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "gpt_market_routing_enabled", True, raising=False)
    friday_close = datetime(2024, 6, 7, 20, 0, 0, tzinfo=timezone.utc)  # Friday 16:00 ET
    route = DataRoute(mode="lkg", as_of=friday_close, planning_context="frozen")
    monkeypatch.setattr(
        agent_server,
        "_SCAN_SYMBOL_REGISTRY",
        {("anonymous", "session-token", "intraday"): ["AAPL"]},
        raising=False,
    )

    def fake_pick() -> DataRoute:
        return route

    async def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("planner offline")

    monkeypatch.setattr(agent_server, "pick_data_source", fake_pick)
    monkeypatch.setattr(plan_fallbacks, "run_plan", boom)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/gpt/plan", json={"symbol": "TSLA"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["planning_context"] == "frozen"
    assert payload["plan_id"].endswith("STUB-LKG")
    assert any("fallback" in trace for trace in payload.get("snap_trace", []))
    assert "LKG_PARTIAL" in payload["warnings"]
    assert payload["data_quality"]["snapshot"]["generated_at"] == friday_close.isoformat()
    assert payload["data_quality"]["snapshot"]["symbol_count"] == 1
    assert payload["key_levels_used"]
