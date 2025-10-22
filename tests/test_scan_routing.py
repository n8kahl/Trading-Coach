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
import src.services.scan_fallbacks as scan_fallbacks


def _scan_payload() -> dict[str, object]:
    return {
        "universe": ["AAPL", "MSFT"],
        "style": "intraday",
        "limit": 5,
    }


@pytest.mark.asyncio()
async def test_scan_route_closed_uses_lkg(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "gpt_market_routing_enabled", True, raising=False)
    sunday = datetime(2024, 6, 9, 14, 0, 0, tzinfo=timezone.utc)

    def fake_pick() -> DataRoute:
        return data_source_module.pick_data_source(now=sunday)

    monkeypatch.setattr(agent_server, "pick_data_source", fake_pick)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/gpt/scan", json=_scan_payload())

    assert response.status_code == 200
    payload = response.json()
    assert payload["planning_context"] == "frozen"
    snapshot = payload["meta"]["snapshot"]
    assert snapshot["generated_at"] == most_recent_regular_close(sunday).isoformat()
    assert snapshot["symbol_count"] == 2
    dq = payload["data_quality"]
    assert dq["snapshot"]["generated_at"] == snapshot["generated_at"]
    assert dq["expected_move"] is not None
    assert dq["remaining_atr"] is not None


@pytest.mark.asyncio()
async def test_scan_route_open_uses_live(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "gpt_market_routing_enabled", True, raising=False)
    tuesday = datetime(2024, 6, 11, 14, 15, 0, tzinfo=timezone.utc)

    def fake_pick() -> DataRoute:
        return data_source_module.pick_data_source(now=tuesday)

    monkeypatch.setattr(agent_server, "pick_data_source", fake_pick)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/gpt/scan", json=_scan_payload())

    assert response.status_code == 200
    payload = response.json()
    assert payload["planning_context"] == "live"
    assert payload["data_quality"]["snapshot"]["generated_at"] == tuesday.isoformat()
    assert payload["warnings"] == []


@pytest.mark.asyncio()
async def test_scan_route_emits_stub_when_series_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "gpt_market_routing_enabled", True, raising=False)
    friday_close = datetime(2024, 6, 7, 20, 0, 0, tzinfo=timezone.utc)
    route = DataRoute(mode="lkg", as_of=friday_close, planning_context="frozen")

    def fake_pick() -> DataRoute:
        return route

    async def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("series unavailable")

    monkeypatch.setattr(agent_server, "pick_data_source", fake_pick)
    monkeypatch.setattr(scan_fallbacks, "fetch_series", boom)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/gpt/scan", json=_scan_payload())

    assert response.status_code == 200
    payload = response.json()
    assert payload["planning_context"] == "frozen"
    assert payload["candidates"] == []
    assert "fallback" in payload.get("snap_trace", [])[0]
    assert "LKG_PARTIAL" in payload["warnings"]
    assert payload["data_quality"]["series_present"] is False
