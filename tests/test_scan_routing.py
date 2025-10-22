from __future__ import annotations

from datetime import datetime, timezone

import pytest
from httpx import ASGITransport, AsyncClient

import src.agent_server as agent_server
from src.agent_server import app
from src.config import get_settings
from src.lib.data_source import DataRoute
import src.lib.data_source as data_source_module
import src.services.scan_fallbacks as scan_fallbacks
import src.universe as universe_module


def _scan_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "universe": ["AAPL", "MSFT"],
        "style": "intraday",
        "limit": 5,
    }
    payload.update(overrides)
    return payload


@pytest.mark.asyncio()
async def test_scan_tomorrow_sim_open_returns_live(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "gpt_market_routing_enabled", True, raising=False)
    sunday = datetime(2024, 6, 9, 14, 0, 0, tzinfo=timezone.utc)

    def fake_pick() -> DataRoute:
        return data_source_module.pick_data_source(now=sunday)

    monkeypatch.setattr(agent_server, "pick_data_source", fake_pick)

    async def fake_load_universe(*args, **kwargs) -> list[str]:
        return ["AAPL", "MSFT", "NVDA"]

    universe_module._UNIVERSE_CACHE.clear()
    monkeypatch.setattr(universe_module, "load_universe", fake_load_universe)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/gpt/scan",
            json=_scan_payload(universe="LAST_SNAPSHOT", simulate_open=True),
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["planning_context"] == "live"
    assert payload["meta"]["route"] == "live"
    assert len(payload["candidates"]) > 0
    assert payload["candidates"][0]["symbol"] == "AAPL"
    assert payload["data_quality"]["expected_move"] == pytest.approx(1.5)


@pytest.mark.asyncio()
async def test_scan_closed_hours_allows_empty_without_banner(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "gpt_market_routing_enabled", True, raising=False)
    saturday = datetime(2024, 6, 8, 16, 0, 0, tzinfo=timezone.utc)

    def fake_pick() -> DataRoute:
        return data_source_module.pick_data_source(now=saturday)

    async def empty_run_scan(*args: object, **kwargs: object) -> dict[str, object]:
        return {
            "candidates": [],
            "meta": {},
            "data_quality": {},
        }

    monkeypatch.setattr(agent_server, "pick_data_source", fake_pick)
    monkeypatch.setattr(scan_fallbacks, "run_scan", empty_run_scan)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/gpt/scan", json=_scan_payload())

    assert response.status_code == 200
    payload = response.json()
    assert payload["planning_context"] == "frozen"
    assert payload["candidates"] == []
    assert payload["count_candidates"] == 0
    assert "PLACEHOLDER" not in payload.get("warnings", [])


@pytest.mark.asyncio()
async def test_banner_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = get_settings()
    monkeypatch.setattr(settings, "gpt_market_routing_enabled", True, raising=False)
    friday = datetime(2024, 6, 7, 20, 0, 0, tzinfo=timezone.utc)

    def fake_pick() -> DataRoute:
        return data_source_module.pick_data_source(now=friday)

    async def banner_scan(*args: object, **kwargs: object) -> dict[str, object]:
        return {
            "banner": "MARKET_HOLIDAY",
            "candidates": [],
            "meta": {},
            "data_quality": {},
        }

    monkeypatch.setattr(agent_server, "pick_data_source", fake_pick)
    monkeypatch.setattr(scan_fallbacks, "run_scan", banner_scan)

    async def empty_expand(*args, **kwargs) -> list[str]:
        return []

    monkeypatch.setattr(agent_server, "expand_universe", empty_expand)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/gpt/scan", json=_scan_payload(universe="LAST_SNAPSHOT"))

    assert response.status_code == 200
    payload = response.json()
    assert payload["banner"] == "MARKET_HOLIDAY"
    assert payload["candidates"]
    assert payload["candidates"][0]["symbol"] == "ADHOC"
    assert "PLACEHOLDER" in payload.get("warnings", [])
