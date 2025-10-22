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

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/gpt/scan",
            json=_scan_payload(simulate_open=True),
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["planning_context"] == "live"
    assert len(payload["candidates"]) > 0
    assert payload["data_quality"]["expected_move"] == pytest.approx(1.5)


@pytest.mark.asyncio()
async def test_scan_closed_hours_non_empty_list(monkeypatch: pytest.MonkeyPatch) -> None:
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
    assert len(payload["candidates"]) == 2
    assert payload["warnings"].count("PLACEHOLDER") == 1
    candidate = payload["candidates"][0]
    assert candidate["score"] == pytest.approx(0.1)
    assert candidate["snap_trace"] == ["fallback: placeholder"]
    assert candidate["actionable_soon"] is False


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

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/gpt/scan", json=_scan_payload())

    assert response.status_code == 200
    payload = response.json()
    assert payload["banner"] == "MARKET_HOLIDAY"
    assert payload["candidates"] == []
    assert "PLACEHOLDER" not in payload.get("warnings", [])
