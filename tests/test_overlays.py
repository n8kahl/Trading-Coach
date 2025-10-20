import pytest
from fastapi import HTTPException
from starlette.requests import Request

from src.agent_server import chart_layers_endpoint, _store_idea_snapshot
from src.config import get_settings


@pytest.mark.asyncio
async def test_overlays_endpoint_returns_layers(monkeypatch):
    monkeypatch.setenv("FF_LAYERS_ENDPOINT", "1")
    get_settings.cache_clear()

    plan_id = "PLAN-123"
    as_of = "2025-10-16T14:30:00Z"
    await _store_idea_snapshot(
        plan_id,
        {
            "plan": {
                "plan_id": plan_id,
                "session_state": {"as_of": as_of},
                "plan_layers": {
                    "plan_id": plan_id,
                    "as_of": as_of,
                    "levels": [
                        {"price": 420.5, "label": "Session High", "kind": "level"},
                    ],
                    "zones": [],
                    "annotations": [],
                },
            },
        },
    )

    request = Request({"type": "http", "method": "GET", "path": "/api/v1/gpt/chart-layers", "headers": []})
    payload = await chart_layers_endpoint(plan_id=plan_id, request=request)

    assert payload["plan_id"] == plan_id
    assert payload["levels"][0]["label"] == "Session High"
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_overlays_endpoint_raises_on_mismatch(monkeypatch):
    monkeypatch.setenv("FF_LAYERS_ENDPOINT", "1")
    get_settings.cache_clear()

    plan_id = "PLAN-MISMATCH"
    await _store_idea_snapshot(
        plan_id,
        {
            "plan": {
                "plan_id": plan_id,
                "session_state": {"as_of": "2025-10-16T14:30:00Z"},
                "plan_layers": {
                    "plan_id": plan_id,
                    "as_of": "2025-10-15T20:00:00Z",
                    "levels": [],
                    "zones": [],
                    "annotations": [],
                },
            },
        },
    )

    request = Request({"type": "http", "method": "GET", "path": "/api/v1/gpt/chart-layers", "headers": []})
    with pytest.raises(HTTPException) as exc:
        await chart_layers_endpoint(plan_id=plan_id, request=request)

    assert exc.value.status_code == 409
    detail = exc.value.detail
    assert detail["plan_id"] == plan_id
    assert "stale" in detail["message"].lower()
    get_settings.cache_clear()
