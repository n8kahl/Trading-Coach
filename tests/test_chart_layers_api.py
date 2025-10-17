import pytest
from starlette.requests import Request

from src.agent_server import chart_layers_endpoint, _store_idea_snapshot
from src.config import get_settings


@pytest.mark.asyncio
async def test_chart_layers_endpoint_returns_layers(monkeypatch):
    monkeypatch.setenv("FF_LAYERS_ENDPOINT", "1")
    get_settings.cache_clear()

    plan_id = "TEST-PLAN-123"
    as_of = "2025-10-16T14:30:00Z"
    sample_layers = {
        "plan_id": plan_id,
        "symbol": "TSLA",
        "interval": "5m",
        "as_of": as_of,
        "planning_context": "live",
        "precision": 2,
        "levels": [
            {"price": 417.86, "label": "Prev Low", "kind": "level"},
            {"price": 434.2, "label": "Prev High", "kind": "level"},
        ],
        "zones": [],
        "annotations": [],
    }

    await _store_idea_snapshot(
        plan_id,
        {
            "plan": {
                "plan_id": plan_id,
                "session_state": {"as_of": as_of},
                "plan_layers": sample_layers,
            },
        },
    )

    request = Request({"type": "http", "method": "GET", "path": "/api/v1/gpt/chart-layers", "headers": []})
    payload = await chart_layers_endpoint(plan_id=plan_id, request=request)

    assert payload["plan_id"] == plan_id
    assert payload["levels"][0]["label"] == "Prev Low"
    get_settings.cache_clear()


@pytest.mark.asyncio
async def test_chart_layers_endpoint_detects_asof_mismatch(monkeypatch):
    monkeypatch.setenv("FF_LAYERS_ENDPOINT", "1")
    get_settings.cache_clear()
    async def fake_rebuild(plan_id, snapshot, request):  # noqa: ARG001
        return None

    monkeypatch.setattr("src.agent_server._rebuild_plan_layers", fake_rebuild)

    plan_id = "TEST-PLAN-STALE"
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
    with pytest.raises(Exception) as excinfo:
        await chart_layers_endpoint(plan_id=plan_id, request=request)

    assert getattr(excinfo.value, "status_code", None) == 409
    get_settings.cache_clear()
