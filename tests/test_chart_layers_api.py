import pytest
from starlette.requests import Request

from src.agent_server import chart_layers_endpoint, _store_idea_snapshot
from src.config import get_settings


@pytest.mark.asyncio
async def test_chart_layers_endpoint_returns_layers(monkeypatch):
    monkeypatch.setenv("FF_CHART_CANONICAL_V1", "1")
    get_settings.cache_clear()

    plan_id = "TEST-PLAN-123"
    sample_layers = {
        "plan_id": plan_id,
        "symbol": "TSLA",
        "interval": "5m",
        "as_of": "2025-10-16T14:30:00Z",
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
                "plan_layers": sample_layers,
            },
        },
    )

    request = Request({"type": "http", "method": "GET", "path": "/api/v1/gpt/chart-layers", "headers": []})
    payload = await chart_layers_endpoint(plan_id=plan_id, request=request)

    assert payload["plan_id"] == plan_id
    assert payload["levels"][0]["label"] == "Prev Low"
    get_settings.cache_clear()
