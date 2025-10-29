import pytest
from starlette.requests import Request

from src.agent_server import _store_idea_snapshot, chart_layers_endpoint
from src.config import get_settings


@pytest.mark.asyncio
async def test_plan_adoption_layers_expose_annotations(monkeypatch):
    monkeypatch.setenv("FF_LAYERS_ENDPOINT", "1")
    get_settings.cache_clear()

    plan_id = "PLAN-ADOPT-001"
    as_of = "2025-02-10T14:30:00Z"

    await _store_idea_snapshot(
        plan_id,
        {
            "plan": {
                "plan_id": plan_id,
                "session_state": {"as_of": as_of, "status": "closed", "tz": "America/New_York"},
                "plan_layers": {
                    "plan_id": plan_id,
                    "as_of": as_of,
                    "levels": [
                        {"price": 412.5, "label": "Session High"},
                        {"price": 405.2, "label": "Session Low"},
                    ],
                    "zones": [
                        {"high": 409.0, "low": 406.5, "label": "Demand", "kind": "demand"}
                    ],
                    "annotations": [
                        {
                            "kind": "forecast_path",
                            "style": {"lineStyle": "dotted", "opacity": 0.6},
                            "points": [
                                {"time": 1739190600, "value": 407.2},
                                {"time": 1739192400, "value": 410.8},
                            ],
                        }
                    ],
                    "meta": {"tz": "America/New_York", "confidence": 0.62},
                },
            }
        },
    )

    request = Request({"type": "http", "method": "GET", "path": "/api/v1/gpt/chart-layers", "headers": []})
    payload = await chart_layers_endpoint(plan_id=plan_id, request=request)

    assert payload["plan_id"] == plan_id
    assert payload["levels"], "expected persisted levels for adoption"
    assert payload["annotations"], "expected annotations to drive overlays"
    assert payload["meta"].get("tz") == "America/New_York"

    get_settings.cache_clear()
