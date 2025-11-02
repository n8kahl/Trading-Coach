from __future__ import annotations

from copy import deepcopy

import pytest
from httpx import ASGITransport, AsyncClient

import src.agent_server as agent_server
from src.agent_server import app
from src.config import get_settings
from _helpers import stub_plan_components


@pytest.mark.asyncio
async def test_chart_layers_endpoint_injects_next_objective(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FF_LAYERS_ENDPOINT", "1")
    get_settings.cache_clear()

    plan_id = "SPY-demo-api"
    plan_core, plan_layers, _ = stub_plan_components("SPY")
    plan_core = deepcopy(plan_core)
    plan_layers = deepcopy(plan_layers)
    plan_core["plan_id"] = plan_id
    plan_layers["plan_id"] = plan_id
    plan_core["plan_layers"] = plan_layers

    # Remove persisted objective meta to confirm API recomputes it.
    meta_block = dict(plan_layers.get("meta") or {})
    meta_block.pop("next_objective", None)
    meta_block.pop("_next_objective_internal", None)
    plan_layers["meta"] = meta_block

    agent_server._IDEA_STORE.clear()
    await agent_server._store_idea_snapshot(
        plan_id,
        {
            "plan": plan_core,
            "plan_layers": plan_layers,
        },
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        response = await client.get(
            "/api/v1/gpt/chart-layers",
            params={"plan_id": plan_id},
        )

    agent_server._IDEA_STORE.clear()
    get_settings.cache_clear()

    assert response.status_code == 200
    payload = response.json()
    meta_block = payload["meta"]
    assert "next_objective" in meta_block
    next_objective = meta_block["next_objective"]
    assert next_objective["state"] in {"arming", "ready", "cooldown", "invalid"}
    assert isinstance(next_objective["objective_price"], (int, float))
    assert isinstance(next_objective["progress"], float)
    assert 0.0 <= next_objective["progress"] <= 1.0
    band = next_objective["band"]
    assert isinstance(band["low"], (int, float))
    assert isinstance(band["high"], (int, float))
    assert band["low"] < band["high"]
    assert isinstance(next_objective["why"], list)
    assert next_objective["timeframe"]
