from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient
import pytest

import src.agent_server as agent_server
from src.agent_server import app
from src.config import get_settings
from _helpers import stub_plan_components


def test_coach_websocket_emits_pulse(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FF_LAYERS_ENDPOINT", "1")
    monkeypatch.setenv("FF_CHART_CANONICAL_V1", "1")
    monkeypatch.setenv("FF_OPTIONS_ALWAYS", "1")
    monkeypatch.setenv("GPT_BACKEND_V2_ENABLED", "1")
    monkeypatch.setenv("GPT_MARKET_ROUTING_ENABLED", "0")
    monkeypatch.setenv("BACKEND_API_KEY", "supersecret")
    get_settings.cache_clear()

    async def fake_generate_plan_v2(
        symbol: str,
        style: str | None,
        route,  # noqa: ANN001
        app,  # noqa: ANN001
    ) -> dict:
        plan_core, plan_layers, session_state = stub_plan_components(symbol.upper())
        return {
            "plan_id": plan_core["plan_id"],
            "version": 1,
            "planning_context": "live",
            "symbol": symbol.upper(),
            "style": style or "intraday",
            "bias": plan_core["direction"],
            "entry": plan_core["entry"],
            "stop": plan_core["stop"],
            "targets": plan_core["targets"],
            "plan": plan_core,
            "plan_layers": plan_layers,
            "session_state": session_state,
            "charts": {},
            "charts_params": {},
        }

    monkeypatch.setattr("src.agent_server.generate_plan_v2", fake_generate_plan_v2)

    with TestClient(app) as client:
        plan_response = client.post(
            "/gpt/plan",
            json={"symbol": "SPY"},
            headers={"Authorization": "Bearer supersecret"},
        )
        assert plan_response.status_code == 200
        plan_payload = plan_response.json()
        plan_id = plan_payload["plan_id"]
        snapshot_plan = dict(plan_payload.get("plan") or {})
        snapshot_plan["plan_id"] = plan_id
        snapshot_plan["version"] = plan_payload.get("version") or 1
        if plan_payload.get("session_state"):
            snapshot_plan.setdefault("session_state", plan_payload.get("session_state"))
        snapshot_plan["plan_layers"] = plan_payload["plan_layers"]

        loop = asyncio.new_event_loop()
        loop.run_until_complete(
            agent_server._store_idea_snapshot(
                plan_id,
                {
                    "plan": snapshot_plan,
                    "plan_layers": plan_payload["plan_layers"],
                },
            )
        )
        loop.close()

        with client.websocket_connect(
            f"/ws/coach/{plan_id}",
            headers={"Authorization": "Bearer supersecret"},
        ) as ws:
            message = ws.receive_json()

    agent_server._IDEA_STORE.clear()
    get_settings.cache_clear()

    assert message["t"] == "coach_pulse"
    assert message["plan_id"] == plan_id
    diff = message["diff"]
    assert diff["waiting_for"]
    progress_block = diff["objective_progress"]
    assert isinstance(progress_block["progress"], float)
    assert 0.0 <= progress_block["progress"] <= 1.0
    assert progress_block["entry_distance_pct"] is None or progress_block["entry_distance_pct"] >= 0.0
    session_block = message["session"]
    assert session_block["status"]
    assert session_block["tz"]
