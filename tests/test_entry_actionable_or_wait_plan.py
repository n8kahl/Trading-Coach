import math
from dataclasses import replace

import pytest

from src import agent_server
from tests.test_trade_detail import _run_fallback_plan


@pytest.mark.asyncio
async def test_wait_plan_when_below_actionability(monkeypatch):
    original_select = agent_server.select_best_entry_plan

    def low_actionability_select(ctx, plan_kwargs, *, builder, min_actionability=None):
        geometry, candidate = original_select(ctx, plan_kwargs, builder=builder, min_actionability=min_actionability)
        low_candidate = replace(candidate, actionability=0.1, actionable_soon=False)
        return geometry, low_candidate

    monkeypatch.setattr(agent_server, "select_best_entry_plan", low_actionability_select)

    response = await _run_fallback_plan(
        monkeypatch,
        user_id="wait-case",
        min_actionability=0.95,
        must_be_actionable=True,
    )

    assert response.plan["plan_state"] == "WAIT"
    assert response.entry is None
    assert response.stop is None
    assert response.targets == []
    assert response.waiting_for

    meta = response.meta or {}
    assert meta.get("actionable_now") is False
    assert meta.get("actionable_soon") is False
    assert meta.get("waiting_for") == response.waiting_for

    structured_entry = response.structured_plan["entry"]
    assert structured_entry["type"] == "wait"
    assert structured_entry["trigger"] == response.waiting_for

    target_profile = response.target_profile
    assert target_profile["entry"] is None
    assert target_profile["stop"] is None
    assert target_profile["targets"] == []
    assert target_profile["actionable_now"] is False
    assert target_profile["actionable_soon"] is False


@pytest.mark.asyncio
async def test_passed_entry_reclaim_sets_waiting_for(monkeypatch):
    original_select = agent_server.select_best_entry_plan
    original_nearest = agent_server._nearest_retest_or_reclaim

    def reclaim_anchor(levels, last, direction):
        return {"label": "VWAP", "level": round(float(last) - 1.0, 2), "kind": "session"}

    def behind_tape_select(ctx, plan_kwargs, *, builder, min_actionability=None):
        geometry, candidate = original_select(ctx, plan_kwargs, builder=builder, min_actionability=min_actionability)
        reclaimed_entry = ctx.last_price - max(ctx.atr * 0.6, 0.5)
        adjusted_kwargs = dict(plan_kwargs)
        adjusted_kwargs["entry"] = reclaimed_entry
        adjusted_geometry = builder(**adjusted_kwargs)
        tweaked_candidate = agent_server.EntryCandidate(
            entry=round(adjusted_geometry.entry, 4),
            stop=round(adjusted_geometry.stop.price, 4),
            tag="structural",
            actionability=0.93,
            actionable_soon=False,
            entry_distance_pct=abs(reclaimed_entry - ctx.last_price) / ctx.last_price if ctx.last_price else math.inf,
            entry_distance_atr=abs(reclaimed_entry - ctx.last_price) / ctx.atr if ctx.atr > 0 else math.inf,
            bars_to_trigger=2,
        )
        return adjusted_geometry, tweaked_candidate

    monkeypatch.setattr(agent_server, "select_best_entry_plan", behind_tape_select)
    monkeypatch.setattr(agent_server, "_nearest_retest_or_reclaim", reclaim_anchor)

    response = await _run_fallback_plan(monkeypatch, user_id="reclaim-case")

    assert response.waiting_for
    assert "Reclaim VWAP" in response.waiting_for
    assert response.entry_anchor == "vwap"
    assert response.entry_actionability >= 0.9
    assert response.entry is not None
    assert response.plan["entry"] == response.entry
    assert response.plan.get("waiting_for") == response.waiting_for
    assert response.structured_plan["waiting_for"] == response.waiting_for
    assert response.target_profile["waiting_for"] == response.waiting_for
    assert response.target_profile["entry_anchor"] == "vwap"
    assert response.structured_plan["entry"]["type"] != "wait"
    meta = response.meta or {}
    assert meta.get("waiting_for") == response.waiting_for
