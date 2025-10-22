import pytest

from tests.test_trade_detail import _run_fallback_plan


@pytest.mark.asyncio
async def test_response_propagates_actionability_flags(monkeypatch):
    response = await _run_fallback_plan(monkeypatch, user_id="consistency")

    assert response.entry_anchor
    assert response.entry_actionability is not None

    meta = response.meta or {}
    assert "actionable_now" in meta
    assert "actionable_soon" in meta
    assert "entry_anchor" in meta
    assert meta.get("entry_anchor") == response.entry_anchor

    structured = response.structured_plan
    assert structured["entry_anchor"] == response.entry_anchor
    assert structured["entry_actionability"] == response.entry_actionability
    assert structured["actionable_now"] == response.actionable_now
    assert structured["actionable_soon"] == response.actionable_soon

    target_profile = response.target_profile
    assert target_profile["entry_anchor"] == response.entry_anchor
    assert target_profile["entry_actionability"] == response.entry_actionability
    assert target_profile["actionable_now"] == response.actionable_now
    assert target_profile["actionable_soon"] == response.actionable_soon

    # ensure actionability gate present for meta consumers
    assert "actionability_gate" in meta
