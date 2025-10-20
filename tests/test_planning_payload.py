from datetime import datetime, timezone

import pytest

from src.agent_server import _planning_scan_to_page
from src.planning.planning_scan import PlanningScanOutput
from src.schemas import ScanRequest
from src.services.contract_rules import ContractRuleBook
from src.services.scan_engine import PlanningCandidate
from src.services.universe import UniverseSnapshot


@pytest.mark.asyncio
async def test_planning_candidate_includes_actionability():
    template = ContractRuleBook().build("swing")
    candidate = PlanningCandidate(
        symbol="AAPL",
        readiness_score=0.88,
        components={"probability": 0.85, "actionability": 0.92, "risk_reward": 0.65},
        metrics={
            "entry_distance_pct": 0.5,
            "entry_distance_atr": 0.4,
            "runner_policy": {"fraction": 0.2, "atr_multiple": 1.0},
            "target_meta": [
                {
                    "price": 262.0,
                    "distance": 12.0,
                    "prob_touch": 0.62,
                    "rr_multiple": 2.4,
                    "snap_tag": "pdh",
                }
            ],
            "snap_trace": ["tp1:262.00 via pdh"],
            "expected_move": 8.5,
            "remaining_atr": 4.2,
            "em_used": False,
            "atr": 2.5,
        },
        levels={"entry": 250.0, "invalidation": 245.0, "targets": [262.0]},
        contract_template=template,
        requires_live_confirmation=True,
        missing_live_inputs=["iv", "spread", "oi"],
    )
    output = PlanningScanOutput(
        as_of_utc=datetime(2025, 1, 1, tzinfo=timezone.utc),
        universe=UniverseSnapshot(
            name="A+",
            source="adhoc",
            as_of_utc=datetime(2025, 1, 1, tzinfo=timezone.utc),
            symbols=["AAPL"],
            metadata={},
        ),
        run_id=None,
        indices_context={"I:SPX": {"close": 5000.0}},
        candidates=[candidate],
    )
    request = ScanRequest(universe="A+", style="swing", limit=20)

    page = _planning_scan_to_page(output, request)

    assert page.candidates[0].confidence == pytest.approx(0.88)
    assert page.candidates[0].actionable_soon is True
    assert any("Actionability" in reason for reason in page.candidates[0].reasons)
    assert page.candidates[0].entry_distance_atr == pytest.approx(0.4)
    assert page.candidates[0].entry_distance_pct == pytest.approx(0.5)
    assert page.candidates[0].planning_snapshot.get("runner_policy") is not None
    assert page.candidates[0].planning_snapshot["runner_policy"]["fraction"] == pytest.approx(0.2)
    candidate_payload = page.candidates[0]
    assert candidate_payload.plan_id
    assert candidate_payload.target_meta and candidate_payload.targets_meta
    assert candidate_payload.tp_reasons
    assert candidate_payload.runner_policy and candidate_payload.runner_policy["fraction"] == pytest.approx(0.2)
    assert candidate_payload.snap_trace and "pdh" in candidate_payload.snap_trace[0]
    assert candidate_payload.structured_plan and candidate_payload.structured_plan.get("target_profile")
    assert candidate_payload.target_profile and candidate_payload.target_profile.get("targets")
    assert candidate_payload.risk_block and candidate_payload.execution_rules
