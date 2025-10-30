from __future__ import annotations

from datetime import datetime, timezone

from urllib.parse import urlsplit, parse_qs

from src.agent_server import _SCAN_STYLE_ANY, _SCAN_SYMBOL_REGISTRY, _planning_scan_to_page
from src.app.services.chart_url import make_chart_url
from src.planning.planning_scan import PlanningScanOutput
from src.schemas import ScanRequest
from src.services.contract_rules import ContractTemplate
from src.services.scan_engine import PlanningCandidate
from src.services.universe import UniverseSnapshot


def _candidate(symbol: str, readiness: float, entry: float, stop: float, targets: list[float]) -> PlanningCandidate:
    metrics = {
        "entry_distance_pct": 0.8,
        "entry_distance_atr": 0.6,
        "bars_to_trigger": 2,
        "probability": 0.55,
        "actionability": 0.85,
        "risk_reward": 1.4,
        "target_meta": [{"label": "TP1", "prob_touch": 0.5}],
        "tp_reasons": [
            {
                "label": "TP1",
                "reason": "Nearest structure within 0.25×ATR",
                "snap_tag": "ORL",
                "candidate_nodes": [{"label": "ORL", "price": 99.1, "picked": True, "rr_multiple": 1.3}],
            }
        ],
        "runner_policy": {"type": "trail"},
        "snap_trace": ["tp_accept:ORL"],
        "expected_move": 3.2,
        "remaining_atr": 1.1,
        "em_used": True,
        "key_levels_used": {"session": [{"label": "ORL", "role": "tp1", "price": 99.1}]},
        "entry_candidates": [{"label": "pullback", "price": entry}],
    }
    components = {key: metrics[key] for key in ("probability", "actionability", "risk_reward")}
    return PlanningCandidate(
        symbol=symbol,
        readiness_score=readiness,
        components=components,
        metrics=metrics,
        levels={"entry": entry, "invalidation": stop, "targets": targets},
        contract_template=ContractTemplate("intraday", "CALL", (0.3, 0.4), (0, 3), 400, 5.0),
        requires_live_confirmation=False,
        missing_live_inputs=[],
    )


def test_scan_banners_and_ordering_no_fallback() -> None:
    universe = UniverseSnapshot(
        name="adhoc",
        source="test",
        as_of_utc=datetime(2024, 1, 2, tzinfo=timezone.utc),
        symbols=["AAPL", "MSFT"],
        metadata={"style": "intraday"},
    )
    candidates = [
        _candidate("AAPL", 0.82, 100.0, 101.2, [98.6, 97.4]),
        _candidate("MSFT", 0.78, 300.0, 303.2, [295.5, 292.0]),
    ]
    planning_output = PlanningScanOutput(
        as_of_utc=datetime(2024, 1, 2, tzinfo=timezone.utc),
        universe=universe,
        run_id=42,
        indices_context={},
        candidates=candidates,
    )

    request_payload = ScanRequest(universe="custom", style="intraday", limit=5)

    page = _planning_scan_to_page(
        planning_output,
        request_payload,
        base_banner="SESSION_OPEN",
        base_meta={"route": "planning"},
        base_data_quality={"expected_move": 1.25},
    )

    assert page.banner == "SESSION_OPEN"
    assert [candidate.symbol for candidate in page.candidates] == ["AAPL", "MSFT"]
    first = page.candidates[0]
    assert first.entry_distance_pct == 0.8
    assert first.bars_to_trigger == 2
    assert first.actionable_soon is True
    assert first.planning_snapshot["target_meta"][0]["label"] == "TP1"


def test_plan_requires_recent_scan_cursor() -> None:
    _SCAN_SYMBOL_REGISTRY.clear()
    user_id = "anonymous"
    session_token = "open|2024-01-02T00:00:00Z|intraday"
    symbols = ["NVDA", "MSFT"]
    _SCAN_SYMBOL_REGISTRY[(user_id, session_token, "intraday")] = symbols
    _SCAN_SYMBOL_REGISTRY[(user_id, session_token, _SCAN_STYLE_ANY)] = symbols

    assert _SCAN_SYMBOL_REGISTRY[(user_id, session_token, "intraday")] == symbols
    assert _SCAN_SYMBOL_REGISTRY[(user_id, session_token, _SCAN_STYLE_ANY)] == symbols


def test_plan_returns_canonical_chart_url() -> None:
    url = make_chart_url(
        {
            "symbol": "NVDA",
            "interval": "5",
            "direction": "short",
            "entry": 500.1234,
            "stop": 505.5678,
            "tp": [495.0, 490.0],
            "ui_state": {"style": "intraday"},
            "levels": "ORL,Session Low",
        },
        base_url="https://example.com/chart",
    )

    parsed = urlsplit(url)
    assert parsed.path in {"/chart", "/chart/"}
    params = parse_qs(parsed.query)
    assert params["symbol"] == ["NVDA"]
    assert params["entry"] == ["500.12"]
    assert params["stop"] == ["505.57"]
    assert params["tp"] == ["495,490"]
