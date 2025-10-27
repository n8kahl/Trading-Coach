from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from src.providers.options import select_contracts
import src.providers.options as options_provider


def _chain_row(option_type: str, strike: float, delta: float, dte: int, spread: float, oi: int, volume: int) -> dict:
    return {
        "symbol": f"OPT-{option_type}-{strike}",
        "option_type": option_type,
        "strike": strike,
        "delta": delta,
        "dte": float(dte),
        "spread_pct": spread,
        "open_interest": oi,
        "volume": volume,
        "bid": 1.0,
        "ask": 1.1,
        "mid": 1.05,
        "expiration_date": "2024-06-21",
    }


@pytest.mark.asyncio()
async def test_options_selection_returns_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = pd.DataFrame(
        [
            _chain_row("call", 105.0, 0.42, 30, 0.05, 800, 400),
            _chain_row("call", 110.0, 0.25, 15, 0.20, 10, 5),
        ]
    )

    async def fake_chain(*args, **kwargs):
        return frame

    monkeypatch.setattr(options_provider, "fetch_polygon_option_chain_asof", fake_chain)

    plan = {"direction": "long", "targets": [107.0], "bias": "long"}
    result = await select_contracts("AAPL", datetime(2024, 6, 10, tzinfo=timezone.utc), plan)
    assert result["options_contracts"]
    assert result["options_contracts"][0]["option_type"] == "call"
    note = result.get("options_note")
    if note is not None:
        assert "Contracts" in note
    for contract in result["options_contracts"]:
        assert contract.get("rating") in {"green", "yellow", "red"}
        assert isinstance(contract.get("reasons"), list)


@pytest.mark.asyncio()
async def test_options_selection_returns_note_when_filtered(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = pd.DataFrame(
        [
            _chain_row("put", 95.0, -0.10, 5, 0.50, 2, 1),
        ]
    )

    async def fake_chain(*args, **kwargs):
        return frame

    monkeypatch.setattr(options_provider, "fetch_polygon_option_chain_asof", fake_chain)

    plan = {"direction": "short", "targets": [93.0], "bias": "short"}
    result = await select_contracts("AAPL", datetime(2024, 6, 10, tzinfo=timezone.utc), plan)
    picks = result.get("options_contracts") or []
    assert picks, "Expect fallback contracts even when guardrails trigger"
    for contract in picks:
        flags = contract.get("guardrail_flags") or []
        assert flags, "fallback contracts should include guardrail flags"
        assert contract.get("rating") in {"green", "yellow", "red"}
        assert isinstance(contract.get("reasons"), list)
    note = result.get("options_note")
    assert note and "Contracts" in note
    assert result.get("rejected_contracts")


@pytest.mark.asyncio()
async def test_delta_missing_fallback_returns_yellow_contracts(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = pd.DataFrame(
        [
            _chain_row("call", 100.0, None, 5, 5.0, 1500, 800),
            _chain_row("call", 101.0, None, 6, 4.5, 1200, 600),
            _chain_row("call", 102.0, None, 7, 4.0, 1100, 500),
        ]
    )

    async def fake_chain(*args, **kwargs):
        return frame

    monkeypatch.setattr(options_provider, "fetch_polygon_option_chain_asof", fake_chain)

    plan = {"direction": "long", "style": "intraday"}
    result = await select_contracts("AAPL", datetime(2025, 10, 27, 20, 0, tzinfo=timezone.utc), plan)
    contracts = result.get("options_contracts") or []
    assert contracts, "Expected delta-missing fallback contracts"
    assert len(contracts) >= 2, "Fallback should return at least two contracts"
    for contract in contracts:
        flags = contract.get("guardrail_flags") or []
        assert "DELTA_MISSING" in flags or "DELTA_MISSING_FALLBACK" in flags
        assert contract.get("rating") == "yellow"
        assert contract.get("status") == "degraded"
        assert contract.get("reason") == "delta_missing_fallback"
