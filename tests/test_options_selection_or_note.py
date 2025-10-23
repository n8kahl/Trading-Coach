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
    assert note is None or "contracts excluded" in note or note == "Filters applied"


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
    assert result["options_contracts"] == []
    assert result["options_note"]
    assert result["rejected_contracts"]
