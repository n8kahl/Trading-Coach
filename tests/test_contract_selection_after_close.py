import pandas as pd
import pytest
from datetime import datetime, timezone

from src.providers.options import select_contracts


@pytest.mark.asyncio
async def test_contracts_return_after_close(monkeypatch):
    symbol = "AAPL"
    as_of = datetime(2025, 10, 27, 20, 0, 0, tzinfo=timezone.utc)
    plan = {"direction": "long", "style": "intraday", "strategy_id": "base"}

    call_tracker = {"asof": False}

    async def fake_chain_asof(symbol_arg, as_of_arg, **kwargs):
        call_tracker["asof"] = True
        records = []
        for strike in (200, 205, 210):
            records.append(
                {
                    "symbol": f"{symbol_arg}C{strike}",
                    "option_type": "call",
                    "strike": strike,
                    "bid": 1.0,
                    "ask": 1.1,
                    "mid": 1.05,
                    "spread_pct": 5.0,
                    "open_interest": 1000,
                    "volume": 500,
                    "delta": None,
                    "dte": 5,
                }
            )
        return pd.DataFrame(records)

    async def fail_chain(*args, **kwargs):
        raise AssertionError("select_contracts should rely on as-of chain when market is closed")

    monkeypatch.setattr("src.providers.options.fetch_polygon_option_chain_asof", fake_chain_asof)
    monkeypatch.setattr("src.providers.options.fetch_polygon_option_chain", fail_chain)

    result = await select_contracts(symbol, as_of, plan)

    contracts = result["options_contracts"]
    assert call_tracker["asof"] is True
    assert len(contracts) >= 2, "Should return at least a pair of fallback contracts"
    assert any("DELTA_MISSING" in c.get("guardrail_flags", []) for c in contracts)
    assert all(c.get("rating") == "yellow" for c in contracts if "DELTA_MISSING" in c.get("guardrail_flags", []))
