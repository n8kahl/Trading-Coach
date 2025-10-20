import pandas as pd

from src.agent_server import _apply_option_guardrails, _screen_contracts


def _make_chain() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": "OPT_PASS",
                "option_type": "call",
                "bid": 1.0,
                "ask": 1.08,
                "mid": 1.04,
                "dte": 12,
                "delta": 0.52,
                "open_interest": 500,
                "volume": 200,
                "strike": 100,
            },
            {
                "symbol": "OPT_LOW_DELTA",
                "option_type": "call",
                "bid": 0.85,
                "ask": 0.95,
                "mid": 0.9,
                "dte": 14,
                "delta": 0.18,
                "open_interest": 900,
                "volume": 400,
                "strike": 105,
            },
            {
                "symbol": "OPT_WIDE_SPREAD",
                "option_type": "call",
                "bid": 0.9,
                "ask": 1.2,
                "mid": 1.05,
                "dte": 10,
                "delta": 0.5,
                "open_interest": 600,
                "volume": 120,
                "strike": 98,
                "spread_pct": 0.2,
            },
        ]
    )


def test_screen_contracts_rejections_include_explicit_reasons():
    chain = _make_chain()
    quotes = {}
    filters = {
        "min_dte": 5,
        "max_dte": 30,
        "min_delta": 0.4,
        "max_delta": 0.6,
        "max_spread_pct": 12.0,
        "min_oi": 200,
    }

    screened = _screen_contracts(
        chain,
        quotes,
        symbol="XYZ",
        style="intraday",
        side="call",
        filters=filters,
    )

    candidates = list(screened)
    rejections = getattr(screened, "rejections", [])

    assert len(candidates) == 1
    assert {item["symbol"] for item in candidates} == {"OPT_PASS"}
    reason_codes = {entry["reason"] for entry in rejections}
    assert "DELTA_OUT_OF_RANGE" in reason_codes
    assert "SPREAD_TOO_WIDE" in reason_codes
    assert all(entry["reason"].isupper() for entry in rejections)
    assert any("message" in entry for entry in rejections)


def test_option_guardrails_reject_and_explain():
    contracts = [
        {"symbol": "AAA 2024-01-19 100C", "spread_pct": 12.5, "open_interest": 500, "bid": 1.0, "ask": 1.3},
        {"symbol": "BBB 2024-01-19 100C", "spread_pct": 4.0, "open_interest": 50, "bid": 2.0, "ask": 2.2},
        {"symbol": "CCC 2024-01-19 100C", "spread_pct": 3.0, "open_interest": 800, "bid": 1.5, "ask": 1.55},
    ]

    filtered, rejected = _apply_option_guardrails(contracts, max_spread_pct=8.0, min_open_interest=200)

    assert len(filtered) == 1
    assert filtered[0]["symbol"] == "CCC 2024-01-19 100C"
    reason_codes = {entry["reason"] for entry in rejected}
    assert "SPREAD_TOO_WIDE" in reason_codes
    assert "OPEN_INTEREST_TOO_LOW" in reason_codes
    assert all("message" in entry for entry in rejected)
