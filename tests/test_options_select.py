import math

from app.engine.options_select import build_example_leg, score_contract


def test_score_contract_prefers_tight_spread_and_delta():
    base_contract = {
        "symbol": "TSLA231215C00200000",
        "option_type": "call",
        "strike": 200.0,
        "expiration_date": "2023-12-15",
        "bid": 5.1,
        "ask": 5.2,
        "spread_pct": 0.019,
        "volume": 1500,
        "open_interest": 3200,
        "delta": 0.52,
    }
    scored = score_contract(dict(base_contract), prefer_delta=0.5)
    assert 0.0 < scored.score <= 1.0

    wider_spread = dict(base_contract, spread_pct=0.40, bid=5.0, ask=6.0)
    scored_wide = score_contract(wider_spread, prefer_delta=0.5)
    assert scored_wide.score < scored.score


def test_build_example_leg_returns_compact_payload():
    contract = {
        "symbol": "TSLA231215C00200000",
        "option_type": "call",
        "strike": 200.0,
        "expiration_date": "2023-12-15",
        "liquidity_score": 0.78,
        "bid": 5.1,
        "ask": 5.2,
        "spread_pct": 0.019,
        "delta": 0.52,
    }
    leg = build_example_leg(contract)
    assert leg["symbol"] == contract["symbol"]
    assert leg["type"] == "call"
    assert leg["strike"] == contract["strike"]
    assert leg["score"] == contract["liquidity_score"]
