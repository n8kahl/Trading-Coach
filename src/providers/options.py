from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from ..polygon_options import fetch_polygon_option_chain_asof, summarize_polygon_chain


async def select_contracts(symbol: str, as_of: datetime, plan: Dict[str, Any]) -> Dict[str, Any]:
    """Select option contracts for a plan using Polygon snapshot data."""

    direction = (plan.get("direction") or plan.get("bias") or "long").lower()
    option_type = "call" if direction == "long" else "put"
    chain = await fetch_polygon_option_chain_asof(symbol, as_of)
    if chain is None or chain.empty:
        return {
            "options_contracts": [],
            "rejected_contracts": [],
            "options_note": "Polygon option chain unavailable",
        }

    filtered_chain = chain[chain["option_type"] == option_type]
    if filtered_chain.empty:
        return {
            "options_contracts": [],
            "rejected_contracts": [
                {
                    "symbol": symbol,
                    "reason": "option_type_mismatch",
                    "message": f"No {option_type} contracts available",
                }
            ],
            "options_note": f"No {option_type} contracts met selection rules",
        }

    rules = {
        "dte_range": (7, 60),
        "delta_range": (0.25, 0.6),
        "max_spread_pct": 0.15,
        "min_open_interest": 50,
        "min_volume": 10,
    }
    eligible = filtered_chain[
        (filtered_chain["dte"].astype(float) >= rules["dte_range"][0])
        & (filtered_chain["dte"].astype(float) <= rules["dte_range"][1])
        & (filtered_chain["delta"].abs() >= rules["delta_range"][0])
        & (filtered_chain["delta"].abs() <= rules["delta_range"][1])
        & (filtered_chain["spread_pct"] <= rules["max_spread_pct"])
        & (filtered_chain["open_interest"] >= rules["min_open_interest"])
        & (filtered_chain["volume"] >= rules["min_volume"])
    ].copy()

    if eligible.empty:
        return {
            "options_contracts": [],
            "rejected_contracts": [
                {
                    "symbol": symbol,
                    "reason": "no_contracts_after_filters",
                    "message": "All contracts filtered out by liquidity/delta rules",
                }
            ],
            "options_note": "No contracts satisfied delta/liquidity filters",
        }

    summary = summarize_polygon_chain(eligible, rules, top_n=3)
    if not summary:
        return {
            "options_contracts": [],
            "rejected_contracts": [
                {
                    "symbol": symbol,
                    "reason": "no_contracts_after_filters",
                    "message": "All contracts filtered out by liquidity/delta rules",
                }
            ],
            "options_note": "No contracts satisfied delta/liquidity filters",
        }

    contracts: List[Dict[str, Any]] = []
    if summary.get("best"):
        contracts.append(summary["best"])
    for alt in summary.get("alternatives") or []:
        contracts.append(alt)

    note = None
    rejected_contracts: List[Dict[str, Any]] = []
    filtered_size = int(summary.get("considered_size") or 0)
    original_size = int(summary.get("chain_size") or filtered_size)
    rejected_size = max(original_size - filtered_size, 0)
    if summary.get("filters_applied"):
        note = f"{rejected_size} contracts excluded by delta/liquidity rules" if rejected_size else "Filters applied"
        if rejected_size > 0:
            rejected_contracts.append(
                {
                    "symbol": symbol,
                    "reason": "filters_applied",
                    "message": "Excluded by delta/liquidity rules",
                }
            )

    return {
        "options_contracts": contracts,
        "rejected_contracts": rejected_contracts,
        "options_note": note,
    }


__all__ = ["select_contracts"]
