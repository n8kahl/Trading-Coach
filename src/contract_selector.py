"""Option contract selection utilities.

These functions help choose appropriate option contracts given a strategy’s
rules and a snapshot of the available option chain.  They enforce liquidity
filters (spread, open interest, volume) and target a delta range and days
until expiry (DTE).

Polygon’s option chain snapshot endpoint returns detailed data including
bid/ask prices, volume, open interest, implied volatility and Greeks【220893600390724†screenshot】.  The
functions below expect the chain to be preloaded into a pandas DataFrame or a
similar structure.
"""

from __future__ import annotations

import pandas as pd
from typing import List, Dict


def filter_chain(chain: pd.DataFrame, rules: Dict[str, object]) -> pd.DataFrame:
    """Filter an option chain based on strategy rules.

    Args:
        chain: DataFrame containing the option chain.  Required columns:
            - `delta`: option delta (float)
            - `dte`: days to expiry (int or float)
            - `spread_pct`: bid/ask spread divided by mid price (float)
            - `open_interest`: open interest (int)
            - `volume`: trading volume (int)
        rules: Dictionary defining `dte_range`, `delta_range`, `max_spread_pct`,
            `min_open_interest`, and `min_volume`.  Additional fields are ignored.

    Returns:
        A filtered DataFrame containing only contracts that satisfy the rules.
    """
    df = chain.copy()
    dte_low, dte_high = rules.get("dte_range", (0, 365))
    delta_low, delta_high = rules.get("delta_range", (0.0, 1.0))
    max_spread = rules.get("max_spread_pct", 1.0)
    min_oi = rules.get("min_open_interest", 0)
    min_vol = rules.get("min_volume", 0)
    df = df[
        (df["dte"] >= dte_low)
        & (df["dte"] <= dte_high)
        & (df["delta"].abs() >= delta_low)
        & (df["delta"].abs() <= delta_high)
        & (df["spread_pct"] <= max_spread)
        & (df["open_interest"] >= min_oi)
        & (df["volume"] >= min_vol)
    ]
    return df


def pick_best_contract(filtered_chain: pd.DataFrame, prefer_delta: float | None = None) -> pd.Series | None:
    """Select the best option contract from a filtered chain.

    This function implements a simple heuristic: choose the contract whose delta
    is closest to `prefer_delta` (if provided) and which has the tightest
    spread.  In practice you may want to add more sophisticated scoring.

    Args:
        filtered_chain: The filtered DataFrame returned by `filter_chain`.
        prefer_delta: Optional float indicating the ideal delta (e.g. 0.50 for
            at‑the‑money contracts).  If not provided, the contract with the
            highest volume/open interest ratio is selected.

    Returns:
        A pandas Series representing the selected contract, or `None` if no
        contracts satisfy the filters.
    """
    if filtered_chain.empty:
        return None
    df = filtered_chain.copy().reset_index(drop=True)
    if prefer_delta is not None:
        df["delta_score"] = (df["delta"].abs() - prefer_delta).abs()
        df = df.sort_values(by=["delta_score", "spread_pct", "dte"])
    else:
        df = df.sort_values(by=["spread_pct", "dte", "volume"], ascending=[True, True, False])
    return df.iloc[0]