"""Declarative strategy library for the trading assistant.

This module defines a collection of strategy descriptions that the market
scanner uses to detect and rank trade setups.  Each strategy entry is a
dictionary containing the following keys:

* `id` – A short identifier used in logs and API responses.
* `name` – Human‑readable name of the strategy.
* `category` – One of `scalp`, `intraday`, `swing`, `leap`, or `index`.
* `description` – A high‑level explanation of the trading idea.
* `triggers` – A list of conditions (written in plain language) that must
  occur for the strategy to be considered active.
* `options_rules` – A dictionary describing how to select option contracts for
  this strategy (delta range, days to expiry, minimum open interest, etc.).
* `stops_tp` – Guidance on stop‑loss placement and take‑profit targets.
* `win_rate_target` – A float representing the aspirational historical win
  rate.  The scanner uses this as a threshold when ranking signals.

These definitions are not hard‑coded trading algorithms; they serve as
metadata for the scanner.  You can edit, add, or remove entries without
modifying the scanning logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Dict, Optional


@dataclass
class Strategy:
    id: str
    name: str
    category: str
    description: str
    triggers: List[str]
    options_rules: Dict[str, object]
    stops_tp: str
    win_rate_target: Optional[float] = None


def load_strategies() -> List[Strategy]:
    """Return a list of predefined trading strategies.

    You can customize this list by adding new `Strategy` instances or modifying
    the parameters of existing ones.  The scanner uses these definitions to
    evaluate market conditions and produce ranked signals.
    """
    strategies: List[Strategy] = []

    # Intraday / scalps
    strategies.append(
        Strategy(
            id="orb_retest",
            name="Opening Range Breakout Retest",
            category="scalp",
            description=(
                "After the first 5–15 minutes of trading (the opening range), "
                "watch for price to break the range and then pull back to retest "
                "the range boundary.  If price reclaims the boundary with rising "
                "volume, enter in the direction of the break."
            ),
            triggers=[
                "Price closes above/below the opening range high/low",
                "Volume is at least 1.5× the 20‑bar median volume",
                "Short‑term EMAs (9/20) are aligned in the direction of the trade",
                "ADX(14) > 18 to confirm trend strength",
                "VWAP acts as support/resistance on the retest"
            ],
            options_rules={
                "dte_range": [0, 2],
                "delta_range": [0.50, 0.65],
                "max_spread_pct": 0.05,
                "min_open_interest": 500,
                "min_volume": 250
            },
            stops_tp=(
                "Place stop just beyond the retest candle’s low/high.  Take profit "
                "at +0.75–1.0× ATR(14) of the underlying.  Trail the remaining position "
                "using a chandelier stop (3× ATR)."
            ),
            win_rate_target=0.58,
        )
    )

    # Power Hour – last hour of RTH (3–4pm ET)
    strategies.append(
        Strategy(
            id="power_hour_trend",
            name="Power Hour Trend Continuation",
            category="intraday",
            description=(
                "During the last hour of regular trading (3–4pm ET), favor continuation in the "
                "prevailing intraday trend when price holds above/below VWAP and short EMAs are "
                "stacked. Look for an expansion out of the afternoon range with rising ADX."
            ),
            triggers=[
                "Time window: 3–4pm ET (Power Hour)",
                "Price holds above (long) or below (short) session VWAP",
                "EMA(9) > EMA(20) > EMA(50) for long; reversed for short",
                "ADX(14) rising and > 18",
            ],
            options_rules={
                "dte_range": [0, 2],
                "delta_range": [0.45, 0.65],
                "max_spread_pct": 0.05,
                "min_open_interest": 500,
                "min_volume": 250,
            },
            stops_tp=(
                "Stop under/over the afternoon range or last swing. Initial target 0.8–1.2× ATR, "
                "secondary target at session high/low or prior day high/low if within reach."
            ),
            win_rate_target=0.57,
        )
    )

    strategies.append(
        Strategy(
            id="vwap_avwap",
            name="VWAP + Anchored VWAP Cluster Continuation",
            category="scalp",
            description=(
                "Use session VWAP and anchored VWAPs (e.g. prior day high/low, earnings) "
                "as a cluster of support/resistance.  Enter when price closes above "
                "session VWAP and multiple anchored VWAPs with rising volume."
            ),
            triggers=[
                "Price closes above the session VWAP and at least two anchored VWAPs",
                "Distance between price and the anchored VWAP cluster is narrowing",
                "Volume is rising on the break",
            ],
            options_rules={
                "dte_range": [0, 3],
                "delta_range": [0.45, 0.60],
                "max_spread_pct": 0.05,
                "min_open_interest": 500,
                "min_volume": 250
            },
            stops_tp=(
                "Place stop just beyond the anchored VWAP cluster.  Initial target is "
                "1.0–1.5× ATR.  Scale 50% at 1× ATR and trail the rest."
            ),
            win_rate_target=0.56,
        )
    )

    # Gap Fill at Market Open
    strategies.append(
        Strategy(
            id="gap_fill_open",
            name="Gap Fill at Open",
            category="scalp",
            description=(
                "In the first 15–30 minutes after the open, play toward prior close when an overnight "
                "gap fails to hold. Look for a move back into the prior day’s range, VWAP cross, and "
                "volume confirmation."
            ),
            triggers=[
                "Time window: 9:30–10:00 ET",
                "Overnight gap ≥ 0.3–0.5× ATR(14) or ≥ 0.3%",
                "Reversion flow toward prior close; VWAP cross in the fill direction",
            ],
            options_rules={
                "dte_range": [0, 2],
                "delta_range": [0.45, 0.65],
                "max_spread_pct": 0.06,
                "min_open_interest": 300,
                "min_volume": 200,
            },
            stops_tp=(
                "Stop beyond the morning swing/OR boundary against the fill. Targets: prior close, then prior day VWAP or range midpoint."
            ),
            win_rate_target=0.55,
        )
    )

    # Midday mean reversion when trend is weak
    strategies.append(
        Strategy(
            id="midday_mean_revert",
            name="Midday VWAP Mean Reversion",
            category="intraday",
            description=(
                "During the midday chop (≈11:30–14:00 ET), fade extensions away from VWAP when trend "
                "strength (ADX) is weak and ranges compress."
            ),
            triggers=[
                "Time window: ≈11:30–14:00 ET",
                "ADX(14) < 15 and contracting",
                "Price stretches > 0.6× ATR from VWAP and stalls",
            ],
            options_rules={
                "dte_range": [0, 2],
                "delta_range": [0.35, 0.55],
                "max_spread_pct": 0.06,
                "min_open_interest": 300,
                "min_volume": 200,
            },
            stops_tp=(
                "Stop beyond the extension swing; target VWAP, then opposite band if range persists."
            ),
            win_rate_target=0.53,
        )
    )

    strategies.append(
        Strategy(
            id="liquidity_sweep",
            name="Liquidity Sweep & Reclaim",
            category="scalp",
            description=(
                "Fade stop runs.  When price spikes through a prior swing high/low with "
                "a large wick and then closes back inside the range on heavy volume, "
                "enter counter‑trend for a move back to VWAP or range midpoint."
            ),
            triggers=[
                "Large wick that pierces a swing level and closes back inside",
                "Volume spike Z‑score > 2.0",
                "RSI(14, 1m) crosses back above/below 35/65",  # depending on direction
            ],
            options_rules={
                "dte_range": [0, 2],
                "delta_range": [0.35, 0.50],
                "max_spread_pct": 0.06,
                "min_open_interest": 300,
                "min_volume": 200
            },
            stops_tp=(
                "Stop goes just beyond the wick.  First target is the VWAP, second is the "
                "range midpoint.  Scale out on the way."
            ),
            win_rate_target=0.54,
        )
    )

    strategies.append(
        Strategy(
            id="inside_breakout",
            name="Inside Bar (NR7) Breakout",
            category="scalp",
            description=(
                "Trade volatility contraction.  When a 5‑minute candle has a range inside "
                "the prior candle’s range (or the narrowest range of the last 7 bars), "
                "a breakout with volume can lead to a sharp move."
            ),
            triggers=[
                "5m bar is an inside bar or the narrowest of the last 7 bars",
                "ADX is rising, signalling a potential trend expansion",
                "Breakout occurs with increased volume"
            ],
            options_rules={
                "dte_range": [0, 3],
                "delta_range": [0.45, 0.55],
                "max_spread_pct": 0.05,
                "min_open_interest": 500,
                "min_volume": 250
            },
            stops_tp=(
                "Stop on the opposite side of the inside bar.  Target 1.2–1.8× ATR."
            ),
            win_rate_target=0.55,
        )
    )

    # Swing / multi‑session / LEAPS
    strategies.append(
        Strategy(
            id="ema_pullback",
            name="EMA Trend Pullback (20/50/200)",
            category="swing",
            description=(
                "Classic trend strategy.  When the 20, 50 and 200 EMAs are stacked in the "
                "direction of the trend and price pulls back to the 20‑EMA near a daily pivot "
                "or prior swing high/low on decreasing volume, enter in the trend direction."
            ),
            triggers=[
                "EMAs (20/50/200) are stacked bullish or bearish",
                "Price pulls back to the 20‑EMA with declining volume",
                "Stochastic RSI turns back in the direction of the trend"
            ],
            options_rules={
                "dte_range": [20, 45],
                "delta_range": [0.30, 0.40],
                "strategy_type": "debit_spread",
                "max_spread_pct": 0.05,
                "min_open_interest": 1000,
                "min_volume": 500
            },
            stops_tp=(
                "Stop below the pullback swing.  Target 1.5–2.5× the daily ATR.  A vertical debit "
                "spread (long ATM, short slightly OTM) helps reduce theta decay."
            ),
            win_rate_target=0.52,
        )
    )

    strategies.append(
        Strategy(
            id="earnings_drift",
            name="Earnings Drift with IV Crush Bias",
            category="swing",
            description=(
                "After earnings announcements, implied volatility collapses and price often drifts "
                "in the direction of the surprise.  Play this drift with delta‑neutral or directional "
                "structures."
            ),
            triggers=[
                "Earnings gap in the direction of the surprise",
                "Price holds above/below the AVWAP anchored to the earnings candle",
                "Polygon news sentiment is positive/negative to confirm direction"
            ],
            options_rules={
                "dte_range": [15, 30],
                "delta_range": [0.30, 0.50],
                "strategy_type": "debit_or_calendar",
                "max_spread_pct": 0.05,
                "min_open_interest": 1000,
                "min_volume": 500
            },
            stops_tp=(
                "Use the low/high of the earnings candle as your stop.  Target depends on sector "
                "volatility; a 1.6× ATR move is typical.  Consider calendars if expecting IV mean reversion."
            ),
            win_rate_target=0.55,
        )
    )

    strategies.append(
        Strategy(
            id="gap_fill_breakaway",
            name="Gap Fill vs Breakaway",
            category="swing",
            description=(
                "Distinguish between exhaustion gaps that tend to fill and breakaway gaps that "
                "trend.  Use pre‑market levels and AVWAP to decide."
            ),
            triggers=[
                "Gap into a higher time frame supply/demand zone or away from it",
                "For fills: fading volume and failure to hold AVWAP(Gap)",
                "For breakaways: price holds above AVWAP(Gap) with sustained volume"
            ],
            options_rules={
                "dte_range": [7, 21],
                "delta_range": [0.35, 0.50],
                "max_spread_pct": 0.05,
                "min_open_interest": 800,
                "min_volume": 400
            },
            stops_tp=(
                "Stop depends on gap type: for fills, use the extremes of the gap range; for breakaways, "
                "use the retest of the gap base.  Target 1.4–2.0× ATR."
            ),
            win_rate_target=0.55,
        )
    )

    strategies.append(
        Strategy(
            id="high_iv_rank",
            name="High IV Rank Credit Spread",
            category="swing",
            description=(
                "When implied volatility rank is elevated (> 50), sell out‑of‑the‑money credit "
                "spreads to collect premium while maintaining a defined risk.  Probability of "
                "profit is estimated using the Black–Scholes distribution."
            ),
            triggers=[
                "Implied volatility rank > 50",
                "No major catalysts (earnings, FOMC) in the next two weeks",
                "Underlying price is not at an extreme trend extension"
            ],
            options_rules={
                "dte_range": [30, 45],
                "delta_range": [0.25, 0.30],
                "strategy_type": "credit_spread",
                "spread_width": 2.0,
                "min_open_interest": 1500,
                "min_volume": 800
            },
            stops_tp=(
                "Maximum loss is capped at the width minus credit.  Manage risk by sizing small and "
                "taking profits at 50–60% of the credit received when IV falls or time passes."
            ),
            win_rate_target=0.65,
        )
    )

    strategies.append(
        Strategy(
            id="pmcc",
            name="Poor Man’s Covered Call (PMCC)",
            category="leap",
            description=(
                "Long a deep in‑the‑money LEAPS call and sell short‑dated calls against it on rips "
                "to collect weekly income while retaining long exposure."
            ),
            triggers=[
                "Long‑term bullish bias on the underlying",
                "Short‑term overbought condition to sell a call against the LEAPS",
                "Implied volatility term structure steep enough to justify a diagonal"
            ],
            options_rules={
                "dte_range": [180, 365],
                "delta_range": [0.70, 0.85],
                "strategy_type": "diagonal",
                "min_open_interest": 1000,
                "min_volume": 500
            },
            stops_tp=(
                "Use the long LEAPS as core exposure.  Roll or buy back short calls when delta rises "
                "above 0.80 or at 50% of max gain.  Take profit on the LEAPS when the trend shows signs "
                "of exhaustion."
            ),
            win_rate_target=0.60,
        )
    )

    # Index‑specific strategies (SPX/NDX)
    strategies.append(
        Strategy(
            id="gex_flip",
            name="Gamma Exposure Flip",
            category="index",
            description=(
                "Compute net gamma exposure (GEX) across index options.  When the underlying crosses "
                "the zero‑gamma level, expect a regime shift between trend and mean‑reversion."
            ),
            triggers=[
                "Price crosses the zero‑gamma level computed from net GEX",
                "Futures volume is rising in the direction of the cross",
                "Polygon news sentiment supports the move"
            ],
            options_rules={
                "dte_range": [0, 5],
                "delta_range": [0.40, 0.60],
                "min_open_interest": 2000,
                "min_volume": 1000
            },
            stops_tp=(
                "Use a tight stop (0.5× ATR) on intraday plays.  For trend days above zero‑gamma, "
                "target 1.5× ATR; for mean‑reversion days below zero‑gamma, fade at VWAP and target "
                "reversion to the mean."
            ),
            win_rate_target=0.56,
        )
    )

    strategies.append(
        Strategy(
            id="vix_gate",
            name="VWAP + VIX Regime Gate",
            category="index",
            description=(
                "Combine index VWAP with volatility regime.  In low VIX environments, favour breakouts; "
                "in high VIX environments, favour mean‑reversion scalps."
            ),
            triggers=[
                "VIX < threshold: look for breakouts above VWAP", 
                "VIX > threshold: fade moves back toward VWAP",
            ],
            options_rules={
                "dte_range": [0, 2],
                "delta_range": [0.45, 0.60],
                "min_open_interest": 1500,
                "min_volume": 800
            },
            stops_tp=(
                "For breakouts, stop below VWAP; for fades, stop above the session high/low.  Target "
                "1.3–1.8× ATR depending on volatility."
            ),
            win_rate_target=0.57,
        )
    )

    strategies.append(
        Strategy(
            id="inside_day_rth",
            name="Inside‑Day then Regular Trading Hours Expansion",
            category="index",
            description=(
                "When a daily candle is completely inside the previous day’s range, the following "
                "regular trading session often produces a strong move once the range is broken."
            ),
            triggers=[
                "Daily range is inside the prior day’s range",
                "ORB (regular session) break aligns with higher time frame bias",
                "Volume confirms the breakout"
            ],
            options_rules={
                "dte_range": [0, 3],
                "delta_range": [0.45, 0.60],
                "min_open_interest": 1500,
                "min_volume": 800
            },
            stops_tp=(
                "Stop on the opposite side of the daily inside bar.  Target 1.5× ATR or key level."
            ),
            win_rate_target=0.55,
        )
    )

    # Additional generic strategies described by the user
    strategies.append(
        Strategy(
            id="break_and_retest",
            name="Break and Retest",
            category="intraday",
            description=(
                "Enter trades on a break of a key level and subsequent retest of the level.  This "
                "pattern is applicable across all timeframes and underlies many specific setups."
            ),
            triggers=[
                "Key level (support/resistance) is broken with conviction",
                "Price retests the level without closing beyond it",
                "Volume confirms the validity of the break and retest"
            ],
            options_rules={
                "dte_range": [0, 10],
                "delta_range": [0.40, 0.60],
                "min_open_interest": 300,
                "min_volume": 200
            },
            stops_tp=(
                "Stop beyond the retest low/high.  First target is the measured move based on the "
                "range of the broken level; second targets come from ATR or prior pivots."
            ),
            win_rate_target=None,
        )
    )

    strategies.append(
        Strategy(
            id="momentum_play",
            name="Momentum Play",
            category="intraday",
            description=(
                "Trade strong momentum moves identified by increasing volume, widening range, and "
                "EMA alignment.  Enter in the direction of momentum and exit on signs of exhaustion."
            ),
            triggers=[
                "Price makes consecutive higher highs (long) or lower lows (short)",
                "Volume increases on the impulsive moves",
                "Short‑term EMAs align with the trade direction"
            ],
            options_rules={
                "dte_range": [0, 5],
                "delta_range": [0.50, 0.65],
                "min_open_interest": 500,
                "min_volume": 300
            },
            stops_tp=(
                "Use a dynamic stop such as a trailing ATR or supertrend.  Scale out as momentum "
                "slows or as the underlying approaches a key level."
            ),
            win_rate_target=None,
        )
    )

    return strategies


def _normalize_category_token(category: Optional[str]) -> str:
    token = (category or "").strip().lower()
    if token in {"leaps", "leap"}:
        return "leap"
    if token == "index":
        return "intraday"
    if token in {"scalp", "intraday", "swing"}:
        return token
    return "intraday"


def _public_category_token(category: Optional[str]) -> str:
    normalized = _normalize_category_token(category)
    if normalized == "leap":
        return "leaps"
    return normalized


@lru_cache(maxsize=1)
def _strategy_category_map() -> Dict[str, str]:
    return {strategy.id.lower(): _normalize_category_token(strategy.category) for strategy in load_strategies()}


def strategy_internal_category(strategy_id: str) -> str:
    """Return the internal style token used for calculations."""
    return _strategy_category_map().get((strategy_id or "").lower(), "intraday")


def strategy_public_category(strategy_id: str) -> str:
    """Return the public style token aligned with the OpenAPI schema."""
    return _public_category_token(strategy_internal_category(strategy_id))


def normalize_style_input(style: Optional[str]) -> Optional[str]:
    """Normalize user-supplied style filters to internal tokens."""
    if style is None:
        return None
    token = _normalize_category_token(style)
    return token


def public_style(style: Optional[str]) -> Optional[str]:
    if style is None:
        return None
    return _public_category_token(style)
