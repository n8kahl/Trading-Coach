"""Market scanning logic for detecting trade setups.

The `scan_market` function iterates over a universe of tickers, calculates
technical indicators using historical and/or streaming data, and matches
conditions defined in the strategy library.  The output is a list of signals
with scores that reflect the quality of the setup.

This starter implementation includes only a skeleton.  You will need to
integrate Polygon’s WebSocket and REST APIs to fetch real‑time bars and
option chain data.  See the README for suggestions on how to extend this.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Any

import pandas as pd

from .strategy_library import load_strategies, Strategy
from .calculations import atr, ema, vwap, adx
from .contract_selector import filter_chain, pick_best_contract


# --- Session/time utilities (America/New_York) ---
def _tz_et_index(frame: pd.DataFrame) -> pd.DatetimeIndex:
    idx = frame.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    return idx.tz_convert("America/New_York")


def _session_phase(ts: pd.Timestamp) -> str:
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC").tz_convert("America/New_York")
    else:
        ts = ts.tz_convert("America/New_York")
    h, m = ts.hour, ts.minute
    wd = ts.weekday()
    if wd >= 5:
        return "off"
    if (h < 9) or (h == 9 and m < 30):
        return "premarket"
    if h == 9 and m >= 30 and m < 60:
        return "open_drive"  # 9:30–10:00
    if (h == 10) or (h == 11 and m < 30):
        return "morning"
    if (h == 11 and m >= 30) or (h >= 12 and h < 14) or (h == 14 and m == 0):
        return "midday"
    if (h == 14 and m > 0):
        return "afternoon"
    if h == 15:
        return "power_hour"
    if h >= 16:
        return "postmarket"
    return "other"


def _get_prev_close(frame: pd.DataFrame) -> float | None:
    et = _tz_et_index(frame)
    dates = pd.Series(et.date).unique()
    if len(dates) < 2:
        return None
    prev_date = dates[-2]
    prev_mask = et.date == prev_date
    prev_df = frame.loc[prev_mask]
    if prev_df.empty:
        return None
    return float(prev_df["close"].iloc[-1])


def _get_first_open(frame: pd.DataFrame) -> float | None:
    et = _tz_et_index(frame)
    today = et[-1].date()
    today_mask = et.date == today
    today_df = frame.loc[today_mask]
    if today_df.empty:
        return None
    return float(today_df["open"].iloc[0])


@dataclass
class Signal:
    """Represents a detected trade opportunity."""
    symbol: str
    strategy_id: str
    description: str
    score: float
    contract: Dict[str, Any] | None = None
    features: Dict[str, Any] = field(default_factory=dict)
    options_rules: Dict[str, Any] | None = None


async def scan_market(tickers: List[str], market_data: Dict[str, pd.DataFrame]) -> List[Signal]:
    """Scan the provided tickers for strategy setups.

    Args:
        tickers: List of stock tickers (strings) to scan.
        market_data: A dictionary mapping tickers to pandas DataFrames containing
            recent OHLCV data.  Each DataFrame should have columns: `open`, `high`,
            `low`, `close`, `volume` and an index of timestamps.  In a full
            implementation this data would come from Polygon’s minute aggregate
            endpoint and be updated continuously via WebSocket.

    Returns:
        A list of `Signal` objects sorted by descending score.
    """
    strategies = load_strategies()
    signals: List[Signal] = []

    for symbol in tickers:
        df = market_data.get(symbol)
        if df is None or len(df) < 20:
            continue  # not enough data to compute indicators

        # Precompute some indicators for this symbol
        df = df.copy()
        df["atr14"] = atr(df["high"], df["low"], df["close"], 14)
        df["ema9"] = ema(df["close"], 9)
        df["ema20"] = ema(df["close"], 20)
        df["ema50"] = ema(df["close"], 50)
        df["vwap"] = vwap(df["close"], df["volume"])
        df["adx14"] = adx(df["high"], df["low"], df["close"], 14)

        latest = df.iloc[-1]

        for strategy in strategies:
            score = evaluate_strategy(strategy, df, latest)
            if score > 0:
                # In a complete implementation you would fetch the option chain
                # snapshot from Polygon here and select an appropriate contract.
                # For now we leave `contract` as None.
                # Derive a direction bias for certain strategies (e.g., Power Hour)
                direction_bias = None
                sid = strategy.id.lower()
                if sid == "power_hour_trend":
                    if latest["close"] > latest["vwap"] and (latest["ema9"] > latest["ema20"] > latest["ema50"]):
                        direction_bias = "long"
                    elif latest["close"] < latest["vwap"] and (latest["ema9"] < latest["ema20"] < latest["ema50"]):
                        direction_bias = "short"
                elif sid == "gap_fill_open":
                    prev_close = _get_prev_close(df)
                    # If gapped up and filling toward prev close -> short bias; opposite for gap down
                    if prev_close is not None:
                        first_open = _get_first_open(df)
                        if first_open is not None:
                            if first_open > prev_close and latest["close"] < first_open:
                                direction_bias = "short"
                            elif first_open < prev_close and latest["close"] > first_open:
                                direction_bias = "long"
                elif sid == "midday_mean_revert":
                    # Bias is toward VWAP
                    if latest["close"] > latest["vwap"]:
                        direction_bias = "short"
                    elif latest["close"] < latest["vwap"]:
                        direction_bias = "long"
                signals.append(
                    Signal(
                        symbol=symbol,
                        strategy_id=strategy.id,
                        description=strategy.description,
                        score=score,
                        contract=None,
                        features={
                            "atr": latest["atr14"],
                            "adx": latest["adx14"],
                            "ema9": latest["ema9"],
                            "ema20": latest["ema20"],
                            "ema50": latest["ema50"],
                            "vwap": latest["vwap"],
                            "direction_bias": direction_bias,
                            "session_phase": _session_phase(df.index[-1]) if len(df.index) else None,
                        },
                        options_rules=strategy.options_rules,
                    )
                )

    # Sort signals by descending score
    signals.sort(key=lambda s: s.score, reverse=True)
    return signals


def evaluate_strategy(strategy: Strategy, df: pd.DataFrame, latest: pd.Series) -> float:
    """Assign a simple score to a potential setup.

    This placeholder function demonstrates how you might grade a strategy using
    indicator values and heuristics.  A real implementation should use the
    confluence engine described in the plan (e.g. logistic regression or
    gradient boosting on labelled outcomes).

    Args:
        strategy: The strategy definition to evaluate.
        df: Historical DataFrame for the symbol.
        latest: The most recent row from `df`.

    Returns:
        A float representing the strength of the signal.  Zero indicates no
        signal.
    """
    score = 0.0

    # Helper: detect Power Hour window using the latest bar timestamp (ET 15:00–16:00)
    def _is_power_hour(ts: pd.Timestamp) -> bool:
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        et = ts.tz_convert("America/New_York")
        # Only weekdays
        if et.weekday() >= 5:
            return False
        return et.hour == 15  # 3pm ET hour

    # Example heuristics for demonstration purposes only
    if strategy.id == "orb_retest":
        # Check if EMAs are aligned (bullish)
        if latest["ema9"] > latest["ema20"] > latest["ema50"]:
            score += 0.5
        # Check ADX is above threshold
        if latest["adx14"] > 18:
            score += 0.2
        # Additional heuristics would go here
    elif strategy.id == "vwap_avwap":
        if latest["close"] > latest["vwap"]:
            score += 0.4
        if latest["adx14"] > 15:
            score += 0.2
    elif strategy.id == "liquidity_sweep":
        # Suppose price made a long wick; for now we mock with random small probability
        pass
    elif strategy.id == "inside_breakout":
        # Here you would check for an inside bar; omitted for brevity
        pass
    elif strategy.id == "power_hour_trend":
        # Score only within Power Hour window
        ts = df.index[-1]
        phase = _session_phase(ts)
        if phase == "power_hour":
            # Trend alignment with VWAP and EMAs
            bullish = latest["close"] > latest["vwap"] and (latest["ema9"] > latest["ema20"] > latest["ema50"]) 
            bearish = latest["close"] < latest["vwap"] and (latest["ema9"] < latest["ema20"] < latest["ema50"]) 
            if bullish or bearish:
                score += 0.6
            # Trend strength
            if latest["adx14"] > 18:
                score += 0.25
            # Mild bonus if afternoon range (last ~30 bars on 1m/5m) is breaking
            try:
                window = df.tail(30)
                if bullish and latest["close"] >= window["high"].max():
                    score += 0.15
                if bearish and latest["close"] <= window["low"].min():
                    score += 0.15
            except Exception:
                pass
        else:
            score = 0.0
    elif strategy.id == "gap_fill_open":
        ts = df.index[-1]
        phase = _session_phase(ts)
        if phase in {"open_drive", "morning"}:
            prev_close = _get_prev_close(df)
            first_open = _get_first_open(df)
            if prev_close is not None and first_open is not None:
                gap = first_open - prev_close
                gap_abs = abs(gap)
                atr_val = float(latest["atr14"]) if pd.notna(latest["atr14"]) else float(df["high"].tail(14).sub(df["low"].tail(14)).median())
                # Basic gap size filter
                if gap_abs >= max(0.3 * atr_val, 0.003 * prev_close):
                    score += 0.45
                    # Filling toward prev close now?
                    filling = (gap > 0 and latest["close"] < first_open) or (gap < 0 and latest["close"] > first_open)
                    if filling:
                        score += 0.25
                    # VWAP cross in fill direction
                    vwap_ok = (gap > 0 and latest["close"] < latest["vwap"]) or (gap < 0 and latest["close"] > latest["vwap"])
                    if vwap_ok:
                        score += 0.2
                    # Still room to the prior close
                    dist = abs(latest["close"] - prev_close)
                    if dist > 0.2 * atr_val:
                        score += 0.1
        else:
            score = 0.0
    elif strategy.id == "midday_mean_revert":
        ts = df.index[-1]
        phase = _session_phase(ts)
        if phase == "midday":
            # Weak trend and extension from VWAP
            adx_ok = float(latest["adx14"]) < 15 if pd.notna(latest["adx14"]) else True
            ext = abs(float(latest["close"]) - float(latest["vwap"]))
            atr_val = float(latest["atr14"]) if pd.notna(latest["atr14"]) else float(df["high"].tail(14).sub(df["low"].tail(14)).median())
            if adx_ok and ext >= 0.6 * atr_val:
                score += 0.55
                # Bonus if momentum is slowing (use last 5-bar range contraction proxy)
                try:
                    last5 = df.tail(5)
                    rng = (last5["high"].max() - last5["low"].min())
                    if rng < 0.8 * atr_val:
                        score += 0.1
                except Exception:
                    pass
        else:
            score = 0.0
    else:
        # Assign a default minimal score; other strategies require more logic
        score = 0.0

    # Multiply by the win rate target if defined
    if strategy.win_rate_target is not None:
        score *= strategy.win_rate_target

    return score
