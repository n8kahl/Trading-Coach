"""Multi-timeframe (MTF) bundle computation helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd

from ..calculations import adx, atr, ema, vwap

logger = logging.getLogger(__name__)

TF = Literal["5m", "15m", "60m", "D"]
VWAPRelation = Literal["above", "below", "near", "unknown"]

TF_WEIGHTS: Dict[TF, float] = {"5m": 0.05, "15m": 0.15, "60m": 0.3, "D": 0.5}
TF_ORDER: List[TF] = ["5m", "15m", "60m", "D"]


@dataclass
class TFState:
    tf: TF
    ema_up: bool
    ema_down: bool
    adx_slope: Optional[float]
    vwap_rel: VWAPRelation
    atr: Optional[float]


@dataclass
class MTFBundle:
    by_tf: Dict[TF, TFState]
    bias_htf: Literal["long", "short", "neutral"]
    agreement: float
    notes: List[str]


def _latest(series: pd.Series) -> Optional[float]:
    if series.empty:
        return None
    val = series.iloc[-1]
    if pd.isna(val):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _linear_slope(series: pd.Series, window: int = 10) -> Optional[float]:
    window = max(2, window)
    clean = series.dropna()
    if clean.empty:
        return None
    values = clean.iloc[-window:]
    if len(values) < 2:
        return None
    y = values.to_numpy(dtype=float)
    x = np.arange(len(y), dtype=float)
    try:
        slope, _ = np.polyfit(x, y, 1)
        return float(slope)
    except (TypeError, ValueError, np.linalg.LinAlgError):
        return None


def _vwap_relation(close: Optional[float], level: Optional[float], threshold: float) -> VWAPRelation:
    if close is None or level is None:
        return "unknown"
    diff = close - level
    if abs(diff) <= threshold:
        return "near"
    return "above" if diff > 0 else "below"


def _compute_tf_state(tf: TF, bars: pd.DataFrame, atr_5m: Optional[float], vwap_hint: Optional[float]) -> TFState:
    if bars is None or bars.empty:
        return TFState(tf=tf, ema_up=False, ema_down=False, adx_slope=None, vwap_rel="unknown", atr=None)

    frame = bars.sort_index()
    close = frame["close"]
    high = frame["high"]
    low = frame["low"]
    volume = frame["volume"] if "volume" in frame.columns else pd.Series([0.0] * len(frame), index=frame.index)

    ema9 = ema(close, 9) if len(close) >= 9 else pd.Series(dtype=float)
    ema20 = ema(close, 20) if len(close) >= 20 else pd.Series(dtype=float)
    ema50 = ema(close, 50) if len(close) >= 50 else pd.Series(dtype=float)

    ema9_latest = _latest(ema9)
    ema20_latest = _latest(ema20)
    ema50_latest = _latest(ema50)

    ema_up = (
        ema9_latest is not None
        and ema20_latest is not None
        and ema50_latest is not None
        and ema9_latest > ema20_latest > ema50_latest
    )
    ema_down = (
        ema9_latest is not None
        and ema20_latest is not None
        and ema50_latest is not None
        and ema9_latest < ema20_latest < ema50_latest
    )

    atr_series = atr(high, low, close, 14) if len(close) >= 14 else pd.Series(dtype=float)
    atr_latest = _latest(atr_series)

    adx_series = adx(high, low, close, 14) if len(close) >= 14 else pd.Series(dtype=float)
    adx_slope = _linear_slope(adx_series, window=10)

    if tf == "5m" and vwap_hint is not None:
        vwap_latest = float(vwap_hint)
    else:
        vwap_series = vwap(close, volume)
        vwap_latest = _latest(vwap_series)

    close_latest = _latest(close)

    if tf in {"60m", "D"}:
        threshold = (atr_5m or 0.0) * 0.25
    elif tf == "15m":
        threshold = (atr_latest or atr_5m or 0.0) * 0.15
    else:
        threshold = (atr_latest or atr_5m or 0.0) * 0.1
    if threshold <= 0:
        threshold = max((close_latest or 0.0) * 0.0005, 0.02)

    vwap_rel = _vwap_relation(close_latest, vwap_latest, threshold)

    return TFState(
        tf=tf,
        ema_up=ema_up,
        ema_down=ema_down,
        adx_slope=adx_slope,
        vwap_rel=vwap_rel,
        atr=atr_latest,
    )


def _trend_descriptor(state: TFState) -> tuple[str, str]:
    if state.ema_up:
        return "up", "↑"
    if state.ema_down:
        return "down", "↓"
    return "flat", "≈"


def _bias_from_score(score: float) -> Literal["long", "short", "neutral"]:
    if score <= -0.2:
        return "short"
    if score >= 0.2:
        return "long"
    return "neutral"


def compute_mtf_bundle(
    symbol: str,
    bars_5m: pd.DataFrame,
    bars_15m: pd.DataFrame,
    bars_60m: pd.DataFrame,
    bars_d: pd.DataFrame,
    vwap_5m: Optional[float],
) -> Optional[MTFBundle]:
    """Compute a multi-timeframe bundle for strategy gating.

    Returns:
        MTFBundle when sufficient data is present, otherwise None.
    """
    try:
        if bars_5m is None or bars_5m.empty:
            return None

        atr5_series = atr(bars_5m["high"], bars_5m["low"], bars_5m["close"], 14) if len(bars_5m) >= 14 else pd.Series(dtype=float)
        atr_5m_latest = _latest(atr5_series)

        tf_frames: Dict[TF, pd.DataFrame] = {
            "5m": bars_5m,
            "15m": bars_15m if bars_15m is not None else pd.DataFrame(),
            "60m": bars_60m if bars_60m is not None else pd.DataFrame(),
            "D": bars_d if bars_d is not None else pd.DataFrame(),
        }

        by_tf: Dict[TF, TFState] = {}
        for tf in TF_ORDER:
            frame = tf_frames.get(tf)
            if frame is None:
                frame = pd.DataFrame()
            state = _compute_tf_state(tf, frame, atr_5m=atr_5m_latest, vwap_hint=vwap_5m if tf == "5m" else None)
            by_tf[tf] = state

        score = 0.0
        long_counts = 0
        short_counts = 0
        for tf, state in by_tf.items():
            weight = TF_WEIGHTS.get(tf, 0.0)
            if state.ema_up:
                score += weight
                long_counts += 1
            elif state.ema_down:
                score -= weight
                short_counts += 1
        bias = _bias_from_score(score)

        if bias == "long" and (long_counts or short_counts):
            agreement = long_counts / max(long_counts + short_counts, 1)
        elif bias == "short" and (long_counts or short_counts):
            agreement = short_counts / max(long_counts + short_counts, 1)
        else:
            agreement = 0.5

        trend_tokens: List[str] = []
        for tf in ["D", "60m", "15m"]:
            state = by_tf.get(tf)
            if not state:
                continue
            trend_state, glyph = _trend_descriptor(state)
            if trend_state == "flat":
                token = f"{tf}≈flat"
            else:
                token = f"{tf}{glyph}"
            trend_tokens.append(token)
        if not trend_tokens:
            for tf in TF_ORDER:
                state = by_tf.get(tf)
                if not state:
                    continue
                _, glyph = _trend_descriptor(state)
                trend_tokens.append(f"{tf}{glyph}")

        vwap_state = by_tf["5m"].vwap_rel if "5m" in by_tf else "unknown"
        if vwap_state == "above":
            vwap_note = "VWAP>"
        elif vwap_state == "below":
            vwap_note = "VWAP<"
        elif vwap_state == "near":
            vwap_note = "VWAP≈"
        else:
            vwap_note = "VWAP?"

        adx_notes: List[str] = []
        for tf in ("5m", "15m"):
            state = by_tf.get(tf)
            if not state or state.adx_slope is None:
                continue
            if state.adx_slope > 0.02:
                adx_notes.append(f"{tf} ADX↗")
            elif state.adx_slope < -0.02:
                adx_notes.append(f"{tf} ADX↘")

        notes = [", ".join(trend_tokens), vwap_note, *adx_notes]

        return MTFBundle(
            by_tf=by_tf,
            bias_htf=bias,
            agreement=agreement,
            notes=[note for note in notes if note],
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("compute_mtf_bundle_failed", exc_info=True, extra={"symbol": symbol, "detail": str(exc)})
        return None


__all__ = ["TF", "TFState", "MTFBundle", "compute_mtf_bundle"]
