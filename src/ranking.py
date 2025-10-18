"""Deterministic ranking helpers for scan results."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Sequence

Style = Literal["scalp", "intraday", "swing", "leaps"]

logger = logging.getLogger(__name__)

ACTIONABILITY_GATE = 0.25

WEIGHTS: Dict[Style, Dict[str, float]] = {
    "scalp": {
        "entry_quality": 0.24,
        "rr1": 0.16,
        "rr2": 0.11,
        "liquidity": 0.1,
        "confluence": 0.1,
        "momentum": 0.08,
        "vol_constraint": 0.05,
        "htf_structure": 0.05,
        "confluence_htf": 0.04,
        "vol_regime": 0.03,
        "momentum_htf": 0.02,
        "context": 0.02,
        "weekly_structure": 0.02,
        "daily_confluence": 0.02,
        "option_efficiency": 0.03,
        "rr_multi": 0.02,
        "macrofit": 0.03,
        "confidence": 0.03,
        "pen_event": 0.35,
        "pen_dq": 0.2,
        "pen_spread": 0.2,
        "pen_chop": 0.25,
        "pen_cluster": 0.1,
    },
    "intraday": {
        "entry_quality": 0.2,
        "rr1": 0.14,
        "rr2": 0.1,
        "liquidity": 0.11,
        "confluence": 0.11,
        "momentum": 0.07,
        "vol_constraint": 0.05,
        "htf_structure": 0.06,
        "confluence_htf": 0.05,
        "vol_regime": 0.04,
        "momentum_htf": 0.03,
        "context": 0.03,
        "weekly_structure": 0.02,
        "daily_confluence": 0.03,
        "option_efficiency": 0.04,
        "rr_multi": 0.03,
        "macrofit": 0.04,
        "confidence": 0.03,
        "pen_event": 0.3,
        "pen_dq": 0.18,
        "pen_spread": 0.18,
        "pen_chop": 0.22,
        "pen_cluster": 0.12,
    },
    "swing": {
        "entry_quality": 0.14,
        "rr1": 0.12,
        "rr2": 0.1,
        "liquidity": 0.09,
        "confluence": 0.1,
        "momentum": 0.06,
        "vol_constraint": 0.05,
        "htf_structure": 0.09,
        "confluence_htf": 0.08,
        "vol_regime": 0.06,
        "momentum_htf": 0.04,
        "context": 0.05,
        "weekly_structure": 0.04,
        "daily_confluence": 0.04,
        "option_efficiency": 0.03,
        "rr_multi": 0.04,
        "macrofit": 0.06,
        "confidence": 0.04,
        "pen_event": 0.28,
        "pen_dq": 0.16,
        "pen_spread": 0.16,
        "pen_chop": 0.18,
        "pen_cluster": 0.12,
    },
    "leaps": {
        "entry_quality": 0.12,
        "rr1": 0.09,
        "rr2": 0.08,
        "liquidity": 0.08,
        "confluence": 0.08,
        "momentum": 0.05,
        "vol_constraint": 0.06,
        "htf_structure": 0.1,
        "confluence_htf": 0.09,
        "vol_regime": 0.08,
        "momentum_htf": 0.05,
        "context": 0.06,
        "weekly_structure": 0.06,
        "daily_confluence": 0.05,
        "option_efficiency": 0.05,
        "rr_multi": 0.05,
        "macrofit": 0.08,
        "confidence": 0.05,
        "pen_event": 0.26,
        "pen_dq": 0.14,
        "pen_spread": 0.18,
        "pen_chop": 0.16,
        "pen_cluster": 0.1,
    },
}

MIN_QUALITY: Dict[Style, float] = {
    "scalp": 0.45,
    "intraday": 0.4,
    "swing": 0.35,
    "leaps": 0.32,
}

SECTOR_CAP = 0.20


@dataclass(slots=True)
class Features:
    symbol: str
    style: Style
    sector: str | None
    entry_quality: float
    rr1: float
    rr2: float
    liquidity: float
    confluence: float
    momentum: float
    vol_constraint: float
    htf_structure: float
    confluence_htf: float
    vol_regime: float
    momentum_htf: float
    context: float
    actionability: float
    weekly_structure: float
    daily_confluence: float
    option_efficiency: float
    rr_multi: float
    macrofit: float
    confidence: float
    pen_event: float
    pen_dq: float
    pen_spread: float
    pen_chop: float
    pen_cluster: float

    def quality(self) -> float:
        baseline = (
            self.entry_quality
            + self.confluence
            + self.liquidity
            + self.context
            + self.actionability
        ) / 5.0
        return _clamp(baseline)


@dataclass(slots=True)
class Scored:
    symbol: str
    score: float
    confidence: float
    sector: str | None
    features: Features


def rank(features: Sequence[Features], style: Style) -> List[Scored]:
    """Rank a list of feature bundles for the requested style."""
    weights = WEIGHTS.get(style, WEIGHTS["intraday"])
    quality_min = MIN_QUALITY.get(style, 0.4)
    scored: List[Scored] = []

    for bundle in features:
        if bundle.quality() < quality_min:
            continue
        if style in {"scalp", "intraday"}:
            if _clamp(bundle.actionability) < ACTIONABILITY_GATE:
                logger.debug(
                    "rank_actionability_gated",
                    extra={
                        "symbol": bundle.symbol,
                        "style": style,
                        "actionability": float(_clamp(bundle.actionability)),
                    },
                )
                continue
            trend_score, rr_score, base_score = _trend_actionability_score(bundle, weights)
            score = base_score
            logger.debug(
                "rank_candidate_score",
                extra={
                    "symbol": bundle.symbol,
                    "style": style,
                    "score": round(score, 4),
                    "trend_component": round(trend_score, 4),
                    "rr_component": round(rr_score, 4),
                    "actionability": round(_clamp(bundle.actionability), 4),
                },
            )
        else:
            score = _weighted_score(bundle, weights)
        if score <= 0:
            continue
        scored.append(
            Scored(
                symbol=bundle.symbol,
                score=score,
                confidence=_clamp(bundle.confidence),
                sector=bundle.sector,
                features=bundle,
            )
        )

    scored.sort(
        key=lambda item: (
            -item.score,
            -_round_for_sort(item.confidence),
            item.symbol,
        )
    )
    return scored


def diversify(scored: Iterable[Scored], limit: int = 100) -> List[Scored]:
    """Clamp ranked results to the requested limit while enforcing sector caps."""
    limit = max(1, min(limit, 250))
    max_per_sector = max(1, int(math.floor(limit * SECTOR_CAP)))
    counts: Dict[str, int] = {}
    selection: List[Scored] = []
    for item in scored:
        sector_key = (item.sector or "UNKNOWN").upper()
        current = counts.get(sector_key, 0)
        if current >= max_per_sector:
            continue
        counts[sector_key] = current + 1
        selection.append(item)
        if len(selection) >= limit:
            break
    return selection


def _trend_actionability_score(features: Features, weights: Dict[str, float]) -> tuple[float, float, float]:
    trend_components = (
        features.momentum,
        features.confluence,
        features.momentum_htf,
        features.htf_structure,
        features.confluence_htf,
    )
    trend_score = _mean(trend_components)
    rr_components = (features.rr1, features.rr2, features.rr_multi)
    rr_score = _mean(rr_components)
    actionability_component = _clamp(features.actionability)
    base = (
        0.4 * _clamp(trend_score)
        + 0.3 * actionability_component
        + 0.3 * _clamp(rr_score)
    )
    penalty = 0.0
    for key in ("pen_event", "pen_dq", "pen_spread", "pen_chop", "pen_cluster"):
        penalty += weights.get(key, 0.0) * _clamp(getattr(features, key, 0.0))
    final_score = max(base - penalty, 0.0)
    return _clamp(trend_score), _clamp(rr_score), final_score


def _weighted_score(features: Features, weights: Dict[str, float]) -> float:
    positive_keys = (
        "entry_quality",
        "rr1",
        "rr2",
        "liquidity",
        "confluence",
        "momentum",
        "vol_constraint",
        "htf_structure",
        "confluence_htf",
        "vol_regime",
        "momentum_htf",
        "context",
        "actionability",
        "weekly_structure",
        "daily_confluence",
        "option_efficiency",
        "rr_multi",
        "macrofit",
        "confidence",
    )
    penalty_keys = (
        "pen_event",
        "pen_dq",
        "pen_spread",
        "pen_chop",
        "pen_cluster",
    )

    score = 0.0
    for key in positive_keys:
        weight = weights.get(key)
        if not weight:
            continue
        value = getattr(features, key, 0.0)
        score += weight * _clamp(value)

    for penalty_key in penalty_keys:
        weight = weights.get(penalty_key)
        if not weight:
            continue
        value = getattr(features, penalty_key, 0.0)
        score -= weight * _clamp(value)

    return max(score, 0.0)


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    if math.isnan(value):
        return low
    return max(low, min(high, value))


def _mean(values: Sequence[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return 0.0
    return sum(finite) / len(finite)


def _round_for_sort(value: float) -> float:
    return round(value, 6)


__all__ = ["SECTOR_CAP", "Features", "Scored", "Style", "diversify", "rank", "WEIGHTS", "MIN_QUALITY", "ACTIONABILITY_GATE"]
