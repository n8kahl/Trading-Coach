"""Strategy selection orchestrator using rule evaluations and MTF context."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ..features.htf_levels import HTFLevels
from ..features.mtf import MTFBundle
from .rules import RuleContext, RuleResult, candidate_rules


@dataclass
class MatchedRule:
    id: str
    score: float
    reasons: List[str]


@dataclass
class StrategyProfile:
    name: str
    badges: List[str]
    waiting_for: str
    mtf: Optional[Dict[str, object]]
    matched_rules: List[MatchedRule]
    mtf_confluence: List[str]


def mtf_amplifier(direction: str, mtf: Optional[MTFBundle]) -> float:
    if not mtf:
        return 1.0
    amp = 1.0
    bias = mtf.bias_htf
    if bias == "long" and direction == "long":
        amp += 0.08
    if bias == "short" and direction == "short":
        amp += 0.08
    if (bias == "long" and direction == "short") or (bias == "short" and direction == "long"):
        amp -= 0.10
    agreement = getattr(mtf, "agreement", 0.5) or 0.5
    amp += (agreement - 0.5) * 0.2
    return max(0.8, min(1.15, amp))


def _update_agreement(direction: str, bundle: Optional[MTFBundle]) -> None:
    if not bundle or not bundle.by_tf:
        return
    total = 0
    aligned = 0
    for state in bundle.by_tf.values():
        if state is None:
            continue
        total += 1
        if direction == "long" and state.ema_up:
            aligned += 1
        elif direction == "short" and state.ema_down:
            aligned += 1
    if total > 0:
        bundle.agreement = aligned / total
    else:
        bundle.agreement = 0.5


def _baseline_rule() -> RuleResult:
    return RuleResult(
        id="baseline_auto",
        name="Baseline Geometry",
        base_score=0.5,
        reasons=["Default structural geometry"],
        waiting_for="Respect planned entry/volume confirmation",
        badges=["Baseline"],
    )


def _mtf_confluence_notes(bundle: Optional[MTFBundle]) -> List[str]:
    notes: List[str] = []
    if not bundle or not bundle.by_tf:
        return notes
    order = ["D", "60m", "15m", "5m"]
    for tf in order:
        state = bundle.by_tf.get(tf)
        if not state:
            continue
        if state.ema_up:
            notes.append(f"{tf} trend ↑")
        elif state.ema_down:
            notes.append(f"{tf} trend ↓")
        else:
            notes.append(f"{tf} trend ≈")
        if len(notes) >= 4:
            break
    state_5m = bundle.by_tf.get("5m")
    if state_5m and state_5m.vwap_rel in {"above", "below"}:
        notes.append(f"VWAP {state_5m.vwap_rel}")
    if bundle.bias_htf and bundle.bias_htf != "neutral":
        notes.append(f"HTF bias {bundle.bias_htf}")
    agreement_tag = f"MTF agreement {round(bundle.agreement, 2)}"
    notes.append(agreement_tag)
    return notes[:6]


def infer_strategy(direction: str, ctx: Dict[str, object]) -> Tuple[str, StrategyProfile]:
    """Infer strategy metadata and best-fitting rule for the current context."""

    bundle = ctx.get("mtf")
    if isinstance(bundle, MTFBundle):
        _update_agreement(direction, bundle)
    else:
        bundle = None
    mtf_confluence_notes = _mtf_confluence_notes(bundle)
    htf_levels = ctx.get("htf_levels")
    if not isinstance(htf_levels, HTFLevels):
        htf_levels = None

    rule_context = RuleContext(
        symbol=str(ctx.get("symbol") or ""),
        direction=direction,
        timestamp=ctx.get("timestamp"),
        mtf=bundle,
        htf_levels=htf_levels,
        price=ctx.get("price"),
        vwap=ctx.get("vwap"),
        opening_range_high=ctx.get("opening_range_high"),
        opening_range_low=ctx.get("opening_range_low"),
        bars_5m=ctx.get("bars_5m") if isinstance(ctx.get("bars_5m"), pd.DataFrame) else None,
        bars_15m=ctx.get("bars_15m") if isinstance(ctx.get("bars_15m"), pd.DataFrame) else None,
        bars_60m=ctx.get("bars_60m") if isinstance(ctx.get("bars_60m"), pd.DataFrame) else None,
    )

    evaluations: List[RuleResult] = [_baseline_rule()]
    for evaluator in candidate_rules(direction):
        try:
            result = evaluator(rule_context)
        except Exception:  # pragma: no cover - defensive
            result = None
        if result:
            evaluations.append(result)

    amplifier = mtf_amplifier(direction, bundle)

    matched: List[MatchedRule] = []
    scored_rules: Dict[str, Tuple[RuleResult, float]] = {}
    for result in evaluations:
        score = round(result.base_score * amplifier, 4)
        matched.append(MatchedRule(id=result.id, score=score, reasons=list(result.reasons)))
        scored_rules[result.id] = (result, score)

    matched.sort(key=lambda item: item.score, reverse=True)

    baseline_score = scored_rules["baseline_auto"][1]
    selected_id = "baseline_auto"
    selected_result = scored_rules["baseline_auto"][0]
    best_score = baseline_score
    mtf_supports_direction = True
    if bundle:
        bias = bundle.bias_htf
        if bias not in {direction, "neutral"}:
            mtf_supports_direction = False

    for rule_id, (result, score) in scored_rules.items():
        if rule_id == "baseline_auto":
            continue
        if not mtf_supports_direction:
            continue
        if score >= baseline_score + 0.03 and score >= best_score:
            selected_id = rule_id
            selected_result = result
            best_score = score

    mtf_payload: Optional[Dict[str, object]] = None
    if bundle:
        mtf_payload = {
            "bias": bundle.bias_htf,
            "agreement": round(bundle.agreement, 2),
            "notes": list(bundle.notes),
        }

    profile = StrategyProfile(
        name=selected_result.name,
        badges=list(selected_result.badges),
        waiting_for=selected_result.waiting_for,
        mtf=mtf_payload,
        matched_rules=matched,
        mtf_confluence=mtf_confluence_notes,
    )

    return selected_id, profile


__all__ = ["StrategyProfile", "MatchedRule", "infer_strategy", "mtf_amplifier"]
