import pytest

from src.config import STYLE_GATES
from src.core.selection_gates import apply_lenient_gate, within_hard_caps


def test_gate_allows_preferred_zone() -> None:
    candidate = {"entry_distance_atr": 0.24, "bars_to_trigger": 1.0, "entry_distance_pct": 0.0007}
    ok, penalty = apply_lenient_gate(dict(candidate), "scalp", STYLE_GATES)
    assert ok is True
    assert penalty == pytest.approx(0.0)


def test_gate_penalizes_soft_zone_without_reject() -> None:
    candidate = {"entry_distance_atr": 0.32, "bars_to_trigger": 1.6, "entry_distance_pct": 0.00105}
    ok, penalty = apply_lenient_gate(dict(candidate), "scalp", STYLE_GATES)
    assert ok is True
    assert 0 < penalty < 0.25


def test_gate_rejects_beyond_hard_caps() -> None:
    candidate = {"entry_distance_atr": 0.9, "bars_to_trigger": 3.5, "entry_distance_pct": 0.0018}
    ok, penalty = apply_lenient_gate(candidate, "scalp", STYLE_GATES)
    assert ok is False
    assert candidate["_gate_reject_reason"] == "beyond_hard_caps"


def test_volatility_relaxation_extends_caps() -> None:
    baseline = {"entry_distance_atr": 0.36, "bars_to_trigger": 2.0, "entry_distance_pct": 0.00115}
    assert within_hard_caps(baseline, STYLE_GATES["scalp"]) is False

    relaxed = dict(baseline)
    relaxed["vol_proxy"] = 30.0
    ok, penalty = apply_lenient_gate(relaxed, "scalp", STYLE_GATES)
    assert ok is True
    assert penalty > 0
