import random

import pytest

from src.plans.invariants import GeometryInvariantError, assert_invariants


def test_assert_invariants_long_valid_random_samples():
    random.seed(42)
    rr_floor = 1.2
    for _ in range(25):
        entry = 100 + random.random()
        stop = entry - random.uniform(0.5, 3.0)
        risk = entry - stop
        tp1 = entry + random.uniform(rr_floor * risk + 0.1, rr_floor * risk + 3.0)
        tp2 = tp1 + random.uniform(0.2, 2.0)
        targets = [round(tp1, 2), round(tp2, 2)]
        assert_invariants("long", entry, stop, targets, rr_min=rr_floor)


def test_assert_invariants_short_invalid_raises():
    entry = 105.0
    stop = 102.0  # invalid, should be above entry for short
    targets = [103.0, 101.5]
    with pytest.raises(GeometryInvariantError):
        assert_invariants("short", entry, stop, targets, rr_min=1.0)


def test_assert_invariants_rr_floor_enforced():
    entry = 100.0
    stop = 98.5
    targets = [101.0]
    # Reward 1.0, risk 1.5 -> rr < 1.2 threshold
    with pytest.raises(GeometryInvariantError):
        assert_invariants("long", entry, stop, targets, rr_min=1.2)
