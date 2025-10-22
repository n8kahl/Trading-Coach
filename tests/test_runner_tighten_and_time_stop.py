import pytest

from src.plans.runner import compute_runner


def test_runner_policy_includes_tighten_and_time_stop():
    runner = compute_runner(
        entry=100.0,
        tps=[100.8, 101.4, 102.0],
        style="intraday",
        em_points=1.5,
        atr=0.7,
        profile_nodes=["VWAP"],
    )

    assert runner["fraction"] > 0
    assert runner["tighten_threshold"] == pytest.approx(0.60, rel=1e-2)
    assert runner["tighten_trail_mult"] < runner["atr_trail_mult"]
    assert "time_stop" in runner
    assert runner["time_stop"]["remaining_atr_fraction"] == 0.5
