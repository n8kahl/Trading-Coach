from pathlib import Path

import pytest

from src.engine.calibration import CalibrationStore
from src.evaluator.runner import PlanEvaluator


def _fixture_path(filename: str) -> Path:
    return Path(__file__).parent / "fixtures" / "plans" / filename


@pytest.mark.parametrize(
    "fixture_name, style, cohort",
    [
        ("open_session.json", "intraday", "open_session"),
        ("closed_session.json", "swing", "closed_session"),
    ],
)
def test_prob_touch_reliability_within_tolerance(fixture_name: str, style: str, cohort: str) -> None:
    evaluator = PlanEvaluator(bin_size=0.2)
    summary = evaluator.evaluate_fixture(_fixture_path(fixture_name))

    assert summary.total > 0
    assert summary.brier_score < 0.25
    assert summary.ece <= 0.07 + 1e-6

    for bin_info in summary.calibration_bins:
        if bin_info.count == 0:
            continue
        diff = abs(bin_info.avg_prediction - bin_info.observed)
        assert diff <= 0.07, f"{fixture_name} bin diff {diff:.4f} exceeds tolerance"

    store = CalibrationStore()
    store.register(summary.to_calibration_table())

    raw_prob = 0.55 if style == "intraday" else 0.52
    calibrated, meta = store.calibrate(style, raw_prob, cohort=cohort)

    assert meta is not None
    assert abs(calibrated - raw_prob) <= 0.07


def test_calibration_store_roundtrip(tmp_path: Path) -> None:
    evaluator = PlanEvaluator(bin_size=0.2)
    open_summary = evaluator.evaluate_fixture(_fixture_path("open_session.json"))

    store = CalibrationStore()
    store.register(open_summary.to_calibration_table())

    out_path = tmp_path / "calibration.json"
    store.save(out_path)

    loaded = CalibrationStore.load(out_path)
    calibrated, meta = loaded.calibrate("intraday", 0.6, cohort="open_session")

    assert meta is not None
    assert 0.0 <= calibrated <= 1.0
    assert abs(calibrated - 0.6) <= 0.07

