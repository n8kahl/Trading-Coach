from src.agent_server import _ensure_monotonic_targets


def test_ensure_monotonic_targets_long():
    targets = _ensure_monotonic_targets("long", 100.0, [100.2, 100.2, 100.19, 100.5])
    assert targets[0] > 100.0
    assert targets == sorted(targets)
    for prev, curr in zip([100.0] + targets[:-1], targets):
        assert curr > prev


def test_ensure_monotonic_targets_short():
    targets = _ensure_monotonic_targets("short", 100.0, [99.8, 99.8, 99.9, 99.1])
    assert all(curr < prev for prev, curr in zip([100.0] + targets[:-1], targets))
