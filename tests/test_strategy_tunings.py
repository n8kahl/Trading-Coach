import importlib


def test_index_preserved():
    lib = importlib.import_module("src.strategy_library")
    assert lib.normalize_style_input("index") == "index"


def test_alias_bridge():
    cat = importlib.import_module("src.strategies.catalog")
    profile = cat.get_strategy_profile("break_and_retest")
    assert "Range Break" in profile["name"]
    profile_scalp = cat.get_strategy_profile("break_and_retest_scalp")
    profile_swing = cat.get_strategy_profile("break_and_retest_swing")
    assert "Range Break" in profile_scalp["name"]
    assert "Range Break" in profile_swing["name"]


def test_win_rate_targets_present():
    lib = importlib.import_module("src.strategy_library")
    strategies = {s.id: s for s in lib.load_strategies()}
    assert strategies["break_and_retest"].win_rate_target and strategies["break_and_retest"].win_rate_target >= 0.56
    assert strategies["momentum_play"].win_rate_target and strategies["momentum_play"].win_rate_target >= 0.54
    assert "break_and_retest_scalp" in strategies
    assert "break_and_retest_swing" in strategies


def test_intraday_rr_floor_bumped():
    execp = importlib.import_module("src.app.engine.execution_profiles")
    intraday = execp.STYLE_PROFILES["intraday"]
    assert tuple(intraday.tp_rr_floor) == (1.4, 1.8, 2.3)
