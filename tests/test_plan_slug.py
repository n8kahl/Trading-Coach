from src.agent_server import _parse_plan_slug


def test_parse_standard_slug():
    result = _parse_plan_slug("spy-intraday-long-2025-10")
    assert result == {"symbol": "SPY", "style": "intraday", "direction": "long"}


def test_parse_offline_slug_skips_prefix():
    result = _parse_plan_slug("offline-spy-swing-20251010")
    assert result == {"symbol": "SPY", "style": "swing", "direction": "unknown"}
