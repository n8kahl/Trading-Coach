from datetime import datetime, timezone

from src.lib.market_clock import NY, get_market_state, most_recent_regular_close


def test_market_state_regular_session_boundaries() -> None:
    pre_open = datetime(2024, 6, 10, 13, 29, 59, tzinfo=timezone.utc)  # 09:29:59 ET
    at_open = datetime(2024, 6, 10, 13, 30, 0, tzinfo=timezone.utc)  # 09:30:00 ET
    pre_close = datetime(2024, 6, 10, 19, 59, 59, tzinfo=timezone.utc)  # 15:59:59 ET
    at_close = datetime(2024, 6, 10, 20, 0, 0, tzinfo=timezone.utc)  # 16:00:00 ET

    assert get_market_state(pre_open) == "closed"
    assert get_market_state(at_open) == "open"
    assert get_market_state(pre_close) == "open"
    assert get_market_state(at_close) == "closed"


def test_market_state_weekends_closed() -> None:
    saturday = datetime(2024, 6, 8, 16, 0, 0, tzinfo=timezone.utc)
    sunday = datetime(2024, 6, 9, 16, 0, 0, tzinfo=timezone.utc)
    assert get_market_state(saturday) == "closed"
    assert get_market_state(sunday) == "closed"


def test_most_recent_regular_close_weekend() -> None:
    saturday = datetime(2024, 6, 8, 12, 0, 0, tzinfo=timezone.utc)
    close_dt = most_recent_regular_close(saturday)
    expected = datetime(2024, 6, 7, 16, 0, 0, tzinfo=NY)
    assert close_dt == expected


def test_most_recent_regular_close_holiday() -> None:
    def is_holiday(dt: datetime) -> bool:
        return dt.date().month == 7 and dt.date().day == 4

    fourth_july = datetime(2024, 7, 4, 15, 0, 0, tzinfo=timezone.utc)  # holiday Thursday
    assert get_market_state(fourth_july, is_holiday=is_holiday) == "closed"

    friday_morning = datetime(2024, 7, 5, 12, 0, 0, tzinfo=timezone.utc)
    close_dt = most_recent_regular_close(friday_morning, is_holiday=is_holiday)
    expected = datetime(2024, 7, 3, 16, 0, 0, tzinfo=NY)
    assert close_dt == expected
