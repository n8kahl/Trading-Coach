import datetime as dt
from zoneinfo import ZoneInfo

import pytest

from src.app.services import session_state


class DummySnapshot:
    def __init__(self, status: str, now: dt.datetime):
        self.status = status
        self.session = "RTH"
        self.now_et = now
        self.next_open_et = now + dt.timedelta(hours=18)
        self.next_close_et = now + dt.timedelta(hours=8)


def _reset_clock(monkeypatch, status: str = "closed", now: dt.datetime | None = None) -> None:
    now = now or dt.datetime.now(ZoneInfo("America/New_York"))
    snapshot = DummySnapshot(status=status, now=now)

    monkeypatch.setattr(session_state, "_CLOCK", session_state.MarketClock())
    monkeypatch.setattr(session_state._CLOCK, "snapshot", lambda: snapshot)
    monkeypatch.setattr(session_state._CLOCK, "last_rth_close", lambda at=None: now - dt.timedelta(days=1))
    monkeypatch.setattr(session_state._CLOCK, "next_open_close", lambda at=None: (now + dt.timedelta(days=1), now + dt.timedelta(days=1, hours=7)))


@pytest.mark.parametrize("market_flag, expected_status", [("open", "open"), ("closed", "closed")])
def test_session_now_polygon_status_overrides_clock(monkeypatch, market_flag, expected_status):
    reference_now = dt.datetime(2025, 10, 14, 15, 0, tzinfo=ZoneInfo("America/New_York"))
    _reset_clock(monkeypatch, status="closed", now=reference_now)
    session_state._STATUS_CACHE = None

    polygon_payload = {
        "serverTime": "2025-10-14T19:00:00Z",
        "market": market_flag,
        "exchanges": {"stocks": "open" if market_flag == "open" else "closed"},
        "marketHours": {
            "stocks": {
                "isOpen": market_flag == "open",
                "session": "trading" if market_flag == "open" else "extended",
                "open": "2025-10-14T13:30:00Z",
                "close": "2025-10-14T20:00:00Z",
            }
        },
        "previous": {
            "stocks": {
                "close": "2025-10-11T20:00:00Z",
            }
        },
    }

    monkeypatch.setattr(session_state, "_polygon_market_status", lambda: polygon_payload)

    state = session_state.session_now()
    assert state.status == expected_status
    if expected_status == "open":
        assert state.banner == "Market open"
    else:
        assert (
            state.banner.startswith("Market closed")
            or "After hours" in state.banner
            or "Premarket" in state.banner
        )
        assert state.as_of.endswith("-04:00")
