import pytest

from src.follower import TradeFollower, TradeState


def test_trade_follower_flow_long():
    follower = TradeFollower(
        plan_id="plan-1",
        symbol="TSLA",
        direction="long",
        entry_price=200.0,
        stop_price=195.0,
        tp_price=205.0,
        atr_value=1.5,
    )
    follower.refresh_plan(entry=200.0, stop=195.0, target=205.0, atr=1.5)

    # Price below entry should not trigger any state change yet
    assert follower.update_from_price(199.0) is None

    entered = follower.update_from_price(200.2)
    assert entered is not None
    assert entered.state == TradeState.ENTERED

    warning = follower.update_from_price(195.3)
    assert warning is not None
    assert warning.event == "stop_warning"

    scaled = follower.update_from_price(205.1)
    assert scaled is not None
    assert scaled.state == TradeState.SCALED
    assert scaled.event == "tp_hit"
    assert follower.trailing_stop is not None

    trail_adjust = follower.update_from_price(208.0)
    assert trail_adjust is not None
    assert trail_adjust.state in {TradeState.TRAILING, TradeState.SCALED}
    trail_stop = follower.trailing_stop
    assert trail_stop is not None

    exited = follower.update_from_price(trail_stop - 0.05)
    assert exited is not None
    assert exited.state == TradeState.EXITED
    assert exited.exit_reason in {"trail", "stop"}
