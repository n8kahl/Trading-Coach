import asyncio
import json

import pytest

from src.realtime_bars import PolygonRealtimeBarStreamer


@pytest.mark.asyncio
async def test_realtime_bar_streamer_parses_bar_event(monkeypatch):
    events = []

    async def fake_on_event(symbol, payload):  # noqa: ANN001
        events.append((symbol, payload))

    streamer = PolygonRealtimeBarStreamer("fake-key", ["SPX"], on_event=fake_on_event)

    message = json.dumps(
        [
            {
                "ev": "XA",
                "sym": "I:SPX",
                "o": 5840.0,
                "h": 5842.0,
                "l": 5838.5,
                "c": 5841.2,
                "v": 123.0,
                "t": 1_729_108_800_000,
            }
        ]
    )

    await streamer._handle_message(message)  # type: ignore[attr-defined]

    assert events, "Expected a bar event to be emitted"
    symbol, payload = events[0]
    assert symbol == "SPX"
    assert payload["t"] == "bar"
    assert payload["close"] == 5841.2
    assert payload["volume"] == 123.0
    assert payload["ts"] == 1_729_108_800


@pytest.mark.asyncio
async def test_realtime_bar_streamer_ignores_status(monkeypatch):
    events = []

    async def fake_on_event(symbol, payload):  # noqa: ANN001
        events.append((symbol, payload))

    streamer = PolygonRealtimeBarStreamer("fake-key", ["SPX"], on_event=fake_on_event)
    await streamer._handle_message(json.dumps({"ev": "status", "message": "connected"}))  # type: ignore[attr-defined]
    assert not events
