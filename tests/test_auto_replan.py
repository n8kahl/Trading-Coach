import pytest

from src.agent_server import _auto_replan


@pytest.mark.asyncio
async def test_auto_replan_publishes_rearm_delta(monkeypatch):
    published: list[tuple[str, dict]] = []

    async def fake_publish(symbol: str, event: dict) -> None:
        published.append((symbol, event))

    monkeypatch.setattr("src.agent_server._publish_stream_event", fake_publish)

    class DummySettings:
        self_base_url = "https://example.test"
        backend_api_key = None

    monkeypatch.setattr("src.agent_server.get_settings", lambda: DummySettings())

    class DummyResponse:
        def __init__(self, payload: dict) -> None:
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return self._payload

    class DummyAsyncClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self) -> "DummyAsyncClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> bool:
            return False

        async def post(self, url: str, json=None, headers=None) -> DummyResponse:
            assert url.endswith("/gpt/plan")
            return DummyResponse({"plan_id": "TSLA-NEXT", "version": 2, "style": "intraday"})

    monkeypatch.setattr("src.agent_server.httpx.AsyncClient", DummyAsyncClient)

    await _auto_replan("TSLA", "intraday", "plan123", "stop")

    rearm_events = [
        event
        for _, event in published
        if isinstance(event, dict)
        and event.get("t") == "plan_delta"
        and event.get("changes", {}).get("coach_event") == "rearm"
    ]

    assert rearm_events, "auto replan should publish a coach event for re-arming"
