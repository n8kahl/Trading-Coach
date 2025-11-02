from __future__ import annotations

from fastapi.testclient import TestClient

from src.agent_server import app


def test_request_id_header_echo() -> None:
    with TestClient(app) as client:
        auto = client.get("/healthz")
        assert auto.status_code == 200
        generated = auto.headers.get("X-Request-ID")
        assert generated is not None and generated != ""

        custom_id = "test-req-123"
        echoed = client.get("/healthz", headers={"X-Request-ID": custom_id})
        assert echoed.status_code == 200
        assert echoed.headers.get("X-Request-ID") == custom_id
