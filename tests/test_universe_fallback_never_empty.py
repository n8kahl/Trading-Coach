from __future__ import annotations

import pytest

import src.services.universe as universe_service


@pytest.mark.asyncio()
async def test_universe_fallback_never_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_expand(universe: str | list[str], *, style: str, limit: int) -> list[str]:
        token = (universe or "").upper() if isinstance(universe, str) else ""
        if token == "LAST_SNAPSHOT":
            return []
        if token == "FT-TOPLIQUIDITY":
            return ["AAPL", "MSFT", "NVDA"]
        return []

    monkeypatch.setattr(universe_service.legacy_universe, "expand_universe", fake_expand)

    symbols = await universe_service.resolve_universe("LAST_SNAPSHOT", "swing")
    assert symbols
    assert symbols[:2] == ["AAPL", "MSFT"]
