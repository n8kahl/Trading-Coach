# Project Status – Trading Coach GPT Backend

_Last updated: 2025-10-08_

## Summary

- Backend has been refactored to serve a GPT Action-first API (`/gpt/*`) with no web build.
- Tradier sandbox integration in place for option-chain pricing (`select_tradier_contract`).
- `/gpt/scan` now returns enriched payloads: `direction`, `levels` (entry/stop/target), and `chart_url`.
- Hosted chart endpoint (`GET /chart/{symbol}`) renders a lightweight TradingView view with overlays.
- Nixpacks build phase overridden with a no-op to skip the legacy npm step.

## Current Deployment Checklist

1. Ensure Railway service uses commit `faa2f1b` (or newer) on `main`.
2. Environment variables required:
   - `TRADIER_SANDBOX_TOKEN` (currently using provided sandbox token).
   - `TRADIER_SANDBOX_ACCOUNT` (optional for now; reserved for future account-level calls).
   - Optional: `BACKEND_API_KEY` if secure access is needed.
3. Redeploy workflow:
   - Clear build cache.
   - Redeploy; verify plan banner shows build phase `echo skipping frontend build`.
4. Smoke test:
   - `POST /gpt/scan` (expect `chart_url` and `levels` in response).
   - Open the returned `chart_url` to confirm chart rendering.

## Outstanding Work / Next Steps

- Replace synthetic OHLCV (`_synth_ohlcv`) with Polygon aggregates (requires `POLYGON_API_KEY` and data fetch).
- Persist watchlists/notes/trades in a datastore instead of in-memory maps.
- Expand Tradier integration to align contract filters with strategy-specific rules (e.g., DTE targets).
- Update GPT Action schema in Assistant Builder with the latest `ScanResponse` structure (including `chart_url` and `levels`).
- Optional: add `/gpt/trades/{trade_id}/close` endpoint back if automated close logic is required.

## Useful References

- Backend entry point: `src/agent_server.py`
- Tradier client helpers: `src/tradier.py`
- Contract filtering heuristics: `src/contract_selector.py`
- OpenAPI schema reference: see latest schema provided to GPT Actions (not auto-generated yet).
- Deployment config: `nixpacks.toml` (Python-only, no build).

## Notes for Next Engineer

- Tests (`pytest`) cover indicator utilities only; no integration tests yet.
- If you reintroduce frontend assets, remember to adjust the Nixpacks build phase.
- When updating the API, keep the GPT Action instructions in sync (especially the chart CTA: “View interactive chart”).
