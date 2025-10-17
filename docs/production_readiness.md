# Production Readiness – Current Build

Status: Production ready (current GitHub build)

- Branch: `main`
- Commit: `90fe80d`
- Timestamp (UTC): 2025-10-17 04:49
- Production host: `https://trading-coach-production.up.railway.app`

Scope considered production-ready
- Core APIs: `/gpt/scan`, `/gpt/context/{symbol}`, `/gpt/multi-context`, `/gpt/contracts`, `/gpt/chart-url`, `/api/v1/gpt/chart-layers`, `/tv`, `/healthz`.
- OpenAPI: served at `/openapi.json` and mirrored in `docs/openapi_v2.2.1.yaml`.
- Auth: optional Bearer via `BACKEND_API_KEY`; honors `X-User-Id` for scoping.
- Deployment: Railway (Nixpacks + Procfile) with in-memory caches (30s/120s/15s windows).

Validation checklist (executed against production)
- Health responds 200: `GET /healthz`.
- OpenAPI available: `GET /openapi.json` parses with `jq`.
- Plans build: `POST /gpt/scan` returns `plan` with entry/stop/targets/confidence and overlays.
- Multi-context returns volatility block and caches: `POST /gpt/multi-context`.
- Contracts ranked with liquidity gates: `POST /gpt/contracts` includes `table` rows and greeks.
- Chart URLs validate and render overlays: `POST /gpt/chart-url` then open returned `/tv` URL.
- Fallback behavior verified: Polygon snapshot 400s degrade gracefully to Tradier.

Operational notes
- Caching is in-memory; restarts clear caches. Configure `DB_URL` to persist plan snapshots.
- Polygon options entitlement is optional; warnings expected without it.
- For GPT Actions, point to the production host and use the served OpenAPI.

Change tracking
- Detailed change log and roadmap live in `docs/progress.md`.
- Integration details and payload contracts live in `docs/gpt_integration.md`.

