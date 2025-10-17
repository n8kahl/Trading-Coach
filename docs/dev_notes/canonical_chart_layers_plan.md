# Canonical Chart URL + Plan Layers Rollout

Last updated: 2025-10-16

## Objectives

1. Emit canonical `/tv` URLs from `/gpt/chart-url` with a minimal, deterministic query string.
2. Persist plan-bound chart layers (levels, zones, annotations) at plan generation time.
3. Serve persisted layers via `/api/v1/gpt/chart-layers?plan_id=...`.
4. Update the `/tv` client to fetch and render overlays using `plan_id`.
5. Respect instrument precision when formatting numeric parameters.
6. Update docs/tests; prompt changes ship after the flag is enabled.

No backwards compatibility is required for historical plans or query parameters.

## Constraints & Assumptions

* Session SSOT already exists; reuse it for `as_of` handling.
* Plan generation has access to key levels and overlays—capture them before returning the response.
* We can persist additional fields inside plan snapshots without migrations (uses JSON).
* No PNG/static preview requirements.

## Rollout Flag

* Introduce `FF_CHART_CANONICAL_V1` (default OFF).
* All new behaviour guarded behind this flag until server & UI are ready.
* Prompt/doc updates land after the flag is verified.

## Implementation Tasks

### 1. Precision Helper
* Create utility (`src/app/services/precision.py`) that returns the appropriate decimal places per symbol.
* Inputs: hard-coded map + ability to extend with instrument metadata later.

### 2. Canonical URL Builder
* Add `src/app/services/chart_url.py` with `make_chart_url(params, precision_resolver)`.
* Allow keys: `symbol, interval, direction, entry, stop, tp, ema, focus, center_time, scale_plan, view, range, theme, plan_id, plan_version`.
* Apply precision helper for `entry/stop/tp`, deterministic ordering, uppercase symbol.
* Update `/gpt/chart-url` (and any call sites) to rely on the helper when the flag is ON.

### 3. Plan Layer Capture
* Introduce `build_plan_layers(...)` helper to gather:
  - numeric levels (`key_levels`, significant overlay lines),
  - zones (supply/demand, gaps),
  - annotations (optional; e.g., runner notes).
* Include metadata: `as_of`, `symbol`, `interval`, `precision`, `planning_context`.
* When generating plan responses (scan, plan, fallback), attach `plan_layers` to:
  - `PlanResponse`,
  - in-memory payloads,
  - persisted snapshots (idea store).
* Only under flag; otherwise maintain current payload (without layers).

### 4. Chart Layers Endpoint
* Add FastAPI router `src/app/routers/chart_layers.py`.
* `GET /api/v1/gpt/chart-layers?plan_id=` -> retrieve snapshot via existing helper (`_ensure_snapshot`).
* Return persisted `plan_layers`; if missing and flag ON, raise 404 (since we do not support reconstruction).

### 5. `/tv` Client Updates
* Parse `plan_id` in `static/tv` (vanilla JS) and/or Next.js front-end (trade-coach-ui).
* When flag ON and `plan_id` exists:
  - Fetch `/api/v1/gpt/chart-layers` and render overlays.
  - Ignore unknown query params; base charts still use canonical fields.
* Provide graceful message if layers missing.

### 6. Tests
* Unit tests for `make_chart_url` (precision, allow-listing, ordering).
* Unit snapshot for `build_plan_layers`.
* API tests:
  - `/gpt/chart-url` returns canonical params under flag.
  - `/api/v1/gpt/chart-layers` yields persisted data.
* Adjust existing plan/chart tests (`test_chart_links.py`, `test_plan_renderer.py`, etc.).

### 7. Docs & Prompt
* Update README + integration docs with new endpoint & URL contract.
* Update master prompt once flag validated (note to revisit).

## Validation Checklist

* `pytest` passes with flag ON.
* Manual verification: request plan → URL contains only canonical params, `/tv` overlays match plan.
* Flag OFF behaviour remains unchanged until rollout complete.

---

### Progress Log

**2025-10-16**

* Documented objectives and tasks (this file).
* Added precision + chart URL builder utilities; `/gpt/chart-url` now emits canonical links when `FF_CHART_CANONICAL_V1=1`.
* Persisting plan-bound layers within plan responses and snapshots; `/api/v1/gpt/chart-layers` exposes stored data.
* `/tv` bootstrap now fetches layers by `plan_id` and upgrades level rendering; canonical fetch errors logged gracefully.
