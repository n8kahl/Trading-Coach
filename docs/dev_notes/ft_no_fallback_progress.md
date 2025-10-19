# Fancy Trader “No Fallback Trades” Hardening – Progress Log

**Updated:** 2025-10-19  
**Owner:** Codex agent (GPT-5)  

## Completed
- Enforced `FT_NO_FALLBACK_TRADES` in `/gpt/scan`; frozen fallback lists are disabled and empty results return `NO_ELIGIBLE_SETUPS` with `X-No-Fabrication: 1`.  
- Persist the latest scan cursor per `(user_id, session_signature, style)` to support plan gating.  
- `/gpt/plan` now rejects symbols not present in the most recent scan (409 `PLAN_NOT_IN_LAST_SCAN`) and always emits `X-No-Fabrication: 1`.  
- Added options guardrails (spread %, open interest, IV presence) and surfaced `rejected_contracts` in plan responses (both fallback and primary).  
- Registered `AllowedHostsMiddleware`; chart generation validates against `FT_ALLOWED_HOSTS`, `PUBLIC_BASE_URL`, and `CHART_BASE_URL`.  
- Hardened chart request models (`extra="forbid"`) and applied host checks to both canonical and fallback chart URLs.  
- In the fallback plan pipeline:
  - Added SL/TP invariant checks (stop-entry-target ordering per direction).  
  - Applied EM cap using `FT_EM_FACTOR`; plans now flag `em_used` when ceilings/floors activate.  
  - Event-window gating populates `within_event_window`/`minutes_to_event`, blocks options when styles are in `FT_EVENT_BLOCKED_STYLES`, and raises an `EVENT_WINDOW_BLOCKED` warning.  
  - Response fields now expose `phase="hydrate"`, `layers_fetched`, enriched `meta`, and expanded session state details.
- Mirrored the same EM-cap, invariant enforcement, event-window gating, and metadata/warning propagation in the main `/gpt/plan` path (including options blocking + hydration metadata).
- Captured a full planning-mode architecture (dynamic universes, readiness scoring, contract templates, finalizer) in `docs/dev_notes/planning_mode_architecture.md` for implementation.

## In Progress / Next Steps
1. Mirror invariant + EM-cap + event-window enforcement in the primary `/gpt/plan` path (currently only covered in fallback generator).  
2. Add data-quality drift detection (`FT_MAX_DRIFT_BPS`) and banner logic in scan results; ensure top-N filtering respects drift flag.  
3. Populate `source_paths`, `accuracy_levels`, `plan_layers`, and other new schema fields for live plans; ensure `tp_reasons`, `confluence`, `options_contracts` are non-null arrays.  
4. Wire hydration progress metadata for `/gpt/chart-layers` (`phase`, `layers_count`).  
5. Test suite additions:
   - Golden deterministic scan fixtures (including rank + JSON diff).  
   - Guardrail/invariant/event-window tests.  
   - Header checks (`X-No-Fabrication`).  
   - Extra-field rejection (thanks to `extra="forbid"`).  
   - Options guardrail rejection coverage.  
6. Add CI step comparing runtime `/openapi.json` vs. `docs/openapi_v2.2.1.yaml`.  
7. Update OpenAPI + prompt docs to reflect new response attributes and error codes.

## Reference
- Env knobs introduced: `FT_NO_FALLBACK_TRADES`, `FT_MAX_SPREAD_PCT`, `FT_MIN_OI`, `FT_MAX_DRIFT_BPS`, `FT_EM_FACTOR`, `FT_ALLOWED_HOSTS`, `FT_EVENT_BLOCKED_STYLES`.  
- Registry key used for scan cursor enforcement: `(user_id, session_signature, style|__any__)`, where `session_signature = "{status}|{as_of}|{style}"`.
