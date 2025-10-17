# Scenario Plans – Market Replay (Client‑side)

Summary
- Add frozen “Scenario” plans alongside the single auto‑updating Live plan per symbol.
- Styles: `scalp`, `intraday`, `swing` (gate `reversal` until the server strategy exists).
- Canonical charts only; overlays fetched by `plan_id` (no inline/session params in URLs).

UI
- Route: `trade-coach-ui/src/app/replay/[symbol]`
- Controls: segmented style switcher + “Generate Plan” → `POST /gpt/plan`.
- Cards per scenario: badges (Scenario + style; Live vs Frozen), entry/stop/tps, confidence, rr_to_t1, actions: Compare | Adopt | Regenerate | Delete, and “Link to Live”.
- Compare overlay: ghost lines drawn by `PriceChart.compare` (entry/stop/TPs) over the Live chart.

State model (client only)
- `scenario_of: string | null` – originating live `plan_id`.
- `scenario_style: 'scalp'|'intraday'|'swing'|'reversal'|null` – style selector.
- `linked_to_live: boolean` – default false; when true, regenerates on live invalidation.
- `frozen: boolean` – scenarios are snapshots (true); Live is false.
- Max 3 scenarios per symbol; stored in `localStorage` (`tc:scenarios:<SYMBOL>`). Live UI pointer in `tc:live:<SYMBOL>`.

APIs
- Generate: `POST /gpt/plan { symbol, style }` → use `plan.plan_id` and `plan.trade_detail|idea_url|charts.interactive` for canonical chart links.
- Overlays: `/api/v1/gpt/chart-layers?plan_id=...` (resolved by `/tv`).
- No server persistence for scenarios in v1.

Adoption
- “Adopt as Live” swaps the UI’s active `plan_id` pointer to the scenario.
- Optional “Regenerate & Adopt” is available by regenerating the same style first, then adopting the new `plan_id`.

Invalidation
- The Replay page subscribes to `/ws/plans/{plan_id}` for the active Live plan; upon `invalidated`, it regenerates any scenarios with `linked_to_live=true`.

Flags
- None. Scenario Plans are enabled by default. The `reversal` button remains disabled until a `strategy_id="reversal"` is available server‑side.

Telemetry (client only)
- Console counters: `scenario_generated`, `scenario_adopted`, `scenario_compared`, `scenario_regenerated`, `scenario_deleted`.
