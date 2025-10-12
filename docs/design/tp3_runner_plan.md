# TP Ladder Enhancements & Runner Management (Design Snapshot)

**Goal**  
Upgrade the take‑profit engine so every plan presents:

1. A third stretch target (TP3) sized from realized statistics + implied volatility.
2. Explicit “runner” management guidance instead of a hard final TP.
3. Supporting metadata (EM / MFE fractions, probability of touch) for transparency.

This is scoped for the Trading Coach backend and charts with minimal prompt changes.

---

## Data Inputs

| Input | Source | Notes |
| --- | --- | --- |
| **Expected Move (EM)** | Options chain (Tradier/Polygon) | Choose expiry closest to style horizon (same day, 0–2 DTE, 7–10 DTE, 30–60 DTE). Fallback to historical ATR when unavailable. |
| **Historical MFE/MAE** | Polygon aggregates | Use rolling windows per style (scalp: 5/15 min; swing: 1h/4h/1D; leaps: 1D/1W). Compute max favorable excursion % and time‑to-hit distribution bucketed by volatility regime + trend alignment. Cache results in memory with TTL (≈30 min) keyed by symbol + style. |
| **Structure Levels** | Existing HTF level collector | Already available via `_collect_htf_levels`, volume profile, Fib projections. |
| **Probability of Touch (POT)** | Approximated from option delta | POT ≈ `min(1, 2 * |delta|)` for the matching expiry/strike. Use when chain available; otherwise derive from historical hit-rate. |

---

## Target Placement Rules

Let EM = implied move for horizon, MFE_Qp = historical quantile (e.g. 0.5, 0.8, 0.9).

| Style | Horizon | TP1 (High probability) | TP2 (Moderate) | TP3 (Stretch) |
| --- | --- | --- | --- | --- |
| **0DTE / Scalp** | 30–120 min | `min(0.30–0.35 × EM, MFE_40–50%)` | `min(0.55–0.70 × EM, MFE_60–75%)` | Optional: `min(0.85 × EM, MFE_80–85%)` only when trend breadth strong. |
| **Intraday** | Session remainder | `min(0.40–0.55 × EM, MFE_50%)` | `min(0.70–0.85 × EM, MFE_70–80%)` | `min(1.0–1.1 × EM, MFE_85–90%)` |
| **Swing** | 3–10 trading days | `snap(MFE_50%)` | `snap(MFE_75–80%)` | `snap(MFE_90%)` (cap ≤ 1.2 × daily EM unless strong HTF confluence) |
| **LEAPS** | 1–3 month campaigns | Weekly MFE_50% | Weekly MFE_75% | Weekly MFE_90% / prior monthly extremes |

*All targets snap to nearest HTF level (POC/VAH/VAL, prior H/L, Fib cluster) within ±0.15–0.25 × ATR.*
*Skip TP3 when RR(TP1) < min_rr or POT(TP3) < 25%.*  

Metadata stored per TP:
- `em_fraction`
- `mfe_quantile`
- `pot_est`
- `structure_tag` (level name snapped to)

---

## Runner Strategy

Instead of a static TP4, emit a trailing rule:

| Style | Trail |
| --- | --- |
| 0DTE/Scalp | Chandelier on 1m/5m (N=10, k=1.4) or VWAP − 1 ATR |
| Intraday | Chandelier on 5m/15m (N=14–20, k=1.8) or EMA20 − 1 ATR |
| Swing | Chandelier on 4h/1D (N=20, k=2.3) or last two swing lows |
| LEAPS | Weekly swing structure / 1D ATR(14) trail (k≈3) |

Plan payload additions:
- `runner`: `{ "type": "chandelier", "tf": "4h", "n": 20, "k": 2.3, "note": "Trail below last two swing lows" }`

Charts: draw runner guide as bright pink dashed line (optional initial anchor at TP3).

---

## Implementation Outline

1. **Data Layer**
   - New module `src/statistics.py` to compute/cache MFE/MAE quantiles per symbol/style.
   - Extend option fetch helpers to expose IV/EM for given horizon.

2. **Target Engine**
   - Enhance `_apply_tp_logic` to accept `target_context` (EM, quantiles, POT thresholds).
   - Compute TP1/TP2/TP3 base distances before snapping.
   - Attach metadata (`calc_notes["tp_meta"]` and per-target dict).

3. **Runner Logic**
   - New helper `_derive_runner_trail(style, bias, ctx)` returning trailing config.
   - Include in `PlanResponse.runner` and persist in idea snapshot.

4. **Schema & Prompt**
   - Update OpenAPI (`PlanResponse.targets` → array of objects with `price`, `label`, metadata).
   - Prompt instructions: describe TP ladder and runner trail; highlight POT/EM context.

5. **Charts/UI**
   - Update `/tv` renderer to plot TP3 (light purple) and runner trail (bright pink).
   - Legend entries show metadata (e.g., “TP3 · 1.05× EM · POT 28%”).

6. **Testing**
   - Unit tests for quantile selection, EM fallback, runner config.
   - Scenario tests ensuring TP3 suppressed when RR or POT fails.

---

## Risks & Mitigations

| Risk | Mitigation |
| --- | --- |
| Polygon historical fetch latency | Cache quantiles per symbol/style (TTL 30–60 min) and precompute asynchronously. |
| IV unavailable (after hours) | Fallback to recent daily EM derived from ATR; log downgrade. |
| Overfitting quantile buckets | Limit segmentation: volatility regime (low/normal/elevated/extreme) + trend alignment (aligned/neutral/opposed). Require ≥100 samples; else fallback to global stats. |
| UX overload | Keep plan UI compact: show TP prices with summarized meta, plus single “Runner: Trail …” line. |

---

**Next actions**  
1. Build `statistics.py` cache + EM helper.  
2. Integrate into `_apply_tp_logic` and emit metadata.  
3. Extend schema/prompt and chart renderer.  
