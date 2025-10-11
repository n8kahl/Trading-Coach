import { z } from "zod";

export const ChartsParams = z.object({
  symbol: z.string(),
  interval: z.string(),
  direction: z.enum(["long", "short"]),
  strategy: z.string().nullable().optional(),
  entry: z.number().nullable().optional(),
  stop: z.number().nullable().optional(),
  tp: z.string().nullable().optional(),
  levels: z.string().nullable().optional(),
  view: z.string().nullable().optional(),
  notes: z.string().nullable().optional(),
  ema: z.string().nullable().optional(),
  vwap: z.string().nullable().optional(),
  atr: z.string().nullable().optional(),
  title: z.string().nullable().optional(),
});

export const PlanCore = z.object({
  plan_id: z.string(),
  version: z.number(),
  symbol: z.string(),
  style: z.enum(["scalp", "intraday", "swing", "leaps"]),
  bias: z.enum(["long", "short"]),
  setup: z.string(),
  entry: z.number(),
  stop: z.number(),
  targets: z.array(z.number()).min(1).max(3),
  rr_to_t1: z.number(),
  confidence: z.number(),
  decimals: z.number(),
  trade_detail: z.string().optional(),
  charts_params: ChartsParams,
  warnings: z.array(z.string()).optional(),
  planning_context: z.enum(["live", "offline", "backtest"]).optional(),
});

export const OfflineBasis = z
  .object({
    htf_snapshot_time: z.string().optional(),
    volatility_regime: z.string().optional(),
    expected_move_days: z.number().optional(),
  })
  .optional();

export const VolatilityRegime = z.object({
  iv_rank: z.number().nullable().optional(),
  iv_percentile: z.number().nullable().optional(),
  hv_20: z.number().nullable().optional(),
  hv_60: z.number().nullable().optional(),
  realized_vol: z.number().nullable().optional(),
  skew_index: z.number().nullable().optional(),
  regime_label: z.enum(["low", "normal", "elevated", "extreme"]).nullable().optional(),
});

export const OptionContract = z.object({
  label: z.string(),
  expiry: z.string(),
  dte: z.number(),
  strike: z.number(),
  type: z.enum(["CALL", "PUT"]),
  price: z.number(),
  bid: z.number(),
  ask: z.number(),
  mark: z.number().nullable().optional(),
  spread_pct: z.number(),
  delta: z.number(),
  theta: z.number().nullable().optional(),
  iv: z.number().nullable().optional(),
  oi: z.number().nullable().optional(),
  liquidity_score: z.number().nullable().optional(),
  last_trade_time: z.string().nullable().optional(),
});

export const IdeaOptions = z
  .object({
    table: OptionContract.array().optional(),
    side: z.enum(["call", "put"]).optional(),
    style: z.enum(["scalp", "intraday", "swing", "leaps"]).optional(),
  })
  .nullable()
  .optional();

export const SummaryDTO = z.object({
  frames_used: z.array(z.string()).optional(),
  confluence_score: z.number().optional(),
  trend_notes: z.record(z.string()).optional(),
  expected_move_horizon: z.number().nullable().optional(),
  nearby_levels: z.array(z.string()).optional(),
});

export const IdeaSnapshot = z.object({
  plan: PlanCore,
  summary: SummaryDTO,
  volatility_regime: VolatilityRegime.optional(),
  htf: z
    .object({
      bias: z.enum(["aligned", "neutral", "opposed", "unknown"]).nullable().optional(),
      snapped_targets: z.array(z.string()).nullable().optional(),
    })
    .optional(),
  data_quality: z
    .object({
      series_present: z.boolean().nullable().optional(),
      iv_present: z.boolean().nullable().optional(),
      earnings_present: z.boolean().nullable().optional(),
    })
    .optional(),
  chart_url: z.string(),
  options: IdeaOptions,
  why_this_works: z.array(z.string()),
  invalidation: z.array(z.string()).optional(),
  risk_note: z.string().nullable().optional(),
  calc_notes: z
    .object({
      atr14: z.number().optional(),
      stop_multiple: z.number().optional(),
      rr_inputs: z
        .object({
          entry: z.number().optional(),
          stop: z.number().optional(),
          tp1: z.number().optional(),
        })
        .optional(),
    })
    .nullable()
    .optional(),
  planning_context: z.enum(["live", "offline", "backtest"]).optional(),
  offline_basis: OfflineBasis,
});

export type TIdeaSnapshot = z.infer<typeof IdeaSnapshot>;
export type Plan = z.infer<typeof PlanCore>;
