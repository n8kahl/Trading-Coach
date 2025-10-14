export type StructuredPlan = {
  plan_id: string;
  symbol: string;
  style: string | null;
  direction: string | null;
  entry: { type: string; level: number };
  stop: number;
  invalid?: boolean;
  targets: number[];
  runner?: Record<string, unknown> | null;
  chart_url?: string | null;
  as_of?: string | null;
};

export type PlanSnapshot = {
  plan: {
    plan_id: string;
    symbol: string;
    style?: string | null;
    bias?: string | null;
    entry?: number | null;
    stop?: number | null;
    targets?: number[];
    rr_to_t1?: number | null;
    confidence?: number | null;
    notes?: string | null;
    warnings?: string[];
    session_state?: {
      status: string;
      banner: string;
      as_of: string;
      next_open?: string | null;
    };
    structured_plan?: StructuredPlan | null;
    target_profile?: Record<string, unknown> | null;
    [key: string]: unknown;
  };
  summary?: Record<string, unknown>;
  volatility_regime?: Record<string, unknown>;
  htf?: Record<string, unknown>;
  data_quality?: Record<string, unknown>;
  options?: Record<string, unknown>;
  chart_url?: string | null;
  idea_url?: string | null;
};

export type PlanDeltaEvent = {
  t: "plan_delta";
  plan_id: string;
  version: number;
  changes: {
    status: string;
    next_step?: string;
    note?: string;
    rr_to_t1?: number | null;
    breach?: string | null;
    timestamp?: string;
    last_price?: number;
    trailing_stop?: number;
  };
  reason?: string;
};

export type SymbolTickEvent =
  | { t: "tick"; p: number; ts: string; source?: string }
  | { t: "market_status"; phase: string; note: string };

export type PlanStreamEvent = PlanDeltaEvent | Record<string, unknown>;
