export type SessionStatus = "open" | "closed";

export interface Session {
  status: SessionStatus;
  as_of: string;
  next_open?: string | null;
  tz?: string;
  banner?: string;
}

export type Style = "scalp" | "intraday" | "swing" | "leap";
export type Direction = "long" | "short";
export type EntryType = "break" | "retest" | "reclaim" | "reject" | "limit";

export interface Entry {
  type: EntryType;
  level: number;
}

export interface KeyLevels {
  ORH?: number;
  ORL?: number;
  session_high?: number;
  session_low?: number;
  prev_high?: number;
  prev_low?: number;
  gap_fill?: number;
  POC?: number;
  VAH?: number;
  VAL?: number;
  HVN?: number;
  LVN?: number;
  liquidity_pools?: number[];
  notes?: string;
}

export interface MtfFrame {
  trend?: string;
  ema_state?: string;
  structure?: string;
  momentum?: string;
}

export interface MtfAnalysis {
  ["5m"]?: MtfFrame;
  ["15m"]?: MtfFrame;
  ["1h"]?: MtfFrame;
  ["4h"]?: MtfFrame;
  ["1D"]?: MtfFrame;
}

export interface ProbabilityComponents {
  trend_alignment?: number;
  liquidity_structure?: number;
  momentum_signal?: number;
  volatility_regime?: number;
}

export interface RiskModel {
  expected_value_r?: number;
  kelly_fraction?: number;
  mfe_projection?: string;
}

export interface ContextBlock {
  macro?: string;
  sector?: { name?: string; rel_vs_spy?: number; z?: number };
  internals?: { breadth?: number; vix?: number; tick?: number };
  rs?: { vs_benchmark?: number };
}

export interface Probabilities {
  tp1?: number;
  tp2?: number;
  [key: string]: number | undefined;
}

export interface OptionExample {
  type: "call" | "put";
  expiry: string;
  delta?: number;
  bid?: number;
  ask?: number;
  spread_pct?: number;
  spread_stability?: number;
  oi?: number;
  volume?: number;
  iv_percentile?: number;
  composite_score?: number;
  tradeability?: number;
}

export interface OptionsBlock {
  style_horizon_applied?: Style;
  dte_window?: string;
  example?: OptionExample;
  note?: string;
}

export interface Setup {
  plan_id: string;
  symbol: string;
  style: Style;
  direction: Direction;
  version?: number;
  entry: Entry;
  invalid?: number;
  stop: number;
  targets: number[];
  probabilities?: Probabilities;
  probability_components?: ProbabilityComponents;
  trade_quality_score?: string;
  runner?: Record<string, unknown>;
  confluence?: string[];
  key_levels?: KeyLevels;
  mtf_analysis?: MtfAnalysis;
  context?: ContextBlock;
  confidence: number;
  rationale?: string;
  options?: OptionsBlock;
  risk_model?: RiskModel;
  em_used?: number;
  atr_used?: number;
  style_horizon_applied?: Style;
  chart_url: string;
  as_of: string;
}

export interface ExecResponse {
  ok: boolean;
  session: Session;
  count: number;
  setups: Setup[];
}

export type PlanEventType =
  | "price"
  | "hit"
  | "stop"
  | "invalidate"
  | "reverse"
  | "update"
  | "note";

export interface PlanEvent {
  type: PlanEventType;
  plan_id: string;
  symbol: string;
  ts?: string;
  price?: number;
  hit?: string;
  setup?: Setup;
  note?: string;
}
