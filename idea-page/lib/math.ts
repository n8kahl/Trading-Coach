export type WhatIfParams = {
  fillPrice: number;
  ivShiftBps: number;
  contracts?: number;
};

export type ContractBaseline = {
  price: number;
  bid: number;
  ask: number;
  mark?: number | null;
  delta: number;
  theta?: number | null;
  iv?: number | null;
};

export function computeWhatIf(
  baseline: ContractBaseline,
  params: WhatIfParams,
): { optimistic: number; pessimistic: number; neutral: number } {
  const { fillPrice, ivShiftBps, contracts = 1 } = params;
  const mark = baseline.mark ?? (baseline.bid + baseline.ask) / 2;
  const spread = Math.abs(baseline.ask - baseline.bid);
  const ivShift = (baseline.iv ?? 0) * (ivShiftBps / 10000);

  const optimistic = (mark + spread * 0.5 + ivShift) * contracts * 100 - fillPrice * contracts * 100;
  const neutral = (mark + ivShift * 0.5) * contracts * 100 - fillPrice * contracts * 100;
  const pessimistic = (mark - spread * 0.5 + ivShift * 0.25) * contracts * 100 - fillPrice * contracts * 100;

  return {
    optimistic: Number.isFinite(optimistic) ? optimistic : 0,
    neutral: Number.isFinite(neutral) ? neutral : 0,
    pessimistic: Number.isFinite(pessimistic) ? pessimistic : 0,
  };
}
