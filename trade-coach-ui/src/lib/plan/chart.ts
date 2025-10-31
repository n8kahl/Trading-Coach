import type { PriceSeriesCandle } from "@/lib/hooks/usePriceSeries";

type FitRangeArgs = {
  bars: PriceSeriesCandle[];
  entry?: number | null;
  stop?: number | null;
  targets?: number[];
  lookback?: number;
  paddingPct?: number;
};

export function fitVisibleRangeForTrade({
  bars,
  entry,
  stop,
  targets = [],
  lookback = 100,
  paddingPct = 0.05,
}: FitRangeArgs): { min: number; max: number } | null {
  if (!Array.isArray(bars) || bars.length === 0) {
    const levels = collectLevels(entry, stop, targets);
    if (levels.length === 0) return null;
    return applyPadding(levels, paddingPct);
  }

  const slice = bars.slice(-lookback);
  const extremes: number[] = [];
  slice.forEach((bar) => {
    if (isFiniteNumber(bar.low)) extremes.push(bar.low);
    if (isFiniteNumber(bar.high)) extremes.push(bar.high);
  });
  extremes.push(...collectLevels(entry, stop, targets));

  const filtered = extremes.filter(isFiniteNumber);
  if (filtered.length === 0) return null;

  return applyPadding(filtered, paddingPct);
}

function collectLevels(entry?: number | null, stop?: number | null, targets: number[] = []): number[] {
  const values: number[] = [];
  if (isFiniteNumber(entry)) values.push(entry as number);
  if (isFiniteNumber(stop)) values.push(stop as number);
  targets.forEach((target) => {
    if (isFiniteNumber(target)) values.push(target as number);
  });
  return values;
}

function applyPadding(values: number[], paddingPct: number): { min: number; max: number } {
  if (values.length === 0) {
    return { min: 0, max: 0 };
  }
  const minValue = Math.min(...values);
  const maxValue = Math.max(...values);
  if (minValue === maxValue) {
    const offset = Math.max(Math.abs(minValue) * paddingPct, 1);
    return { min: minValue - offset, max: maxValue + offset };
  }
  const span = maxValue - minValue;
  const padding = span * (paddingPct > 0 ? paddingPct : 0);
  return {
    min: minValue - padding,
    max: maxValue + padding,
  };
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}
