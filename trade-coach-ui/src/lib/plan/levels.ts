import type { PlanSnapshot } from "@/lib/types";
import type { Level } from "./coach";
import { resolvePlanEntry, resolvePlanStop, resolvePlanTargets } from "./coach";

type ExtractOptions = {
  trailingStop?: number | null;
};

export function extractPlanLevels(
  plan: PlanSnapshot["plan"],
  { trailingStop }: ExtractOptions = {},
): { levels: Level[]; entry: number | null; stop: number | null; trailingStop: number | null } {
  const entry = resolvePlanEntry(plan);
  const stop = resolvePlanStop(plan, trailingStop ?? resolveTrailingStop(plan));
  const trailing = resolveTrailingStop(plan, trailingStop);
  const targets = resolvePlanTargets(plan);

  const levels: Level[] = [];
  if (entry != null) {
    levels.push({
      id: "plan:entry",
      type: "entry",
      price: entry,
      label: "Entry",
      color: "#10b981",
    });
  }
  if (stop != null) {
    levels.push({
      id: "plan:stop",
      type: "stop",
      price: stop,
      label: "Stop",
      color: "#f87171",
    });
  }
  if (trailing != null && (!stop || Math.abs(trailing - stop) > 1e-6)) {
    levels.push({
      id: "plan:trail",
      type: "stop",
      price: trailing,
      label: "Trail",
      color: "#f59e0b",
    });
  }
  targets.forEach((target, index) => {
    levels.push({
      id: `plan:tp:${index + 1}`,
      type: "tp",
      price: target.price,
      label: target.label,
      color: "#22d3ee",
    });
  });

  return { levels, entry, stop, trailingStop: trailing };
}

export function resolveTrailingStop(plan: PlanSnapshot["plan"], override?: number | null): number | null {
  if (override != null) return override;
  const details = (plan as Record<string, unknown>).details;
  if (details && typeof details === "object") {
    const detailStop = (details as Record<string, unknown>).stop;
    if (isFiniteNumber(detailStop)) return detailStop as number;
  }
  const direct = (plan as Record<string, unknown>).trailing_stop ?? (plan as Record<string, unknown>).trail_stop;
  if (isFiniteNumber(direct)) return direct as number;
  const chartsParams = plan.charts_params as Record<string, unknown> | undefined;
  if (chartsParams) {
    const tokens = ["trailingStop", "trailing_stop", "trail_stop", "trail"];
    for (const token of tokens) {
      const candidate = chartsParams[token];
      if (isFiniteNumber(candidate)) return candidate as number;
    }
  }
  return null;
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}
