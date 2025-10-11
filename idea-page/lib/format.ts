import type { Plan } from "@/lib/types";

const MARKET_PHASE_LABELS: Record<string, string> = {
  regular: "REG",
  premarket: "PRE",
  afterhours: "AFTER",
  closed: "CLOSED",
};

export function formatPrice(value: number | null | undefined, decimals = 2): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "—";
  }
  return value.toFixed(decimals);
}

export function formatRR(rr: number | null | undefined): string {
  if (rr === null || rr === undefined || Number.isNaN(rr)) {
    return "—";
  }
  return rr.toFixed(2);
}

export function formatConfidence(value: number | null | undefined): { emoji: string; label: string } {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return { emoji: "❓", label: "N/A" };
  }
  if (value >= 0.65) return { emoji: "🟢", label: value.toFixed(2) };
  if (value >= 0.35) return { emoji: "🟠", label: value.toFixed(2) };
  return { emoji: "🔴", label: value.toFixed(2) };
}

export function formatMarketPhase(phase: string | null | undefined): string {
  if (!phase) return "REG";
  return MARKET_PHASE_LABELS[phase.toLowerCase()] ?? phase.toUpperCase();
}

export function formatDateTime(timestamp?: string | null): string {
  if (!timestamp) return "—";
  try {
    const date = new Date(timestamp);
    return date.toLocaleString("en-US", {
      timeZone: "America/Chicago",
      dateStyle: "medium",
      timeStyle: "short",
    });
  } catch (error) {
    return timestamp;
  }
}

export function buildLevelsCopy(plan: Plan): string {
  const rr = formatRR(plan.rr_to_t1);
  const targets = [...plan.targets, ...Array(3 - plan.targets.length).fill(null)]
    .slice(0, 2)
    .map((tp) => (tp === null ? "—" : tp.toFixed(plan.decimals)));
  return `${plan.symbol} | ${plan.bias} | entry ${plan.entry.toFixed(plan.decimals)} | stop ${plan.stop.toFixed(
    plan.decimals,
  )} | tp1 ${targets[0]} | tp2 ${targets[1]} | RR: ${rr}`;
}

export function buildPlanString(plan: Plan, snappedTargets?: string[] | null): string {
  const rr = formatRR(plan.rr_to_t1);
  const atrMultiple = snappedTargets && snappedTargets.length > 0 ? ` | ${snappedTargets[0]}` : "";
  return `${plan.bias === "long" ? "Long" : "Short"} ${plan.entry.toFixed(plan.decimals)} | s ${plan.stop.toFixed(
    plan.decimals,
  )} | t1 ${plan.targets[0]?.toFixed(plan.decimals)} | t2 ${plan.targets[1]?.toFixed(
    plan.decimals,
  )}${atrMultiple}. Conf ${rr}`;
}
