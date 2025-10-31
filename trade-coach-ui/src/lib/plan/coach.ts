import type { PlanSnapshot } from "@/lib/types";

export type CoachGoal =
  | "approach_tp"
  | "tp_hit"
  | "approach_stop"
  | "stop_hit"
  | "reentry_zone"
  | "neutral";

export type CoachNote = {
  text: string;
  goal: CoachGoal;
  progressPct: number;
  updatedAt: number;
};

export type Level = {
  id: string;
  type: "entry" | "stop" | "tp" | "reentry" | "vwap" | "ema";
  price: number;
  label: string;
  color?: string;
};

type Direction = "long" | "short";

const PRICE_TOLERANCE_PCT = 0.0015;
const MIN_PROGRESS_TOLERANCE = 0.02;

const NUMBER_FORMATTER = new Intl.NumberFormat("en-US", {
  style: "decimal",
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

export function deriveProgressPct({
  price,
  nextTarget,
  stop,
}: {
  price: number | null | undefined;
  nextTarget: number | null | undefined;
  stop: number | null | undefined;
}): number {
  if (!isFiniteNumber(price) || !isFiniteNumber(nextTarget) || !isFiniteNumber(stop)) {
    return 0;
  }
  if (nextTarget === stop) return 0;
  if (nextTarget > stop) {
    const span = nextTarget - stop;
    if (span === 0) return 0;
    const pct = ((price as number) - (stop as number)) / span;
    return clamp(pct * 100, 0, 100);
  }
  const span = (stop as number) - (nextTarget as number);
  if (span === 0) return 0;
  const pct = ((stop as number) - (price as number)) / span;
  return clamp(pct * 100, 0, 100);
}

export function deriveCoachMessage({
  plan,
  price,
  trailingStop,
  fills,
  now = Date.now(),
}: {
  plan: PlanSnapshot["plan"];
  price: number | null | undefined;
  trailingStop?: number | null;
  fills?: Array<{ type: "tp_hit" | "stop_hit" | "scale" | "reload"; targetPrice?: number | null; price?: number | null; at?: number | null }>;
  now?: number;
}): CoachNote {
  if (!isFiniteNumber(price)) {
    return {
      text: "Awaiting tick…",
      goal: "neutral",
      progressPct: 0,
      updatedAt: now,
    };
  }

  const entry = resolvePlanEntry(plan);
  const stop = resolvePlanStop(plan, trailingStop);
  const targets = resolvePlanTargets(plan);
  const direction = inferDirection(entry, stop, targets);
  const nextTargetInfo = pickNextTarget(direction, price as number, targets);
  const nextTargetPrice = nextTargetInfo?.price ?? null;

  let goal: CoachGoal = "neutral";
  let text = `Price ${formatPrice(price as number)}`;

  const progressPct = deriveProgressPct({
    price,
    nextTarget: nextTargetPrice,
    stop: stop ?? null,
  });

  const tolerance = computeTolerance(nextTargetPrice ?? stop ?? (price as number));
  const stopTolerance = computeTolerance(stop ?? (price as number));

  const stopTouch = stop != null && Math.abs((price as number) - stop) <= stopTolerance;
  const targetTouch =
    nextTargetPrice != null && Math.abs((price as number) - nextTargetPrice) <= tolerance;

  const hasStopFill = fills?.some((event) => event.type === "stop_hit") ?? false;
  const hasTargetFill = fills?.some((event) => event.type === "tp_hit") ?? false;

  if (stopTouch || hasStopFill) {
    goal = hasStopFill ? "stop_hit" : "approach_stop";
    const stopLabel = stop != null ? formatPrice(stop) : "stop";
    if (hasStopFill && fills) {
      const fill = fills.find((event) => event.type === "stop_hit" && isFiniteNumber(event.price));
      const fillLabel = fill?.price != null ? formatPrice(fill.price) : stopLabel;
      text = `Stop filled at ${fillLabel}.`;
    } else {
      text = `Protecting stop ${stopLabel}; price ${formatPrice(price as number)}.`;
    }
    return {
      text,
      goal,
      progressPct,
      updatedAt: now,
    };
  }

  if (nextTargetInfo) {
    const { label } = nextTargetInfo;
    const targetLabel = `${label} ${formatPrice(nextTargetPrice ?? price as number)}`;
    if (targetTouch || hasTargetFill) {
      goal = hasTargetFill ? "tp_hit" : "tp_hit";
      if (hasTargetFill && fills) {
        const fill = fills
          .filter((event) => event.type === "tp_hit" && isFiniteNumber(event.price))
          .sort((a, b) => (b.at ?? 0) - (a.at ?? 0))[0];
        if (fill?.price != null) {
          text = `${label} reached at ${formatPrice(fill.price)}.`;
        } else {
          text = `${label} reached at ${formatPrice(price as number)}.`;
        }
      } else {
        text = `${label} reached at ${formatPrice(price as number)}.`;
      }
    } else {
      goal = "approach_tp";
      const distance = Math.abs((nextTargetPrice ?? price as number) - (price as number));
      text = `Tracking toward ${targetLabel} · Δ ${formatPrice(distance)}.`;
    }
  } else if (direction && entry != null) {
    goal = "neutral";
    const entryLabel = formatPrice(entry);
    text = `Between entry ${entryLabel} and stop ${stop != null ? formatPrice(stop) : "—"}.`;
  }

  return {
    text,
    goal,
    progressPct,
    updatedAt: now,
  };
}

export function resolvePlanTargets(plan: PlanSnapshot["plan"]): Array<{ price: number; label: string }> {
  const targetsRaw = Array.isArray(plan.targets) ? plan.targets : [];
  const targetMeta = Array.isArray(plan.target_meta)
    ? (plan.target_meta as Array<{ label?: string | null; price?: number | null }>)
    : [];
  return targetsRaw
    .map((value, index) => {
      if (!isFiniteNumber(value)) return null;
      const meta = targetMeta[index];
      const label = typeof meta?.label === "string" && meta.label.trim() ? meta.label.trim() : `TP${index + 1}`;
      return { price: value as number, label };
    })
    .filter((entry): entry is NonNullable<typeof entry> => entry !== null);
}

export function resolvePlanEntry(plan: PlanSnapshot["plan"]): number | null {
  const direct = (plan as Record<string, unknown>).entry;
  if (isFiniteNumber(direct)) return direct as number;
  const structured = plan.structured_plan as { entry?: { level?: unknown } } | null | undefined;
  const structuredEntry = structured?.entry?.level;
  if (isFiniteNumber(structuredEntry)) return structuredEntry as number;
  const chartsParams = plan.charts_params as Record<string, unknown> | undefined;
  const paramEntry = chartsParams?.entry;
  if (isFiniteNumber(paramEntry)) return paramEntry as number;
  return null;
}

export function resolvePlanStop(plan: PlanSnapshot["plan"], trailingStop: number | null | undefined): number | null {
  if (isFiniteNumber(trailingStop)) return trailingStop as number;
  const direct = (plan as Record<string, unknown>).stop;
  if (isFiniteNumber(direct)) return direct as number;
  const structured = plan.structured_plan as { stop?: unknown } | null | undefined;
  if (structured && isFiniteNumber(structured.stop)) return structured.stop as number;
  const chartsParams = plan.charts_params as Record<string, unknown> | undefined;
  const paramStop = chartsParams?.stop;
  if (isFiniteNumber(paramStop)) return paramStop as number;
  const details = (plan.details ?? {}) as Record<string, unknown>;
  if (isFiniteNumber(details.stop)) return details.stop as number;
  return null;
}

function pickNextTarget(direction: Direction | null, price: number, targets: Array<{ price: number; label: string }>) {
  if (!direction || targets.length === 0) return null;
  const sorted = [...targets].sort((a, b) => (direction === "long" ? a.price - b.price : b.price - a.price));
  if (direction === "long") {
    const ahead = sorted.find((target) => price <= target.price);
    return ahead ?? sorted[sorted.length - 1];
  }
  const ahead = sorted.find((target) => price >= target.price);
  return ahead ?? sorted[sorted.length - 1];
}

function inferDirection(
  entry: number | null,
  stop: number | null,
  targets: Array<{ price: number; label: string }>,
): Direction | null {
  if (entry != null && stop != null) {
    if (stop < entry) return "long";
    if (stop > entry) return "short";
  }
  if (targets.length >= 2) {
    const first = targets[0].price;
    const last = targets[targets.length - 1].price;
    if (last > first) return "long";
    if (last < first) return "short";
  }
  return null;
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function clamp(value: number, min: number, max: number): number {
  if (Number.isNaN(value)) return min;
  return Math.min(Math.max(value, min), max);
}

function computeTolerance(anchor: number): number {
  if (!isFiniteNumber(anchor)) return MIN_PROGRESS_TOLERANCE;
  const scaled = Math.abs(anchor) * PRICE_TOLERANCE_PCT;
  return Math.max(scaled, MIN_PROGRESS_TOLERANCE);
}

function formatPrice(value: number): string {
  if (!isFiniteNumber(value)) return "—";
  return NUMBER_FORMATTER.format(value);
}
