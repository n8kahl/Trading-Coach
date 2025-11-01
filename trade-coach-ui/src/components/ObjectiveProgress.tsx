"use client";

import clsx from "clsx";
import { useMemo } from "react";
import { useStore } from "@/store/useStore";

type ObjectiveMeta = {
  state?: string;
  why?: string[];
  objective_price?: number;
  band?: { low?: number; high?: number };
  timeframe?: string;
  progress?: number;
};

type ObjectiveInternalMeta = {
  _entry_distance_pct?: number;
  _last_price?: number;
  _tick_size?: number;
};

function coerceNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const parsed = Number.parseFloat(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return null;
}

function clamp(value: number, min = 0, max = 1): number {
  if (Number.isNaN(value)) return min;
  return Math.min(max, Math.max(min, value));
}

function formatPrice(value: number | null, precision = 2): string {
  if (value == null || Number.isNaN(value)) return "—";
  return value.toLocaleString("en-US", {
    minimumFractionDigits: precision,
    maximumFractionDigits: precision,
  });
}

function friendlyState(token: string | null | undefined): string {
  if (!token) return "Unknown";
  const normalized = token.toLowerCase();
  if (normalized === "arming") return "Arming";
  if (normalized === "ready") return "Ready";
  if (normalized === "cooldown") return "Cooling Down";
  if (normalized === "invalid") return "Invalid";
  return token.replace(/[_-]/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

export default function ObjectiveProgress() {
  const plan = useStore((state) => state.plan);
  const planLayers = useStore((state) => state.planLayers);
  const objective = useMemo(() => {
    if (!planLayers?.meta) return null;
    const meta = planLayers.meta as Record<string, unknown>;
    const publicRaw = meta.next_objective;
    if (!publicRaw || typeof publicRaw !== "object") return null;
    const payload = publicRaw as ObjectiveMeta;
    const internalRaw = meta._next_objective_internal;
    const internal =
      internalRaw && typeof internalRaw === "object" ? (internalRaw as Record<string, unknown>) : ({} as Record<string, unknown>);

    const objectivePrice = coerceNumber(payload.objective_price) ?? null;
    const bandLow = coerceNumber(payload.band?.low) ?? null;
    const bandHigh = coerceNumber(payload.band?.high) ?? null;
    const lastPriceCandidate =
      coerceNumber(internal._last_price) ??
      coerceNumber(plan?.last_price) ??
      coerceNumber(plan?.mark) ??
      coerceNumber((plan?.details as Record<string, unknown> | undefined)?.last);
    const entryDistancePct =
      coerceNumber(internal._entry_distance_pct) ?? coerceNumber((payload as Record<string, unknown>).entry_distance_pct);
    const tickSize =
      coerceNumber(internal._tick_size) ??
      (typeof planLayers.precision === "number" ? 1 / Math.pow(10, Math.max(0, planLayers.precision)) : null);
    const progress = clamp(coerceNumber(payload.progress) ?? 0);

    return {
      raw: payload,
      internal: internal as ObjectiveInternalMeta,
      objectivePrice,
      bandLow,
      bandHigh,
      lastPrice: lastPriceCandidate,
      entryDistancePct,
      progress,
      tickSize,
    };
  }, [plan, planLayers]);

  if (!objective) {
    return null;
  }

  const { raw, objectivePrice, bandLow, bandHigh, lastPrice, entryDistancePct, progress, tickSize } = objective;

  const pricePrecision =
    typeof planLayers?.precision === "number" && Number.isFinite(planLayers.precision) ? Math.max(0, planLayers.precision) : 2;

  const distancePts = lastPrice != null && objectivePrice != null ? lastPrice - objectivePrice : null;
  const distancePct = entryDistancePct ?? (distancePts != null && objectivePrice ? Math.abs(distancePts) / Math.abs(objectivePrice) : null);

  const progressPercent = Math.round(progress * 100);
  const distanceLabel = distancePts != null ? `${distancePts >= 0 ? "+" : "-"}${formatPrice(Math.abs(distancePts), pricePrecision)} pts` : "—";
  const distancePctLabel = distancePct != null ? `${Math.round(distancePct * 1000) / 10}%` : "—";
  const bandLabel =
    bandLow != null && bandHigh != null
      ? `${formatPrice(bandLow, pricePrecision)} – ${formatPrice(bandHigh, pricePrecision)}`
      : "Band unavailable";

  const whyTokens = Array.isArray(raw.why) ? raw.why.filter((entry): entry is string => typeof entry === "string") : [];

  return (
    <div className="flex flex-col gap-2 rounded-lg border border-slate-700/60 bg-slate-900/40 p-4">
      <div className="flex items-center justify-between text-xs text-slate-200">
        <span className="font-semibold uppercase tracking-wide">Objective</span>
        <span className="text-slate-300">{friendlyState(raw.state)}</span>
      </div>
      <div className="flex items-center justify-between text-xs text-slate-300">
        <span>{raw.timeframe ?? "—"}</span>
        <span>Progress {progressPercent}%</span>
      </div>
      <div className="h-2 w-full rounded-full bg-slate-800/80">
        <div
          className={clsx(
            "h-full rounded-full transition-[width]",
            progress >= 1 ? "bg-emerald-400" : progress >= 0.75 ? "bg-amber-400" : "bg-sky-400",
          )}
          style={{ width: `${Math.max(0, Math.min(100, progress * 100))}%` }}
        />
      </div>
      <div className="grid grid-cols-2 gap-2 text-[11px] text-slate-300">
        <div>
          <div className="text-slate-400/70">Objective</div>
          <div className="font-semibold text-slate-100">{formatPrice(objectivePrice, pricePrecision)}</div>
        </div>
        <div>
          <div className="text-slate-400/70">Band</div>
          <div className="font-semibold text-slate-100">{bandLabel}</div>
        </div>
        <div>
          <div className="text-slate-400/70">Distance</div>
          <div className="font-semibold text-slate-100">
            {distanceLabel}
            <span className="ml-1 text-slate-300/70">({distancePctLabel})</span>
          </div>
        </div>
        <div>
          <div className="text-slate-400/70">Tick Size</div>
          <div className="font-semibold text-slate-100">{tickSize != null ? formatPrice(tickSize, pricePrecision) : "—"}</div>
        </div>
      </div>
      {whyTokens.length > 0 ? (
        <div className="flex flex-wrap gap-2">
          {whyTokens.map((token) => (
            <span
              key={token}
              className="rounded-full border border-slate-700/50 bg-slate-800/60 px-2 py-0.5 text-[10px] uppercase tracking-wide text-slate-200"
            >
              {token}
            </span>
          ))}
        </div>
      ) : null}
    </div>
  );
}
