"use client";

import clsx from "clsx";

type ConfidenceComponent = {
  label: string;
  value?: number | string | null;
  tooltip?: string | null;
};

type ConfidenceSurfaceProps = {
  confidence?: number | null;
  components?: ConfidenceComponent[];
};

export default function ConfidenceSurface({ confidence, components }: ConfidenceSurfaceProps) {
  const score = typeof confidence === "number" && Number.isFinite(confidence) ? Math.max(0, Math.min(confidence, 1)) : null;
  const scorePercent = score != null ? Math.round(score * 100) : null;
  const bandColor =
    score == null
      ? "from-neutral-700/60 to-neutral-800/40"
      : score >= 0.7
        ? "from-emerald-500/30 to-emerald-500/10"
        : score >= 0.4
          ? "from-amber-500/30 to-amber-500/10"
          : "from-rose-500/30 to-rose-500/10";

  return (
    <section className="space-y-4">
      <div
        className={clsx(
          "relative overflow-hidden rounded-2xl border border-neutral-800/80 bg-neutral-950/60 p-5 shadow-md shadow-black/40",
          "bg-gradient-to-br",
          bandColor,
        )}
      >
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-xs font-semibold uppercase tracking-[0.3em] text-neutral-300">Confidence</h3>
            <p className="mt-2 text-3xl font-semibold text-white">{scorePercent != null ? `${scorePercent}%` : "Unknown"}</p>
          </div>
          <div className="flex h-12 w-12 items-center justify-center rounded-full border border-neutral-700 bg-neutral-900/80 text-sm font-semibold text-neutral-300">
            {score != null ? score.toFixed(2) : "—"}
          </div>
        </div>
        <p className="mt-3 text-xs text-neutral-300/80">
          Confidence band blends multi-timeframe bias, volatility regime, and liquidity context. Use it to size and pace execution.
        </p>
      </div>

      {components && components.length ? (
        <div className="flex flex-wrap gap-2">
          {components.map((component) => (
            <span
              key={component.label}
              title={component.tooltip ?? undefined}
              className="rounded-full border border-neutral-800/80 bg-neutral-900/80 px-3 py-1 text-[0.7rem] uppercase tracking-[0.25em] text-neutral-200"
            >
              {component.label}
              {component.value != null && component.value !== "" ? (
                <span className="ml-2 text-neutral-400">{String(component.value)}</span>
              ) : null}
            </span>
          ))}
        </div>
      ) : null}
    </section>
  );
}
