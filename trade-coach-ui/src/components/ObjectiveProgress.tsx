"use client";

import clsx from "clsx";

export type NextObjectiveMeta = {
  state?: string;
  why?: string[];
  objective_price?: number | null;
  band?: { low: number; high: number } | null;
  timeframe?: string;
  progress?: number;
};

type ObjectiveProgressProps = {
  meta?: NextObjectiveMeta | null;
  className?: string;
};

function clamp(value: number | null | undefined): number {
  if (!Number.isFinite(value ?? NaN)) return 0;
  return Math.max(0, Math.min(1, value as number));
}

export default function ObjectiveProgress({ meta, className }: ObjectiveProgressProps) {
  if (!meta) {
    return null;
  }

  const progress = clamp(meta.progress);
  const percent = Math.round(progress * 100);
  const timeframe = meta.timeframe?.toUpperCase();

  return (
    <div
      className={clsx(
        "flex min-w-[220px] flex-col gap-2 rounded-xl border border-neutral-900/70 bg-neutral-950/70 px-3 py-2",
        className,
      )}
    >
      <div className="flex items-center justify-between text-[10px] font-semibold uppercase tracking-[0.24em] text-neutral-200">
        <span>
          Next Objective
          {timeframe ? (
            <span className="text-neutral-400"> • {timeframe}</span>
          ) : null}
        </span>
        <span className="text-neutral-400">{percent}%</span>
      </div>
      <div className="h-1.5 overflow-hidden rounded-full bg-neutral-900/80">
        <span
          className="block h-full origin-left rounded-full bg-emerald-400 transition-transform duration-300"
          style={{ transform: `scaleX(${progress})` }}
        />
      </div>
      {meta.state ? (
        <div className="text-[9px] uppercase tracking-[0.22em] text-neutral-500">{meta.state}</div>
      ) : null}
    </div>
  );
}
