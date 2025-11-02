"use client";

import * as React from "react";
import clsx from "clsx";

export type ConfluenceComponents = {
  atr?: number | null;
  vwap?: number | null;
  emas?: number | null;
  orderflow?: number | null;
  liquidity?: number | null;
  why?: string | null;
  confidence?: number | null;
  dataAge?: Date | null;
};

function clampScore(value: number | null | undefined) {
  if (!Number.isFinite(value as number)) return 0;
  const v = Number(value);
  const pct = v <= 1 ? v * 100 : v;
  return Math.max(0, Math.min(100, Math.round(pct)));
}

function fmtContribution(v: number | null | undefined) {
  if (!Number.isFinite(v as number)) return "+0";
  const scaled = Math.round(Number(v) * 100);
  const sign = scaled >= 0 ? "+" : "";
  return `${sign}${scaled}`;
}

const items: Array<{ key: keyof ConfluenceComponents; label: string }> = [
  { key: "atr", label: "ATR" },
  { key: "vwap", label: "VWAP" },
  { key: "emas", label: "EMAs" },
  { key: "orderflow", label: "Order-Flow" },
  { key: "liquidity", label: "Liquidity" },
];

export default function ConfluenceStrength({
  model,
  className,
}: {
  model: ConfluenceComponents;
  className?: string;
}) {
  const score = clampScore(model.confidence ?? 0);
  const why = model.why ?? "Model alignment across frames.";

  return (
    <div className={clsx("rounded-xl border border-neutral-800/60 bg-neutral-950/30 p-2", className)}>
      <div className="mb-1 flex items-center justify-between">
        <span className="text-[10px] font-semibold uppercase tracking-[0.28em] text-neutral-400">Confluence</span>
        <span
          className={clsx(
            "inline-flex h-6 w-6 items-center justify-center rounded-full text-[10px] font-semibold",
            score >= 70
              ? "border border-emerald-500/60 bg-emerald-500/10 text-emerald-200"
              : score >= 50
                ? "border border-amber-500/60 bg-amber-500/10 text-amber-200"
                : "border border-rose-500/60 bg-rose-500/10 text-rose-200",
          )}
          title={`Confidence ${score}`}
          aria-label={`Confidence ${score}`}
        >
          {score}
        </span>
      </div>

      <div className="flex flex-wrap items-center gap-1">
        {items.map(({ key, label }) => (
          <span
            key={String(key)}
            className="inline-flex items-center gap-1 rounded-full border border-neutral-800/60 bg-neutral-900/60 px-2 py-0.5 text-[10px] text-neutral-200"
            data-testid={`conf-${String(key)}`}
            title={label}
          >
            <span className="block h-1.5 w-1.5 rounded-full bg-emerald-400" aria-hidden />
            <span className="tabular-nums">
              {label} {fmtContribution((model as Record<string, unknown>)[key] as number | null | undefined)}
            </span>
          </span>
        ))}
      </div>

      {why ? <p className="mt-1 text-[10px] leading-snug text-neutral-500">{why}</p> : null}
    </div>
  );
}
