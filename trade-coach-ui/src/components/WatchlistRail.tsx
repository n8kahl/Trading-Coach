"use client";

import clsx from "clsx";
import Link from "next/link";

export type WatchItem = {
  planId: string;
  symbol: string;
  actionableSoon?: boolean | null;
  entryDistancePct?: number | null;
  barsToTrigger?: number | null;
  bias?: "long" | "short" | null;
};

type WatchlistRailProps = {
  items?: WatchItem[] | null;
  className?: string;
};

function sortItems(items: WatchItem[]): WatchItem[] {
  return [...items].sort((a, b) => {
    const aSoon = a.actionableSoon ? 0 : 1;
    const bSoon = b.actionableSoon ? 0 : 1;
    if (aSoon !== bSoon) return aSoon - bSoon;

    const aDist = Number.isFinite(a.entryDistancePct) ? (a.entryDistancePct as number) : 9;
    const bDist = Number.isFinite(b.entryDistancePct) ? (b.entryDistancePct as number) : 9;
    if (aDist !== bDist) return aDist - bDist;

    return a.symbol.localeCompare(b.symbol);
  });
}

function formatDistance(value: number | null | undefined): string {
  if (!Number.isFinite(value ?? NaN)) return "—";
  return `${Math.round((value as number) * 10) / 10}%`;
}

function formatBias(bias: WatchItem["bias"]): string | null {
  if (!bias) return null;
  return bias === "long" ? "Long Bias" : "Short Bias";
}

export default function WatchlistRail({ items = [], className }: WatchlistRailProps) {
  if (!items.length) {
    return (
      <div
        className={clsx(
          "space-y-3 rounded-xl border border-neutral-900/70 bg-neutral-950/60 p-4 text-sm text-neutral-400",
          className,
        )}
      >
        <div className="text-xs font-semibold uppercase tracking-[0.24em] text-neutral-500">Watchlist</div>
        <p className="text-[11px] text-neutral-500">No candidates available.</p>
      </div>
    );
  }

  const sorted = sortItems(items);

  return (
    <div className={clsx("space-y-3 p-4", className)}>
      <div className="text-xs font-semibold uppercase tracking-[0.24em] text-neutral-500">Watchlist</div>
      <ul className="space-y-2 text-sm">
        {sorted.map((item) => {
          const biasLabel = formatBias(item.bias);
          const distance = formatDistance(item.entryDistancePct);
          const soonLabel = item.actionableSoon ? "Soon" : "Later";
          return (
            <li
              key={item.planId}
              data-testid="watchlist-item"
              className="flex flex-col gap-2 rounded-lg border border-neutral-900/70 bg-neutral-950/70 p-3 text-[11px] uppercase tracking-[0.22em] text-neutral-300"
            >
              <div className="flex items-center gap-2">
                <span
                  data-testid="wl-symbol"
                  className="inline-flex items-center gap-1 rounded-full border border-emerald-500/60 bg-emerald-500/15 px-2 py-0.5 text-[10px] font-semibold text-emerald-100"
                >
                  {item.symbol.toUpperCase()}
                </span>
                {biasLabel ? (
                  <span className="rounded-full border border-neutral-800/60 bg-neutral-900/70 px-2 py-0.5 text-[9px] text-neutral-300">
                    {biasLabel}
                  </span>
                ) : null}
              </div>
              <div className="flex items-center justify-between gap-2 text-[10px]">
                <span data-testid="wl-soon" className="font-semibold text-neutral-200">
                  {soonLabel}
                </span>
                <span data-testid="wl-distance-pct" className="text-neutral-400">
                  {distance}
                </span>
              </div>
              <div className="flex items-center justify-between text-[9px] text-neutral-500">
                <span>{Number.isFinite(item.barsToTrigger ?? NaN) ? `${item.barsToTrigger} bars out` : "—"}</span>
                <Link
                  href={`/plan/${encodeURIComponent(item.planId)}`}
                  className="rounded-full border border-emerald-500/50 bg-emerald-500/10 px-2 py-1 text-[9px] font-semibold uppercase tracking-[0.24em] text-emerald-100 transition hover:border-emerald-400 hover:text-emerald-50"
                >
                  Open
                </Link>
              </div>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
