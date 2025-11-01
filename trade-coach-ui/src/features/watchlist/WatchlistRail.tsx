"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import React from "react";
import type { WatchlistItem } from "@/store/useStore";

type WatchlistRailProps = {
  items: WatchlistItem[];
  status: "idle" | "loading" | "ready" | "error";
  error: string | null;
  lastUpdated: number | null;
  onRefresh: () => void;
};

function formatPercent(value: number | null): string {
  if (value == null || Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(Math.abs(value) < 0.01 ? 2 : 1)}%`;
}

function formatAtr(value: number | null): string {
  if (value == null || Number.isNaN(value)) return "—";
  return `${value.toFixed(2)} ATR`;
}

function formatBars(value: number | null): string {
  if (value == null || Number.isNaN(value)) return "—";
  return value <= 1 ? "≤1 bar" : `${Math.round(value)} bars`;
}

function formatRelative(ts: number | null): string {
  if (!ts) return "—";
  const diffMs = Date.now() - ts;
  const diffMinutes = Math.round(diffMs / 60000);
  if (diffMinutes <= 1) return "just now";
  if (diffMinutes < 60) return `${diffMinutes} min ago`;
  const diffHours = Math.round(diffMinutes / 60);
  if (diffHours < 24) return `${diffHours} hr ago`;
  const diffDays = Math.round(diffHours / 24);
  return `${diffDays} day${diffDays === 1 ? "" : "s"} ago`;
}

export function WatchlistRail({ items, status, error, lastUpdated, onRefresh }: WatchlistRailProps) {
  const router = useRouter();

  return (
    <div className="flex h-full w-80 flex-col gap-4 rounded-3xl border border-[color:var(--tc-border-subtle)] bg-[color:var(--tc-surface-raised)]/95 p-4 backdrop-blur-lg">
      <header className="flex items-center justify-between">
        <div>
          <p className="text-[0.65rem] uppercase tracking-[0.28em] text-[color:var(--tc-neutral-400)]">Watchlist</p>
          <h2 className="mt-1 text-lg font-semibold text-[color:var(--tc-neutral-50)]">Actionable radar</h2>
        </div>
        <button
          type="button"
          onClick={onRefresh}
          className="rounded-full border border-[color:var(--tc-border-subtle)] px-3 py-1 text-xs font-semibold uppercase tracking-[0.18em] text-[color:var(--tc-neutral-300)] transition hover:border-[color:var(--tc-emerald-400)] hover:text-[color:var(--tc-emerald-300)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[color:var(--tc-emerald-400)]"
        >
          Refresh
        </button>
      </header>

      {lastUpdated ? (
        <p className="text-[0.65rem] uppercase tracking-[0.24em] text-[color:var(--tc-neutral-500)]">
          Updated {formatRelative(lastUpdated)}
        </p>
      ) : (
        <p className="text-[0.65rem] uppercase tracking-[0.24em] text-[color:var(--tc-neutral-500)]">
          {status === "loading" ? "Loading…" : "Awaiting scan"}
        </p>
      )}

      {error ? <p className="rounded-md border border-[color:var(--tc-chip-red-border)] bg-[color:var(--tc-chip-red-surface)] px-3 py-2 text-xs text-[color:var(--tc-red-300)]">{error}</p> : null}

      <div className="flex-1 overflow-y-auto pr-1">
        {status === "loading" && items.length === 0 ? (
          <div className="flex h-full items-center justify-center text-sm text-[color:var(--tc-neutral-500)]">Fetching scan…</div>
        ) : null}
        {items.length === 0 && status === "ready" ? (
          <div className="flex h-full flex-col items-center justify-center gap-3 text-center text-sm text-[color:var(--tc-neutral-400)]">
            <span>No actionable setups right now.</span>
            <span className="text-xs text-[color:var(--tc-neutral-500)]">Watchlist auto-refreshes every minute.</span>
          </div>
        ) : null}
        <ul className="flex flex-col gap-3">
          {items.map((item) => {
            const actionableClass = item.actionable_soon
              ? "border-[color:var(--tc-chip-emerald-border)] bg-[color:var(--tc-chip-emerald-surface)] text-[color:var(--tc-emerald-200)]"
              : "border-[color:var(--tc-chip-neutral-border)] bg-[color:var(--tc-chip-neutral-surface)] text-[color:var(--tc-neutral-200)]";
            return (
              <li
                key={item.plan_id}
                data-testid="watchlist-item"
                data-plan-id={item.plan_id}
                data-actionable={item.actionable_soon ? "soon" : "watching"}
                data-distance={item.entry_distance_pct != null ? item.entry_distance_pct.toString() : ""}
              >
                <button
                  type="button"
                  onClick={() => {
                    router.push(item.plan_url);
                    if (typeof window !== "undefined" && "vibrate" in navigator) {
                      navigator.vibrate?.(8);
                    }
                  }}
                  className="group flex w-full flex-col gap-3 rounded-2xl border border-[color:var(--tc-border-subtle)] bg-[color:var(--tc-surface-muted)] px-4 py-3 text-left transition hover:border-[color:var(--tc-emerald-400)] hover:bg-[color:var(--tc-surface-primary)] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[color:var(--tc-emerald-400)]"
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="text-sm font-semibold text-[color:var(--tc-neutral-50)]">{item.symbol}</div>
                      <div className="text-[0.68rem] uppercase tracking-[0.3em] text-[color:var(--tc-neutral-500)]">
                        {item.style || "intraday"}
                      </div>
                    </div>
                    <span className={`rounded-full px-3 py-1 text-[0.68rem] font-semibold uppercase tracking-[0.18em] ${actionableClass}`}>
                      {item.actionable_soon ? "Soon" : "Watching"}
                    </span>
                  </div>
                  <div className="flex flex-wrap gap-2 text-xs text-[color:var(--tc-neutral-300)]">
                    <span className="rounded-full border border-[color:var(--tc-border-subtle)] px-2 py-1 text-[0.68rem] uppercase tracking-[0.18em] text-[color:var(--tc-neutral-200)]">
                      Dist {formatPercent(item.entry_distance_pct)}
                    </span>
                    <span className="rounded-full border border-[color:var(--tc-border-subtle)] px-2 py-1 text-[0.68rem] uppercase tracking-[0.18em] text-[color:var(--tc-neutral-200)]">
                      {formatAtr(item.entry_distance_atr)}
                    </span>
                    <span className="rounded-full border border-[color:var(--tc-border-subtle)] px-2 py-1 text-[0.68rem] uppercase tracking-[0.18em] text-[color:var(--tc-neutral-200)]">
                      {formatBars(item.bars_to_trigger)}
                    </span>
                  </div>
                  {item.chart_url ? (
                    <div className="flex items-center justify-between text-[0.68rem] text-[color:var(--tc-neutral-400)]">
                      <span>Canonical chart</span>
                      <Link
                        href={item.chart_url}
                        target="_blank"
                        rel="noreferrer"
                        className="text-[color:var(--tc-emerald-300)] transition hover:text-[color:var(--tc-emerald-200)]"
                        onClick={(event) => {
                          event.stopPropagation();
                        }}
                      >
                        Open ↗
                      </Link>
                    </div>
                  ) : (
                    <div className="text-[0.68rem] text-[color:var(--tc-neutral-500)]">Chart link pending…</div>
                  )}
                </button>
              </li>
            );
          })}
        </ul>
      </div>
    </div>
  );
}
