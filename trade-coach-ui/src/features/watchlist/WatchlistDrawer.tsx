"use client";

import Link from "next/link";
import { useEffect } from "react";
import type { WatchlistItem } from "@/store/useStore";

type WatchlistDrawerProps = {
  open: boolean;
  onClose: () => void;
  items: WatchlistItem[];
  status: "idle" | "loading" | "ready" | "error";
  error: string | null;
  onSelect: (planUrl: string) => void;
};

export function WatchlistDrawer({ open, onClose, items, status, error, onSelect }: WatchlistDrawerProps) {
  useEffect(() => {
    if (open && typeof window !== "undefined" && "vibrate" in navigator) {
      navigator.vibrate?.(10);
    }
  }, [open]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape" && open) {
        onClose();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  return (
    <div
      className={`pointer-events-none fixed inset-0 z-50 transition ${open ? "" : "delay-150"}`}
      aria-hidden={!open}
    >
      <div
        className={`absolute inset-0 bg-black/50 transition-opacity ${open ? "pointer-events-auto opacity-100" : "opacity-0"}`}
        onClick={onClose}
      />
      <div
        className={`pointer-events-auto absolute inset-x-0 bottom-0 transform-gpu rounded-t-3xl border border-[color:var(--tc-border-strong)] bg-[color:var(--tc-surface-primary)]/98 p-4 shadow-2xl transition-transform duration-250 ease-out ${open ? "translate-y-0" : "translate-y-full"}`}
        role="dialog"
        aria-modal="true"
      >
        <div className="mb-3 flex items-center justify-between">
          <div>
            <p className="text-[0.65rem] uppercase tracking-[0.28em] text-[color:var(--tc-neutral-500)]">Watchlist</p>
            <h2 className="text-lg font-semibold text-[color:var(--tc-neutral-50)]">Actionable radar</h2>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-full border border-[color:var(--tc-border-subtle)] px-3 py-1 text-xs font-semibold uppercase tracking-[0.18em] text-[color:var(--tc-neutral-300)]"
          >
            Close
          </button>
        </div>
        {error ? <p className="mb-2 rounded-md border border-[color:var(--tc-chip-red-border)] bg-[color:var(--tc-chip-red-surface)] px-3 py-2 text-xs text-[color:var(--tc-red-200)]">{error}</p> : null}
        <div className="max-h-[60vh] overflow-y-auto pr-1">
          {status === "loading" && items.length === 0 ? (
            <div className="py-12 text-center text-sm text-[color:var(--tc-neutral-400)]">Fetching scan…</div>
          ) : null}
          {items.length === 0 && status === "ready" ? (
            <div className="py-10 text-center text-sm text-[color:var(--tc-neutral-400)]">No actionable setups detected.</div>
          ) : null}
          <ul className="flex flex-col gap-3">
            {items.map((item) => (
              <li
                key={item.plan_id}
                data-testid="watchlist-drawer-item"
                data-plan-id={item.plan_id}
                data-actionable={item.actionable_soon ? "soon" : "watching"}
                data-distance={item.entry_distance_pct != null ? item.entry_distance_pct.toString() : ""}
              >
                <button
                  type="button"
                  onClick={() => onSelect(item.plan_url)}
                  className="w-full rounded-2xl border border-[color:var(--tc-border-subtle)] bg-[color:var(--tc-surface-muted)] px-4 py-3 text-left"
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="text-sm font-semibold text-[color:var(--tc-neutral-50)]">{item.symbol}</div>
                      <div className="text-[0.68rem] uppercase tracking-[0.28em] text-[color:var(--tc-neutral-500)]">{item.style || "intraday"}</div>
                    </div>
                    <span className={`rounded-full px-3 py-1 text-[0.68rem] font-semibold uppercase tracking-[0.18em] ${item.actionable_soon ? "border border-[color:var(--tc-chip-emerald-border)] bg-[color:var(--tc-chip-emerald-surface)] text-[color:var(--tc-emerald-200)]" : "border border-[color:var(--tc-chip-neutral-border)] bg-[color:var(--tc-chip-neutral-surface)] text-[color:var(--tc-neutral-200)]"}`}>
                      {item.actionable_soon ? "Soon" : "Watching"}
                    </span>
                  </div>
                  {item.chart_url ? (
                    <div className="mt-3 text-[0.68rem] text-[color:var(--tc-neutral-400)]">
                      <Link
                        href={item.chart_url}
                        target="_blank"
                        rel="noreferrer"
                        className="text-[color:var(--tc-emerald-300)] underline"
                        onClick={(event) => event.stopPropagation()}
                      >
                        Canonical chart ↗
                      </Link>
                    </div>
                  ) : null}
                </button>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}
