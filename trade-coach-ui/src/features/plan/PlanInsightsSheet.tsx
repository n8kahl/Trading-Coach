"use client";

import clsx from "clsx";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { PlanLayers, PlanSnapshot } from "@/lib/types";
import { resolvePlanEntry, resolvePlanStop, resolvePlanTargets } from "@/lib/plan/coach";

type PlanInsightsSheetProps = {
  plan: PlanSnapshot["plan"];
  layers: PlanLayers | null;
  trailingStop: number | null;
  open: boolean;
  onOpenChange(next: boolean): void;
};

type TabKey = "targets" | "risk" | "confluence";

const TABS: Array<{ key: TabKey; label: string }> = [
  { key: "targets", label: "Targets" },
  { key: "risk", label: "Risk" },
  { key: "confluence", label: "Confluence" },
];

function formatPrice(value: number | null | undefined, precision = 2): string {
  if (value == null || Number.isNaN(value)) return "—";
  return value.toLocaleString("en-US", {
    minimumFractionDigits: precision,
    maximumFractionDigits: precision,
  });
}

function clamp(value: number, min = 0, max = 1): number {
  if (Number.isNaN(value)) return min;
  return Math.min(max, Math.max(min, value));
}

function extractConfluence(plan: PlanSnapshot["plan"], layers: PlanLayers | null): string[] {
  const tokens = new Set<string>();
  const meta = layers?.meta;
  if (meta && typeof meta === "object") {
    const objective = (meta as Record<string, unknown>).next_objective;
    if (objective && typeof objective === "object") {
      const why = (objective as Record<string, unknown>).why;
      if (Array.isArray(why)) {
        why.forEach((token) => {
          if (typeof token === "string" && token.trim()) {
            tokens.add(token.trim());
          }
        });
      }
    }
  }
  const badges = Array.isArray(plan.badges) ? plan.badges : [];
  badges.forEach((badge) => {
    if (badge && typeof badge.label === "string" && badge.label.trim()) {
      tokens.add(badge.label.trim());
    }
  });
  const structured = plan.structured_plan;
  if (structured && Array.isArray(structured.badges)) {
    structured.badges.forEach((label) => {
      if (typeof label === "string" && label.trim()) {
        tokens.add(label.trim());
      }
    });
  }
  return Array.from(tokens);
}

export function PlanInsightsSheet({ plan, layers, trailingStop, open, onOpenChange }: PlanInsightsSheetProps) {
  const [activeTabIndex, setActiveTabIndex] = useState(0);
  const activeTab = TABS[activeTabIndex];
  const touchStartRef = useRef<{ x: number; y: number } | null>(null);
  const touchActiveRef = useRef(false);

  useEffect(() => {
    if (!open) {
      setActiveTabIndex(0);
    }
  }, [open]);

  const precision = typeof layers?.precision === "number" && Number.isFinite(layers.precision) ? Math.max(0, layers.precision) : 2;
  const entry = useMemo(() => resolvePlanEntry(plan), [plan]);
  const stop = useMemo(() => resolvePlanStop(plan, trailingStop), [plan, trailingStop]);
  const targets = useMemo(() => resolvePlanTargets(plan), [plan]);

  const targetMeta = useMemo(
    () => (Array.isArray(plan.target_meta) ? (plan.target_meta as Array<Record<string, unknown>>) : []),
    [plan.target_meta],
  );

  const riskStats = useMemo(() => {
    const rr = typeof plan.rr_to_t1 === "number" ? plan.rr_to_t1 : null;
    const distance = entry != null && stop != null ? Math.abs(entry - stop) : null;
    const direction = entry != null && stop != null ? (entry > stop ? "long" : "short") : null;
    const trailing = typeof trailingStop === "number" ? trailingStop : null;
    return {
      rr,
      distance,
      direction,
      trailing,
      entry,
      stop,
    };
  }, [entry, stop, trailingStop, plan.rr_to_t1]);

  const confluenceTokens = useMemo(() => extractConfluence(plan, layers), [plan, layers]);

  const handleTouchStart = useCallback((event: React.TouchEvent<HTMLDivElement>) => {
    if (!open) return;
    const touch = event.touches[0];
    touchStartRef.current = { x: touch.clientX, y: touch.clientY };
    touchActiveRef.current = true;
  }, [open]);

  const handleTouchEnd = useCallback(
    (event: React.TouchEvent<HTMLDivElement>) => {
      if (!touchActiveRef.current || !touchStartRef.current) return;
      const touch = event.changedTouches[0];
      const start = touchStartRef.current;
      const dx = touch.clientX - start.x;
      const dy = touch.clientY - start.y;
      touchActiveRef.current = false;
      touchStartRef.current = null;
      const absDx = Math.abs(dx);
      const absDy = Math.abs(dy);
      const SWIPE_THRESHOLD = 40;
      if (absDx > absDy && absDx > SWIPE_THRESHOLD) {
        if (dx < 0 && activeTabIndex < TABS.length - 1) {
          setActiveTabIndex((index) => Math.min(index + 1, TABS.length - 1));
        } else if (dx > 0 && activeTabIndex > 0) {
          setActiveTabIndex((index) => Math.max(index - 1, 0));
        }
        return;
      }
      if (absDy > absDx && dy > SWIPE_THRESHOLD) {
        onOpenChange(false);
      }
    },
    [activeTabIndex, onOpenChange],
  );

  const handleBackdropClick = useCallback(() => {
    onOpenChange(false);
  }, [onOpenChange]);

  const handleTabSelect = useCallback((index: number) => {
    setActiveTabIndex(index);
  }, []);

  const renderTargets = () => {
    if (!targets.length) {
      return <p className="text-sm text-[color:var(--tc-neutral-400)]">No targets available.</p>;
    }
    return (
      <ul className="space-y-3">
        {targets.map((target, index) => {
          const meta = targetMeta[index] ?? {};
          const probability =
            typeof meta.prob_touch === "number"
              ? meta.prob_touch
              : typeof meta.prob_touch_calibrated === "number"
                ? meta.prob_touch_calibrated
                : typeof meta.prob_touch_raw === "number"
                  ? meta.prob_touch_raw
                  : null;
          const rr = typeof meta.rr_multiple === "number" ? meta.rr_multiple : null;
          const distancePts = entry != null ? target.price - entry : null;
          return (
            <li
              key={`${target.label}-${target.price}`}
              className="rounded-2xl border border-[color:var(--tc-border-subtle)] bg-[color:var(--tc-surface-primary)]/90 px-4 py-3"
            >
              <div className="flex items-center justify-between text-sm text-[color:var(--tc-neutral-200)]">
                <span className="font-semibold text-[color:var(--tc-neutral-50)]">{target.label}</span>
                <span>{formatPrice(target.price, precision)}</span>
              </div>
              <div className="mt-2 flex flex-wrap items-center gap-3 text-[0.7rem] uppercase tracking-[0.18em] text-[color:var(--tc-neutral-400)]">
                {distancePts != null ? <span>Δ {formatPrice(distancePts, precision)} pts</span> : null}
                {probability != null ? <span>Prob {Math.round(clamp(probability, 0, 1) * 100)}%</span> : null}
                {rr != null ? <span>RR {rr.toFixed(2)}</span> : null}
              </div>
            </li>
          );
        })}
      </ul>
    );
  };

  const renderRisk = () => {
    return (
      <div className="space-y-3 text-sm text-[color:var(--tc-neutral-200)]">
        <div className="grid grid-cols-2 gap-3">
          <div className="rounded-xl border border-[color:var(--tc-border-subtle)] bg-[color:var(--tc-surface-primary)]/80 px-3 py-2">
            <div className="text-[0.6rem] uppercase tracking-[0.2em] text-[color:var(--tc-neutral-400)]">Entry</div>
            <div className="text-base font-semibold text-[color:var(--tc-neutral-50)]">{formatPrice(riskStats.entry, precision)}</div>
          </div>
          <div className="rounded-xl border border-[color:var(--tc-border-subtle)] bg-[color:var(--tc-surface-primary)]/80 px-3 py-2">
            <div className="text-[0.6rem] uppercase tracking-[0.2em] text-[color:var(--tc-neutral-400)]">Stop</div>
            <div className="text-base font-semibold text-[color:var(--tc-neutral-50)]">{formatPrice(riskStats.stop, precision)}</div>
          </div>
          <div className="rounded-xl border border-[color:var(--tc-border-subtle)] bg-[color:var(--tc-surface-primary)]/80 px-3 py-2">
            <div className="text-[0.6rem] uppercase tracking-[0.2em] text-[color:var(--tc-neutral-400)]">Distance</div>
            <div className="text-base font-semibold text-[color:var(--tc-neutral-50)]">
              {riskStats.distance != null ? `${formatPrice(riskStats.distance, precision)} pts` : "—"}
            </div>
          </div>
          <div className="rounded-xl border border-[color:var(--tc-border-subtle)] bg-[color:var(--tc-surface-primary)]/80 px-3 py-2">
            <div className="text-[0.6rem] uppercase tracking-[0.2em] text-[color:var(--tc-neutral-400)]">RR to TP1</div>
            <div className="text-base font-semibold text-[color:var(--tc-neutral-50)]">
              {riskStats.rr != null ? riskStats.rr.toFixed(2) : "—"}
            </div>
          </div>
        </div>
        <div className="rounded-xl border border-[color:var(--tc-border-subtle)] bg-[color:var(--tc-surface-primary)]/80 px-3 py-2 text-xs text-[color:var(--tc-neutral-300)]">
          <div className="font-semibold uppercase tracking-[0.22em] text-[color:var(--tc-neutral-400)]">Context</div>
          <ul className="mt-2 space-y-1">
            {riskStats.direction ? <li>Direction: {riskStats.direction.toUpperCase()}</li> : null}
            {riskStats.trailing != null ? <li>Trailing stop: {formatPrice(riskStats.trailing, precision)}</li> : null}
            {plan.session_state?.status ? <li>Session: {plan.session_state.status}</li> : null}
          </ul>
        </div>
      </div>
    );
  };

  const renderConfluence = () => {
    if (!confluenceTokens.length) {
      return <p className="text-sm text-[color:var(--tc-neutral-400)]">Confluence signals will appear as the plan updates.</p>;
    }
    return (
      <div className="flex flex-wrap gap-2">
        {confluenceTokens.map((token) => (
          <span
            key={token}
            className="rounded-full border border-[color:var(--tc-chip-neutral-border)] bg-[color:var(--tc-chip-neutral-surface)] px-3 py-1 text-[0.7rem] font-semibold uppercase tracking-[0.2em] text-[color:var(--tc-neutral-200)]"
          >
            {token}
          </span>
        ))}
      </div>
    );
  };

  const content = (() => {
    switch (activeTab.key) {
      case "targets":
        return renderTargets();
      case "risk":
        return renderRisk();
      case "confluence":
      default:
        return renderConfluence();
    }
  })();

  return (
    <div className={clsx("fixed inset-x-0 bottom-0 z-40 md:hidden", open ? "pointer-events-auto" : "pointer-events-none")}>
      <div
        className={clsx(
          "absolute inset-0 bg-black/50 transition-opacity",
          open ? "opacity-100" : "opacity-0",
        )}
        aria-hidden={!open}
        onClick={handleBackdropClick}
      />
      <div
        className={clsx(
          "relative z-10 mt-auto origin-bottom rounded-t-3xl border border-[color:var(--tc-border-strong)] bg-[color:var(--tc-surface-primary)]/95 px-5 pb-8 pt-4 shadow-2xl transition-transform",
          open ? "translate-y-0" : "translate-y-full",
        )}
        role="dialog"
        aria-modal="true"
        aria-label="Plan insights"
        onTouchStart={handleTouchStart}
        onTouchEnd={handleTouchEnd}
      >
        <div className="mx-auto mb-3 h-1.5 w-12 rounded-full bg-[color:var(--tc-border-subtle)]" />
        <div className="flex items-center justify-between">
          <div className="text-[0.7rem] uppercase tracking-[0.24em] text-[color:var(--tc-neutral-400)]">Plan Insights</div>
          <button
            type="button"
            onClick={() => onOpenChange(false)}
            className="rounded-full border border-[color:var(--tc-border-subtle)] px-3 py-1 text-[0.65rem] font-semibold uppercase tracking-[0.2em] text-[color:var(--tc-neutral-200)]"
          >
            Close
          </button>
        </div>
        <div className="mt-4 flex items-center gap-2">
          {TABS.map((tab, index) => (
            <button
              key={tab.key}
              type="button"
              onClick={() => handleTabSelect(index)}
              className={clsx(
                "flex-1 rounded-full border px-3 py-1 text-[0.7rem] font-semibold uppercase tracking-[0.22em] transition",
                index === activeTabIndex
                  ? "border-[color:var(--tc-chip-emerald-border)] bg-[color:var(--tc-chip-emerald-surface)] text-[color:var(--tc-emerald-200)]"
                  : "border-[color:var(--tc-border-subtle)] bg-[color:var(--tc-surface-muted)] text-[color:var(--tc-neutral-300)]",
              )}
            >
              {tab.label}
            </button>
          ))}
        </div>
        <div className="mt-5 min-h-[180px] space-y-4 text-[color:var(--tc-neutral-100)]">{content}</div>
      </div>
    </div>
  );
}

export default PlanInsightsSheet;
