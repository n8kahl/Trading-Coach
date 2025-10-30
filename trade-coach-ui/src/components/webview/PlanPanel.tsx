"use client";

import clsx from "clsx";
import { useMemo } from "react";
import ConfidenceSurface from "./ConfidenceSurface";
import ConfluenceOverlay from "./ConfluenceOverlay";
import type { SupportingLevel } from "@/lib/chart";
import type { Badge, PlanSnapshot, StructuredPlan, TargetMetaEntry } from "@/lib/types";

type PlanPanelProps = {
  plan: PlanSnapshot["plan"];
  structured?: StructuredPlan | null;
  badges?: Badge[];
  confidence?: number | null;
  supportingLevels: SupportingLevel[];
  highlightedLevel: SupportingLevel | null;
  onSelectLevel: (level: SupportingLevel | null) => void;
  theme?: "dark" | "light";
  targetsAwaiting?: boolean;
};

export default function PlanPanel({
  plan,
  structured,
  badges,
  confidence,
  supportingLevels,
  highlightedLevel,
  onSelectLevel,
  theme = "dark",
  targetsAwaiting = false,
}: PlanPanelProps) {
  const isLight = theme === "light";
  const entry = structured?.entry?.level ?? plan.entry ?? null;
  const stop = plan.stop ?? structured?.stop ?? null;
  const rr = plan.rr_to_t1 ?? null;
  const targets = useMemo(() => {
    if (Array.isArray(plan.targets) && plan.targets.length) return plan.targets;
    return structured?.targets ?? [];
  }, [plan.targets, structured?.targets]);

  const targetMeta = useMemo(() => (Array.isArray(plan.target_meta) ? (plan.target_meta as TargetMetaEntry[]) : []), [plan.target_meta]);
  const targetBySnap = useMemo(() => {
    const map = new Map<string, TargetMetaEntry[]>();
    targetMeta.forEach((entry) => {
      if (!entry.snap_tag) return;
      const key = entry.snap_tag.toString().toLowerCase();
      const existing = map.get(key) ?? [];
      existing.push(entry);
      map.set(key, existing);
    });
    return map;
  }, [targetMeta]);

  const confluence = Array.isArray(plan.confluence) ? (plan.confluence as string[]) : [];

  const confidenceComponents = useMemo(() => {
    const components = [];
    const expectedMove = getNumber(plan?.data_quality?.expected_move);
    if (expectedMove != null) components.push({ label: "ATR", value: expectedMove.toFixed(2) });
    const remainingAtr = getNumber(plan?.data_quality?.remaining_atr);
    if (remainingAtr != null) components.push({ label: "Remaining ATR", value: remainingAtr.toFixed(2) });
    if (plan.htf && typeof plan.htf === "object" && plan.htf.bias) components.push({ label: "HTF Bias", value: String(plan.htf.bias) });
    if (plan.volatility_regime && typeof plan.volatility_regime === "object" && plan.volatility_regime.label) {
      components.push({ label: "Volatility", value: String(plan.volatility_regime.label) });
    }
    if (components.length === 0 && confluence.length) {
      components.push(...confluence.slice(0, 4).map((item) => ({ label: item.toUpperCase(), value: null })));
    }
    return components;
  }, [plan?.data_quality, plan.htf, plan.volatility_regime, confluence]);

  const highlightedRationale = useMemo(() => {
    if (!highlightedLevel) return [];
    const key = highlightedLevel.label.toLowerCase();
    const rationales: string[] = [];
    const matchingTargetMeta = Array.from(targetBySnap.entries()).filter(([snap]) => key.includes(snap) || snap.includes(key));
    matchingTargetMeta.forEach(([, entries]) => {
      entries.forEach((entry) => {
        const parts: string[] = [];
        if (entry.label) parts.push(entry.label);
        if (entry.rr_multiple != null) parts.push(`R:R ${entry.rr_multiple.toFixed(2)}`);
        if (entry.prob_touch_calibrated != null) parts.push(`${Math.round(entry.prob_touch_calibrated * 100)}% path`);
        if (entry.em_fraction != null) parts.push(`${Math.round(entry.em_fraction * 100)}% EM`);
        if (parts.length) rationales.push(parts.join(" · "));
      });
    });
    confluence.forEach((item) => {
      const normalized = item.toLowerCase();
      if (normalized.includes(key) && !rationales.includes(item)) rationales.push(item);
    });
    return rationales;
  }, [highlightedLevel, targetBySnap, confluence]);

  const highlightedTargetTag = useMemo(() => {
    if (!highlightedLevel) return null;
    const key = highlightedLevel.label.toLowerCase();
    for (const [snap, entries] of targetBySnap.entries()) {
      if (!key.includes(snap) && !snap.includes(key)) continue;
      const primary = entries[0];
      if (primary?.label) return primary.label;
    }
    return null;
  }, [highlightedLevel, targetBySnap]);

  return (
    <div className="flex h-full flex-col gap-6">
      <header className="space-y-3">
        <div className="flex flex-wrap items-center gap-2">
          <span
            className={clsx(
              "rounded-full px-3 py-1 text-xs font-semibold uppercase tracking-[0.3em]",
              isLight ? "border border-emerald-500/40 bg-emerald-400/15 text-emerald-700" : "border border-emerald-500/40 bg-emerald-500/15 text-emerald-200",
            )}
          >
            Plan Summary
          </span>
          {badges?.slice(0, 3).map((badge) => (
            <span
              key={`${badge.kind}-${badge.label}`}
              className={clsx(
                "rounded-full border px-3 py-1 text-xs uppercase tracking-[0.25em]",
                isLight ? "border-slate-200 bg-white text-slate-600" : "border-neutral-700/70 bg-neutral-900/70 text-neutral-300",
              )}
            >
              {badge.label}
            </span>
          ))}
        </div>
        <div className={clsx("grid grid-cols-2 gap-3 text-sm", isLight ? "text-slate-700" : "text-neutral-200")}>
          <Metric label="Entry" value={entry != null ? entry.toFixed(2) : "—"} accent={isLight ? "text-emerald-600" : "text-emerald-300"} theme={theme} />
          <Metric label="Stop" value={stop != null ? stop.toFixed(2) : "—"} accent={isLight ? "text-rose-600" : "text-rose-300"} theme={theme} />
          <Metric label="R:R to TP1" value={rr != null ? rr.toFixed(2) : "—"} theme={theme} />
          <Metric label="Targets" value={targets.length ? `${targets.length}` : "0"} theme={theme} />
        </div>
      </header>

      <ConfidenceSurface confidence={confidence} components={confidenceComponents} theme={theme} />

      <section className="space-y-3">
        <h3 className={clsx("text-xs font-semibold uppercase tracking-[0.3em]", isLight ? "text-slate-500" : "text-neutral-400")}>Targets</h3>
        {targetsAwaiting ? (
          <div
            className={clsx(
              "rounded-xl border px-3 py-2 text-xs uppercase tracking-[0.25em]",
              isLight ? "border-amber-200/80 bg-amber-100/60 text-amber-700" : "border-amber-400/40 bg-amber-400/10 text-amber-200",
            )}
          >
            Targets unavailable (awaiting updates)
          </div>
        ) : null}
        <ul className={clsx("space-y-2 text-sm", isLight ? "text-slate-700" : "text-neutral-200")}>
          {targets.length === 0 ? (
            <li
              className={clsx(
                "rounded-xl border px-3 py-3",
                isLight ? "border-slate-200 bg-white text-slate-500" : "border-neutral-800/70 bg-neutral-900/60 text-neutral-400",
              )}
            >
              No targets published.
            </li>
          ) : (
            targets.map((target, index) => {
              const meta = targetMeta[index];
              return (
                <li
                  key={`${target}-${index}`}
                  className={clsx(
                    "flex items-center justify-between rounded-xl border px-4 py-3",
                    isLight ? "border-slate-200 bg-white" : "border-neutral-800/70 bg-neutral-900/60",
                  )}
                >
                  <div>
                    <span className={clsx("text-xs uppercase tracking-[0.25em]", isLight ? "text-slate-500" : "text-neutral-400")}>
                      TP{index + 1}
                    </span>
                    {meta?.snap_tag ? <span className={clsx("ml-2 text-xs", isLight ? "text-slate-400" : "text-neutral-500")}>{meta.snap_tag}</span> : null}
                  </div>
                  <div className="text-right">
                    <div className={clsx("font-semibold", isLight ? "text-emerald-600" : "text-emerald-300")}>{target.toFixed(2)}</div>
                    {meta?.prob_touch_calibrated != null ? (
                      <div className={clsx("text-xs", isLight ? "text-slate-400" : "text-neutral-400")}>{Math.round(meta.prob_touch_calibrated * 100)}% touch</div>
                    ) : null}
                  </div>
                </li>
              );
            })
          )}
        </ul>
      </section>

      <section className="space-y-3">
        <div className="flex items-center justify-between">
          <h3 className={clsx("text-xs font-semibold uppercase tracking-[0.3em]", isLight ? "text-slate-500" : "text-neutral-400")}>Supporting levels</h3>
          <span className={clsx("text-[0.65rem] uppercase tracking-[0.25em]", isLight ? "text-slate-400" : "text-neutral-500")}>
            {supportingLevels.length} tracked
          </span>
        </div>
        <ul
          className={clsx(
            "max-h-48 space-y-1 overflow-auto rounded-2xl border p-2",
            isLight ? "border-slate-200 bg-white" : "border-neutral-800/60 bg-neutral-900/50",
          )}
        >
          {supportingLevels.length === 0 ? (
            <li className={clsx("rounded-xl px-3 py-2 text-sm", isLight ? "text-slate-500" : "text-neutral-400")}>No supporting levels detected.</li>
          ) : (
            supportingLevels.map((level) => {
              const active = highlightedLevel?.label === level.label && highlightedLevel?.price === level.price;
              return (
                <li key={`${level.label}-${level.price}`} className="rounded-xl">
                  <button
                    type="button"
                    onMouseEnter={() => onSelectLevel(level)}
                    onFocus={() => onSelectLevel(level)}
                    onMouseLeave={() => onSelectLevel(null)}
                    onBlur={() => onSelectLevel(null)}
                    className={clsx(
                      "flex w-full items-center justify-between rounded-xl border px-3 py-2 text-sm transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400",
                      active
                        ? isLight
                          ? "border-emerald-500/60 bg-emerald-400/20 text-emerald-700"
                          : "border-emerald-500/60 bg-emerald-500/10 text-emerald-100"
                        : isLight
                          ? "border-transparent text-slate-700 hover:border-slate-200 hover:bg-slate-100"
                          : "border-transparent text-neutral-200 hover:border-neutral-700 hover:bg-neutral-900",
                    )}
                  >
                    <span className={clsx("text-left text-xs uppercase tracking-[0.25em]", isLight ? "text-slate-500" : "text-neutral-400")}>
                      {level.label}
                    </span>
                    <span className={clsx("text-right font-semibold", isLight ? "text-slate-800" : "text-neutral-100")}>
                      {level.price.toFixed(2)}
                    </span>
                  </button>
                </li>
              );
            })
          )}
        </ul>

        <ConfluenceOverlay level={highlightedLevel} rationale={highlightedRationale} targetTag={highlightedTargetTag} theme={theme} />
      </section>

      {plan.notes ? (
        <section
          className={clsx(
            "rounded-2xl border px-4 py-4 text-sm",
            isLight ? "border-slate-200 bg-white text-slate-700" : "border-neutral-800/70 bg-neutral-900/70 text-neutral-200",
          )}
        >
          <h3 className={clsx("text-xs font-semibold uppercase tracking-[0.3em]", isLight ? "text-slate-500" : "text-neutral-400")}>Coach note</h3>
          <p className={clsx("mt-2 leading-relaxed", isLight ? "text-slate-700" : "text-neutral-200")}>{plan.notes}</p>
        </section>
      ) : null}
    </div>
  );
}

function Metric({
  label,
  value,
  accent,
  theme,
}: {
  label: string;
  value: string;
  accent?: string;
  theme: "dark" | "light";
}) {
  const isLight = theme === "light";
  return (
    <div>
      <span className={clsx("text-[0.68rem] uppercase tracking-[0.3em]", isLight ? "text-slate-500" : "text-neutral-500")}>{label}</span>
      <div className={clsx("mt-1 text-lg font-semibold", accent ?? (isLight ? "text-slate-800" : "text-white"))}>{value}</div>
    </div>
  );
}

function getNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const num = Number.parseFloat(value);
    if (Number.isFinite(num)) return num;
  }
  return null;
}
