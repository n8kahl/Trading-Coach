"use client";

import clsx from "clsx";
import { useMemo } from "react";
import ConfluenceOverlay from "./ConfluenceOverlay";
import type { SupportingLevel } from "@/lib/chart";
import type { PlanSnapshot, TargetMetaEntry } from "@/lib/types";

type PlanPanelProps = {
  plan: PlanSnapshot["plan"];
  supportingLevels: SupportingLevel[];
  highlightedLevel: SupportingLevel | null;
  onSelectLevel: (level: SupportingLevel | null) => void;
  theme?: "dark" | "light";
};

export default function PlanPanel({
  plan,
  supportingLevels,
  highlightedLevel,
  onSelectLevel,
  theme = "dark",
}: PlanPanelProps) {
  const isLight = theme === "light";

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
    <div className="flex h-full flex-col gap-4">
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
    </div>
  );
}
