"use client";

import type { SupportingLevel } from "@/lib/chart";
import clsx from "clsx";

type ConfluenceOverlayProps = {
  level: SupportingLevel | null;
  rationale?: string[];
  targetTag?: string | null;
};

export default function ConfluenceOverlay({ level, rationale, targetTag }: ConfluenceOverlayProps) {
  if (!level) {
    return (
      <div className="rounded-2xl border border-dashed border-neutral-800/70 bg-neutral-950/40 px-4 py-5 text-sm text-neutral-400">
        Hover a supporting level or align the crosshair to preview its confluence notes.
      </div>
    );
  }

  return (
    <div className="space-y-3 rounded-2xl border border-neutral-800/70 bg-neutral-900/70 px-4 py-5 shadow-inner shadow-black/40">
      <div className="flex items-baseline justify-between">
        <h3 className="text-xs font-semibold uppercase tracking-[0.3em] text-neutral-400">{level.label || "Supporting level"}</h3>
        <span className="rounded-full border border-emerald-500/40 bg-emerald-500/10 px-3 py-1 text-xs font-semibold text-emerald-200">
          {level.price.toFixed(2)}
        </span>
      </div>
      {targetTag ? (
        <p className="text-xs uppercase tracking-[0.2em] text-neutral-400">
          Aligns with <span className="text-emerald-200">{targetTag}</span>
        </p>
      ) : null}
      {rationale && rationale.length ? (
        <ul className="space-y-2 text-sm text-neutral-200">
          {rationale.map((item) => (
            <li key={item} className="rounded-xl border border-neutral-800/80 bg-neutral-900/80 px-3 py-2">
              {item}
            </li>
          ))}
        </ul>
      ) : (
        <p className="text-sm text-neutral-400">Level is tracked for structure awareness. Awaiting live rationale.</p>
      )}
    </div>
  );
}
