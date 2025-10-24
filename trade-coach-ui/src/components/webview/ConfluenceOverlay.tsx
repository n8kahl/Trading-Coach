"use client";

import type { SupportingLevel } from "@/lib/chart";
import clsx from "clsx";

type ConfluenceOverlayProps = {
  level: SupportingLevel | null;
  rationale?: string[];
  targetTag?: string | null;
  theme?: "dark" | "light";
};

export default function ConfluenceOverlay({ level, rationale, targetTag, theme = "dark" }: ConfluenceOverlayProps) {
  const isLight = theme === "light";
  if (!level) {
    return (
      <div
        className={clsx(
          "rounded-2xl border border-dashed px-4 py-5 text-sm",
          isLight ? "border-slate-300 bg-white text-slate-500" : "border-neutral-800/70 bg-neutral-950/40 text-neutral-400",
        )}
      >
        Hover a supporting level or align the crosshair to preview its confluence notes.
      </div>
    );
  }

  return (
    <div
      className={clsx(
        "space-y-3 rounded-2xl border px-4 py-5 shadow-inner",
        isLight ? "border-slate-200 bg-white text-slate-700 shadow-slate-200/60" : "border-neutral-800/70 bg-neutral-900/70 text-neutral-200 shadow-black/40",
      )}
    >
      <div className="flex items-baseline justify-between">
        <h3 className={clsx("text-xs font-semibold uppercase tracking-[0.3em]", isLight ? "text-slate-500" : "text-neutral-400")}>
          {level.label || "Supporting level"}
        </h3>
        <span
          className={clsx(
            "rounded-full border px-3 py-1 text-xs font-semibold",
            isLight ? "border-emerald-500/40 bg-emerald-400/20 text-emerald-700" : "border-emerald-500/40 bg-emerald-500/10 text-emerald-200",
          )}
        >
          {level.price.toFixed(2)}
        </span>
      </div>
      {targetTag ? (
        <p className={clsx("text-xs uppercase tracking-[0.2em]", isLight ? "text-slate-500" : "text-neutral-400")}>
          Aligns with{" "}
          <span className={clsx(isLight ? "text-emerald-600" : "text-emerald-200")}>{targetTag}</span>
        </p>
      ) : null}
      {rationale && rationale.length ? (
        <ul className="space-y-2 text-sm">
          {rationale.map((item) => (
            <li
              key={item}
              className={clsx(
                "rounded-xl border px-3 py-2",
                isLight ? "border-slate-200 bg-white text-slate-700" : "border-neutral-800/80 bg-neutral-900/80 text-neutral-200",
              )}
            >
              {item}
            </li>
          ))}
        </ul>
      ) : (
        <p className={clsx("text-sm", isLight ? "text-slate-500" : "text-neutral-400")}>
          Level is tracked for structure awareness. Awaiting live rationale.
        </p>
      )}
    </div>
  );
}
