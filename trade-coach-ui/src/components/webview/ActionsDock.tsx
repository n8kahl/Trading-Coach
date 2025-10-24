"use client";

import clsx from "clsx";

export type ActionKey = "size" | "alerts" | "validate" | "coach" | "share";

const ACTION_LABELS: Record<ActionKey, string> = {
  size: "Size It",
  alerts: "Alerts",
  validate: "Validate Plan",
  coach: "Ask Coach",
  share: "Share Chart",
};

type ActionsDockProps = {
  onAction?: (action: ActionKey) => void;
  disabledActions?: Partial<Record<ActionKey, boolean>>;
  layout?: "vertical" | "horizontal";
  theme?: "dark" | "light";
};

export default function ActionsDock({ onAction, disabledActions, layout = "vertical", theme = "dark" }: ActionsDockProps) {
  const entries: ActionKey[] = ["size", "alerts", "validate", "coach", "share"];
  const isLight = theme === "light";
  return (
    <div
      className={clsx(
        "pointer-events-auto",
        layout === "vertical" ? "flex flex-col gap-3" : "flex flex-wrap justify-center gap-2",
      )}
    >
      {entries.map((action) => {
        const disabled = !!disabledActions?.[action];
        return (
          <button
            key={action}
            type="button"
            onClick={() => !disabled && onAction?.(action)}
            disabled={disabled}
            className={clsx(
              "rounded-full px-5 py-3 text-sm font-semibold uppercase tracking-[0.2em] transition focus:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400/80",
              isLight
                ? "border border-slate-300 bg-white text-slate-700 shadow-sm hover:border-emerald-500/60 hover:bg-emerald-400/15 hover:text-emerald-600"
                : "border border-neutral-800/80 bg-neutral-950/80 text-neutral-100 shadow-lg shadow-emerald-500/20 hover:border-emerald-500/60 hover:bg-emerald-500/15 hover:text-emerald-200",
              layout === "horizontal" && "px-4 py-2 text-xs",
              disabled &&
                (isLight
                  ? "cursor-not-allowed border-slate-200 bg-slate-100 text-slate-400 hover:border-slate-200 hover:bg-slate-100 hover:text-slate-400"
                  : "cursor-not-allowed border-neutral-800/40 bg-neutral-900/60 text-neutral-600 hover:border-neutral-800/40 hover:bg-neutral-900/60 hover:text-neutral-600"),
            )}
          >
            {ACTION_LABELS[action]}
          </button>
        );
      })}
    </div>
  );
}
