"use client";

import clsx from "clsx";
import { useMemo } from "react";

export type StatusToken = "connected" | "connecting" | "disconnected";

type StatusStripProps = {
  wsStatus: StatusToken;
  priceStatus: StatusToken;
  dataAgeSeconds?: number | null;
  riskBanner?: string | null;
  sessionBanner?: string | null;
  theme?: "dark" | "light";
  onToggleTheme?: () => void;
};

const STATUS_LABEL: Record<StatusToken, string> = {
  connected: "Live",
  connecting: "Connecting",
  disconnected: "Offline",
};

const STATUS_STYLE_DARK: Record<StatusToken, string> = {
  connected: "bg-emerald-500/15 text-emerald-200 border border-emerald-500/40",
  connecting: "bg-amber-500/15 text-amber-100 border border-amber-500/40",
  disconnected: "bg-rose-500/10 text-rose-200 border border-rose-500/40",
};

const STATUS_STYLE_LIGHT: Record<StatusToken, string> = {
  connected: "bg-emerald-500/15 text-emerald-700 border border-emerald-600/30",
  connecting: "bg-amber-400/15 text-amber-700 border border-amber-500/40",
  disconnected: "bg-rose-400/15 text-rose-700 border border-rose-500/40",
};

export default function StatusStrip({
  wsStatus,
  priceStatus,
  dataAgeSeconds,
  riskBanner,
  sessionBanner,
  theme = "dark",
  onToggleTheme,
}: StatusStripProps) {
  const formattedAge = useMemo(() => {
    if (dataAgeSeconds == null) return "n/a";
    if (dataAgeSeconds < 60) return `${Math.max(0, Math.round(dataAgeSeconds))}s`;
    const minutes = Math.floor(dataAgeSeconds / 60);
    if (minutes < 60) return `${minutes}m`;
    const hours = (dataAgeSeconds / 3600).toFixed(1);
    return `${hours}h`;
  }, [dataAgeSeconds]);

  const isLight = theme === "light";
  const statusStyles = isLight ? STATUS_STYLE_LIGHT : STATUS_STYLE_DARK;

  return (
    <div
      className={clsx(
        "flex flex-wrap items-center justify-between gap-4 px-4 py-3 sm:px-6",
        isLight ? "text-slate-600" : "text-neutral-400",
      )}
      role="status"
      aria-live="polite"
    >
      <div className="flex flex-wrap items-center gap-3 text-xs font-medium uppercase tracking-[0.25em]">
        <span
          className={clsx(
            "rounded-full px-3 py-1",
            isLight ? "bg-emerald-500/15 text-emerald-700" : "bg-emerald-500/10 text-emerald-200",
          )}
        >
          Trader Console
        </span>
        {sessionBanner ? (
          <span className={clsx(isLight ? "text-slate-700" : "text-neutral-300")}>{sessionBanner}</span>
        ) : null}
      </div>
      <div className="flex flex-wrap items-center gap-3 text-sm">
        <StatusPill label="Plan WS" status={wsStatus} theme={theme} styleMap={statusStyles} />
        <StatusPill label="Price Stream" status={priceStatus} theme={theme} styleMap={statusStyles} />
        <div
          className={clsx(
            "rounded-full px-3 py-1 text-xs uppercase tracking-[0.18em]",
            isLight
              ? "border border-slate-200 bg-white/80 text-slate-700 shadow-sm"
              : "border border-neutral-800/80 bg-neutral-900/80 text-neutral-300",
          )}
        >
          Data age <span className={clsx(isLight ? "text-slate-900" : "text-white")}>{formattedAge}</span>
        </div>
        {riskBanner ? (
          <div
            className={clsx(
              "rounded-full px-3 py-1 text-xs uppercase tracking-[0.18em]",
              isLight
                ? "border border-amber-500/40 bg-amber-400/20 text-amber-700"
                : "border border-neutral-700/60 bg-neutral-800/80 text-amber-200",
            )}
          >
            {riskBanner}
          </div>
        ) : null}
        {onToggleTheme ? (
          <button
            type="button"
            onClick={onToggleTheme}
            className={clsx(
              "rounded-full px-3 py-1 text-xs uppercase tracking-[0.18em] transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400",
              isLight
                ? "border border-slate-300 bg-white text-slate-700 hover:border-emerald-400 hover:text-emerald-600"
                : "border border-neutral-700 bg-neutral-900 text-neutral-200 hover:border-emerald-400 hover:text-emerald-200",
            )}
            aria-label={`Switch to ${isLight ? "dark" : "light"} theme`}
          >
            {isLight ? "Dark Mode" : "Light Mode"}
          </button>
        ) : null}
      </div>
    </div>
  );
}

function StatusPill({
  label,
  status,
  theme,
  styleMap,
}: {
  label: string;
  status: StatusToken;
  theme: "dark" | "light";
  styleMap: Record<StatusToken, string>;
}) {
  return (
    <span
      className={clsx(
        "rounded-full px-3 py-1 text-xs uppercase tracking-[0.18em] border",
        theme === "light" ? "shadow-sm" : "",
        styleMap[status],
      )}
    >
      {label} · {STATUS_LABEL[status]}
    </span>
  );
}
