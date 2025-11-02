"use client";

import clsx from "clsx";

export type SessionSSOT = {
  status: string;
  tz: string;
  as_of: string;
  next_open?: string | null;
};

export function humanStatus(status: string | null | undefined): string {
  if (!status) return "Unknown";
  const token = status.trim().toLowerCase();
  if (token.includes("pre")) return "Premarket";
  if (token.includes("after") || token.includes("post") || token.includes("extended")) return "After Hours";
  if (token.includes("open")) return "Open";
  if (token.includes("close")) return "Closed";
  return status.trim();
}

export function safeTime(iso: string | null | undefined): string {
  if (!iso) return "—";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) {
    return "—";
  }
  return date.toLocaleString(undefined, {
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });
}

type SessionChipProps = {
  session?: SessionSSOT | null;
  className?: string;
};

export default function SessionChip({ session, className }: SessionChipProps) {
  if (!session) {
    return null;
  }

  const statusLabel = humanStatus(session.status);
  const statusToken = statusLabel.toLowerCase();
  const toneClass = statusToken === "open"
    ? "border-emerald-500/60 bg-emerald-500/15 text-emerald-100"
    : statusToken.includes("market") || statusToken.includes("hours")
      ? "border-amber-400/50 bg-amber-400/15 text-amber-100"
      : "border-neutral-800/70 bg-neutral-900/70 text-neutral-200";

  const tzLabel = session.tz?.toUpperCase() || "UTC";
  const asOfLabel = safeTime(session.as_of);

  return (
    <span
      className={clsx(
        "inline-flex items-center gap-2 rounded-full border px-3 py-1 text-[10px] font-semibold uppercase tracking-[0.2em] transition-colors",
        toneClass,
        className,
      )}
    >
      <span className="text-[9px] tracking-[0.26em] text-white/90">{statusLabel}</span>
      <span className="rounded-full border border-neutral-800/60 bg-neutral-950/70 px-2 py-0.5 text-[9px] tracking-[0.24em] text-neutral-300">
        {tzLabel}
      </span>
      <span className="text-[9px] tracking-[0.22em] text-neutral-300/90">As of {asOfLabel}</span>
    </span>
  );
}
