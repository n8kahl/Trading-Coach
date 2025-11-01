"use client";

import clsx from "clsx";
import { useMemo } from "react";
import { useStore } from "@/store/useStore";

type PhaseTone = {
  label: string;
  tone: "open" | "pre" | "closed";
};

function derivePhase(statusRaw: string | null | undefined): PhaseTone {
  const token = (statusRaw || "").toLowerCase();
  if (!token) {
    return { label: "—", tone: "closed" };
  }
  if (token.includes("open")) {
    return { label: "RTH", tone: "open" };
  }
  if (token.includes("pre")) {
    return { label: "Pre-Market", tone: "pre" };
  }
  if (token.includes("after") || token.includes("post") || token.includes("extended")) {
    return { label: "After Hours", tone: "pre" };
  }
  return { label: token.replace(/[_-]/g, " ").toUpperCase(), tone: "closed" };
}

function formatWithTimezone(iso: string | null | undefined, tz: string | null | undefined): string | null {
  if (!iso) return null;
  const date = new Date(iso);
  if (!Number.isFinite(date.getTime())) {
    return null;
  }
  const formatter = new Intl.DateTimeFormat("en-US", {
    hour: "2-digit",
    minute: "2-digit",
    hour12: true,
    timeZone: tz || "UTC",
    timeZoneName: "short",
  });
  return formatter.format(date);
}

function formatDateShort(iso: string | null | undefined, tz: string | null | undefined): string | null {
  if (!iso) return null;
  const date = new Date(iso);
  if (!Number.isFinite(date.getTime())) {
    return null;
  }
  const formatter = new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hour12: true,
    timeZone: tz || "UTC",
  });
  return formatter.format(date);
}

export default function SessionChip() {
  const session = useStore((state) => state.session);
  const { label, tone } = useMemo(() => derivePhase(session?.status ?? null), [session?.status]);
  const asOfLabel = useMemo(() => formatWithTimezone(session?.as_of ?? null, session?.tz ?? null), [session?.as_of, session?.tz]);
  const nextOpenLabel = useMemo(
    () => formatDateShort(session?.next_open ?? null, session?.tz ?? null),
    [session?.next_open, session?.tz],
  );

  if (!session) {
    return null;
  }

  const toneClass =
    tone === "open"
      ? "border-emerald-400/40 bg-emerald-400/10 text-emerald-100"
      : tone === "pre"
        ? "border-amber-300/40 bg-amber-300/10 text-amber-100"
        : "border-slate-600/50 bg-slate-700/10 text-slate-200";

  return (
    <div
      className={clsx(
        "inline-flex items-center gap-3 rounded-full border px-4 py-2 text-sm transition-colors",
        toneClass,
      )}
      title={session.status ? `Session status ${session.status}` : "Session status unavailable"}
      aria-label={session.status ? `Session status ${session.status}` : "Session status unavailable"}
    >
      <span className="text-xs font-semibold uppercase tracking-wide">{label}</span>
      <div className="flex flex-col leading-tight text-[11px] text-slate-200">
        <span>{asOfLabel ? `As of ${asOfLabel}` : "As of —"}</span>
        {nextOpenLabel ? <span className="text-slate-300/70">Next open {nextOpenLabel}</span> : null}
      </div>
    </div>
  );
}
