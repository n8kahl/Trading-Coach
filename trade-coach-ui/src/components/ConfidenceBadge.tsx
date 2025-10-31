"use client";

import clsx from "clsx";

type ConfidenceBadgeProps = {
  value?: number | null;
  className?: string;
};

function normalizeConfidence(raw: number | null | undefined): number | null {
  if (typeof raw !== "number" || !Number.isFinite(raw)) return null;
  if (raw > 1 && raw <= 100) return Math.max(0, Math.min(raw / 100, 1));
  if (raw > 100) return 1;
  if (raw < 0) return 0;
  return raw;
}

export default function ConfidenceBadge({ value, className }: ConfidenceBadgeProps) {
  const normalized = normalizeConfidence(value);
  const percent = normalized != null ? Math.round(normalized * 100) : null;
  let tone: "green" | "amber" | "red" | "neutral" = "neutral";
  if (percent != null) {
    if (percent > 60) tone = "green";
    else if (percent >= 40) tone = "amber";
    else tone = "red";
  }

  const toneClass =
    tone === "green"
      ? "border-emerald-500/50 bg-emerald-500/20 text-emerald-100"
      : tone === "amber"
        ? "border-amber-500/50 bg-amber-500/20 text-amber-100"
        : tone === "red"
          ? "border-rose-500/50 bg-rose-500/20 text-rose-100"
          : "border-neutral-700 bg-neutral-900/70 text-neutral-300";

  const label = percent != null ? `${percent}%` : "—";

  return (
    <div
      className={clsx(
        "flex h-10 w-10 items-center justify-center rounded-full border text-sm font-semibold uppercase tracking-[0.2em]",
        toneClass,
        className,
      )}
      title={percent != null ? `Confidence ${percent}%` : "Confidence unavailable"}
      aria-label={percent != null ? `Confidence ${percent}%` : "Confidence unavailable"}
    >
      {label}
    </div>
  );
}
