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
};

const STATUS_LABEL: Record<StatusToken, string> = {
  connected: "Live",
  connecting: "Connecting",
  disconnected: "Offline",
};

const STATUS_STYLE: Record<StatusToken, string> = {
  connected: "bg-emerald-500/15 text-emerald-200 border border-emerald-500/40",
  connecting: "bg-amber-500/15 text-amber-100 border border-amber-500/40",
  disconnected: "bg-rose-500/10 text-rose-200 border border-rose-500/40",
};

export default function StatusStrip({ wsStatus, priceStatus, dataAgeSeconds, riskBanner, sessionBanner }: StatusStripProps) {
  const formattedAge = useMemo(() => {
    if (dataAgeSeconds == null) return "n/a";
    if (dataAgeSeconds < 60) return `${Math.max(0, Math.round(dataAgeSeconds))}s`;
    const minutes = Math.floor(dataAgeSeconds / 60);
    if (minutes < 60) return `${minutes}m`;
    const hours = (dataAgeSeconds / 3600).toFixed(1);
    return `${hours}h`;
  }, [dataAgeSeconds]);

  return (
    <div className="flex flex-wrap items-center justify-between gap-4 px-4 py-3 sm:px-6">
      <div className="flex flex-wrap items-center gap-3 text-xs font-medium uppercase tracking-[0.25em] text-neutral-400">
        <span className="rounded-full bg-emerald-500/10 px-3 py-1 text-emerald-200">Trader Console</span>
        {sessionBanner ? <span className="text-neutral-300">{sessionBanner}</span> : null}
      </div>
      <div className="flex flex-wrap items-center gap-3 text-sm text-neutral-300">
        <StatusPill label="Plan WS" status={wsStatus} />
        <StatusPill label="Price Stream" status={priceStatus} />
        <div className="rounded-full border border-neutral-800/80 bg-neutral-900/80 px-3 py-1 text-xs uppercase tracking-[0.18em] text-neutral-300">
          Data age <span className="text-white">{formattedAge}</span>
        </div>
        {riskBanner ? (
          <div className="rounded-full border border-neutral-700/60 bg-neutral-800/80 px-3 py-1 text-xs uppercase tracking-[0.18em] text-amber-200">
            {riskBanner}
          </div>
        ) : null}
      </div>
    </div>
  );
}

function StatusPill({ label, status }: { label: string; status: StatusToken }) {
  return (
    <span className={clsx("rounded-full px-3 py-1 text-xs uppercase tracking-[0.18em]", STATUS_STYLE[status])}>
      {label} · {STATUS_LABEL[status]}
    </span>
  );
}
