"use client";
import Link from 'next/link';

export default function PlanHeader({ planId, legacyUrl, uiUrl }: { planId: string; legacyUrl?: string | null; uiUrl?: string | null }) {
  const uiHref = uiUrl || `/plan/${encodeURIComponent(planId)}`;
  return (
    <div className="mb-3 flex flex-wrap items-center gap-2 text-sm">
      <Link href={uiHref} className="rounded bg-[var(--chip)] px-2 py-1 hover:bg-[var(--border)]">Open in UI</Link>
      {legacyUrl ? (
        <a href={legacyUrl} target="_blank" rel="noreferrer" className="rounded bg-[var(--chip)] px-2 py-1 hover:bg-[var(--border)]">
          Open in Legacy Chart
        </a>
      ) : null}
    </div>
  );
}
