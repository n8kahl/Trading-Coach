"use client";
import Link from 'next/link';
import clsx from 'clsx';
import { PUBLIC_UI_BASE_URL } from '@/lib/env';

type PlanHeaderProps = {
  planId: string;
  uiUrl?: string | null;
  theme?: 'dark' | 'light';
};

export default function PlanHeader({ planId, uiUrl, theme = 'dark' }: PlanHeaderProps) {
  const fallbackHref = `${PUBLIC_UI_BASE_URL}/plan/${encodeURIComponent(planId)}`;
  const uiHref = uiUrl && uiUrl.trim() ? uiUrl : fallbackHref;
  const buttonClass = clsx(
    'inline-flex items-center rounded-full px-3 py-1 text-xs font-semibold uppercase tracking-[0.2em] transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-400',
    theme === 'light'
      ? 'border border-slate-200 bg-white text-slate-700 hover:border-emerald-400 hover:text-emerald-600'
      : 'border border-neutral-800 bg-neutral-900 text-neutral-100 hover:border-emerald-400/60 hover:text-emerald-200',
  );

  return (
    <div className="mb-4 flex flex-wrap items-center gap-2 text-sm">
      <Link href={uiHref} className={buttonClass}>
        Open Plan
      </Link>
    </div>
  );
}
