"use client";

import clsx from "clsx";
import type { ReactNode } from "react";

type PlanShellProps = {
  header: ReactNode;
  leftRail?: ReactNode;
  coach?: ReactNode;
  chart: ReactNode;
  sidePanel?: ReactNode;
  footer?: ReactNode;
  className?: string;
};

export default function PlanShell({
  header,
  leftRail = null,
  coach = null,
  chart,
  sidePanel = null,
  footer = null,
  className,
}: PlanShellProps) {
  return (
    <div className={clsx("flex min-h-screen flex-col bg-neutral-950 text-neutral-100", className)}>
      <header className="sticky top-0 z-40 border-b border-neutral-900/80 bg-neutral-950/95 backdrop-blur">
        <div className="mx-auto flex w-full max-w-[1400px] flex-col gap-3 px-4 py-4 sm:px-6">{header}</div>
      </header>
      <main className="flex-1">
        <div className="mx-auto grid max-w-[1400px] grid-cols-12 gap-4 px-4 py-4 sm:px-6">
          <aside className="hidden md:col-span-3 md:block">
            {leftRail ? (
              <div className="sticky top-[88px] space-y-3">
                <div className="rounded-xl border border-neutral-900/70 bg-neutral-950/60">{leftRail}</div>
              </div>
            ) : null}
          </aside>
          <section className="col-span-12 flex flex-col gap-4 md:col-span-9">
            {coach ? (
              <div className="sticky top-[72px] z-10">
                <div className="rounded-2xl border border-neutral-900/70 bg-neutral-950/70 p-0.5 backdrop-blur">
                  {coach}
                </div>
              </div>
            ) : null}
            <div className="rounded-2xl border border-neutral-900/70 bg-neutral-950/60 p-1" data-testid="chart-container">
              <div className="rounded-[1.1rem] border border-neutral-900/60 bg-neutral-950/80 p-1" data-testid="chart-canvas">
                {chart}
              </div>
            </div>
            {sidePanel}
          </section>
        </div>
      </main>
      {footer ? (
        <footer className="border-t border-neutral-900/80">
          <div className="mx-auto flex w-full max-w-[1400px] items-center justify-between px-4 py-3 text-sm text-neutral-400 sm:px-6">
            {footer}
          </div>
        </footer>
      ) : null}
    </div>
  );
}
