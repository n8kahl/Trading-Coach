'use client';

import { useRouter } from "next/navigation";
import { useState } from "react";
import Link from "next/link";

const featuredPlans = [
  { label: "TSLA · Intraday", planId: "tsla-intraday-latest" },
  { label: "SPY · Swing", planId: "spy-swing-latest" },
  { label: "QQQ · 0DTE", planId: "qqq-0dte-latest" },
];

export default function Home() {
  const router = useRouter();
  const [planId, setPlanId] = useState("");
  const [symbol, setSymbol] = useState("");
  const [style, setStyle] = useState("intraday");

  const handlePlanSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!planId.trim()) return;
    router.push(`/plan/${encodeURIComponent(planId.trim())}`);
  };

  const handleSymbolSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!symbol.trim()) return;
    const slug = `${symbol.trim().toUpperCase()}-${style}`;
    router.push(`/plan/${encodeURIComponent(slug)}`);
  };

  return (
    <div className="relative flex flex-col items-center gap-12 px-6 py-16 sm:px-12">
      <header className="w-full max-w-5xl">
        <div className="flex flex-col gap-6">
          <span className="inline-flex w-fit items-center gap-2 rounded-full border border-emerald-400/40 bg-emerald-400/10 px-4 py-1 text-sm uppercase tracking-[0.3em] text-emerald-200">
            Fancy Trader 2.0
          </span>
          <h1 className="text-4xl font-semibold tracking-tight text-white sm:text-6xl">
            Real-time plan console for{" "}
            <span className="bg-gradient-to-r from-emerald-300 via-emerald-400 to-cyan-300 bg-clip-text text-transparent">
              active traders
            </span>
          </h1>
          <p className="max-w-2xl text-lg text-neutral-300 sm:text-xl">
            Plug directly into live Polygon data, ATR-based trail logic, and Fancy Trader coaching. Track execution, get alerts, and auto-replan when the market flips.
          </p>
        </div>
      </header>

      <section className="grid w-full max-w-5xl gap-8 md:grid-cols-2">
        <form
          onSubmit={handlePlanSubmit}
          className="rounded-3xl border border-neutral-800/70 bg-neutral-900/60 p-6 shadow-lg shadow-emerald-500/5 backdrop-blur"
        >
          <h2 className="text-lg font-semibold text-white">Jump to a plan ID</h2>
          <p className="mt-1 text-sm text-neutral-400">
            Paste any plan identifier (e.g. <code className="rounded bg-neutral-800 px-1 py-0.5">offline-SPY-swing-20251010</code>).
          </p>
          <div className="mt-4 flex gap-3">
            <input
              value={planId}
              onChange={(event) => setPlanId(event.target.value)}
              placeholder="plan-id-or-slug"
              className="flex-1 rounded-xl border border-neutral-700 bg-neutral-800 px-4 py-3 text-sm text-neutral-100 outline-none ring-emerald-400 transition focus:ring"
            />
            <button
              type="submit"
              className="rounded-xl bg-emerald-400 px-5 py-3 text-sm font-semibold text-emerald-950 transition hover:bg-emerald-300"
            >
              Open
            </button>
          </div>
        </form>

        <form
          onSubmit={handleSymbolSubmit}
          className="rounded-3xl border border-neutral-800/70 bg-neutral-900/60 p-6 shadow-lg shadow-cyan-500/5 backdrop-blur"
        >
          <h2 className="text-lg font-semibold text-white">Find the latest plan</h2>
          <p className="mt-1 text-sm text-neutral-400">
            Enter a symbol and style. The console will auto-resolve the freshest plan slug.
          </p>
          <div className="mt-4 grid gap-3 sm:grid-cols-[1fr_auto]">
            <input
              value={symbol}
              onChange={(event) => setSymbol(event.target.value)}
              placeholder="Symbol (e.g. TSLA)"
              className="rounded-xl border border-neutral-700 bg-neutral-800 px-4 py-3 text-sm text-neutral-100 outline-none ring-cyan-400 transition focus:ring"
            />
            <select
              value={style}
              onChange={(event) => setStyle(event.target.value)}
              className="rounded-xl border border-neutral-700 bg-neutral-800 px-4 py-3 text-sm text-neutral-100 outline-none ring-cyan-400 transition focus:ring"
            >
              <option value="0dte">0DTE</option>
              <option value="scalp">Scalp</option>
              <option value="intraday">Intraday</option>
              <option value="swing">Swing</option>
              <option value="leaps">Leaps</option>
            </select>
            <button
              type="submit"
              className="rounded-xl bg-cyan-400 px-5 py-3 text-sm font-semibold text-cyan-950 transition hover:bg-cyan-300 sm:col-span-2"
            >
              Resolve latest plan
            </button>
          </div>
        </form>
      </section>

      <section className="w-full max-w-5xl rounded-3xl border border-neutral-800/70 bg-neutral-900/40 p-6 backdrop-blur">
        <h3 className="text-sm font-semibold uppercase tracking-[0.3em] text-neutral-400">Quick links</h3>
        <div className="mt-4 grid gap-4 sm:grid-cols-3">
          {featuredPlans.map((item) => (
            <Link
              key={item.planId}
              href={`/plan/${encodeURIComponent(item.planId)}`}
              className="group rounded-2xl border border-neutral-800/80 bg-neutral-900/70 px-4 py-5 transition hover:border-emerald-400/50 hover:bg-neutral-900/90"
            >
              <div className="text-sm font-medium text-neutral-300 group-hover:text-emerald-200">
                {item.label}
              </div>
              <div className="mt-1 truncate text-xs text-neutral-500 group-hover:text-neutral-400">
                {item.planId}
              </div>
            </Link>
          ))}
        </div>
      </section>
    </div>
  );
}
