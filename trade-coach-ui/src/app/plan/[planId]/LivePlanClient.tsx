'use client';

import { useEffect, useMemo, useReducer, useState } from "react";
import clsx from "clsx";
import PriceChart from "@/components/PriceChart";
import { API_BASE_URL, WS_BASE_URL } from "@/lib/env";
import type { PlanDeltaEvent, PlanSnapshot, SymbolTickEvent } from "@/lib/types";
import Link from "next/link";

type LivePlanClientProps = {
  initialSnapshot: PlanSnapshot;
  planId: string;
  symbol: string | undefined;
};

type CoachingEvent = {
  id: string;
  timestamp: string;
  message: string;
  tone: "positive" | "warning" | "neutral";
};

type PlanState = {
  status: string;
  nextStep?: string;
  rr?: number | null;
  trailingStop?: number | null;
  lastPrice?: number | null;
  note?: string;
};

type PricePoint = { time: number; value: number };

const MAX_POINTS = 720;

function deriveInitialState(snapshot: PlanSnapshot): PlanState {
  const structured = snapshot.plan.structured_plan;
  const lastPrice = structured?.entry?.level ?? snapshot.plan.entry ?? null;
  return {
    status: structured?.invalid ? "invalid" : "planned",
    nextStep: structured?.options ? "Review options block" : undefined,
    rr: snapshot.plan.rr_to_t1 ?? null,
    lastPrice,
    note: snapshot.plan.notes ?? snapshot.plan.warnings?.[0],
  };
}

function planReducer(prev: PlanState, event: PlanDeltaEvent["changes"]): PlanState {
  return {
    status: event.status ?? prev.status,
    nextStep: event.next_step ?? prev.nextStep,
    rr: event.rr_to_t1 ?? prev.rr,
    trailingStop: event.trailing_stop ?? prev.trailingStop,
    lastPrice: event.last_price ?? prev.lastPrice,
    note: event.note ?? prev.note,
  };
}

export default function LivePlanClient({ initialSnapshot, planId, symbol }: LivePlanClientProps) {
  const upperSymbol = (symbol ?? initialSnapshot.plan.symbol ?? "").toUpperCase();
  const [planState, dispatchPlan] = useReducer(planReducer, deriveInitialState(initialSnapshot));
  const [coachingLog, setCoachingLog] = useState<CoachingEvent[]>(() => {
    const banner = initialSnapshot.plan.session_state?.banner;
    const note = initialSnapshot.plan.notes;
    const items: CoachingEvent[] = [];
    if (banner) {
      items.push({
        id: `banner-${Date.now()}`,
        timestamp: initialSnapshot.plan.session_state?.as_of ?? new Date().toISOString(),
        message: banner,
        tone: "neutral",
      });
    }
    if (note) {
      items.push({
        id: `note-${Date.now() + 1}`,
        timestamp: new Date().toISOString(),
        message: note,
        tone: note.toLowerCase().includes("stop") ? "warning" : "neutral",
      });
    }
    return items;
  });
  const [priceSeries, setPriceSeries] = useState<PricePoint[]>([]);

  const plan = initialSnapshot.plan;
  const structured = plan.structured_plan;
  const summaryLevels = useMemo(() => {
    const summary = initialSnapshot.summary as { key_levels?: Record<string, number> } | undefined;
    if (summary?.key_levels && typeof summary.key_levels === "object") {
      return summary.key_levels as Record<string, number>;
    }
    const planLevels = (initialSnapshot.plan as { key_levels?: Record<string, number> }).key_levels;
    if (planLevels && typeof planLevels === "object") {
      return planLevels as Record<string, number>;
    }
    return {} as Record<string, number>;
  }, [initialSnapshot]);

  useEffect(() => {
    const wsUrl = `${WS_BASE_URL}/ws/plans/${encodeURIComponent(planId)}`;
    const socket = new WebSocket(wsUrl);

    socket.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data) as PlanDeltaEvent;
        if (payload.t !== "plan_delta") return;
        dispatchPlan(payload.changes);
        if (payload.changes.note) {
          setCoachingLog((prev) => [
            {
              id: `plan-${payload.changes.timestamp ?? Date.now()}`,
              timestamp: payload.changes.timestamp ?? new Date().toISOString(),
              message: payload.changes.note,
              tone: payload.changes.status === "invalidated" || payload.changes.status === "plan_invalidated" ? "warning" : "neutral",
            },
            ...prev.slice(0, 49),
          ]);
        }
      } catch (error) {
        if (process.env.NODE_ENV !== "production") {
          console.error("plan ws parse error", error);
        }
      }
    };

    socket.onerror = () => {
      setCoachingLog((prev) => [
        {
          id: `error-${Date.now()}`,
          timestamp: new Date().toISOString(),
          message: "Plan stream disconnected. Attempting to reconnect…",
          tone: "warning",
        },
        ...prev,
      ]);
    };

    return () => {
      socket.close();
    };
  }, [planId]);

  useEffect(() => {
    if (!upperSymbol) return;
    const streamUrl = `${API_BASE_URL}/stream/${encodeURIComponent(upperSymbol)}`;
    const source = new EventSource(streamUrl);

    source.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data) as SymbolTickEvent;
        if (payload.t === "tick") {
          const timestamp = Math.floor(new Date(payload.ts).getTime() / 1000);
          setPriceSeries((prev) => {
            const next = [...prev, { time: timestamp, value: payload.p }];
            if (next.length > MAX_POINTS) {
              next.splice(0, next.length - MAX_POINTS);
            }
            return next;
          });
          dispatchPlan({ last_price: payload.p });
        } else if (payload.t === "market_status") {
          setCoachingLog((prev) => [
            {
              id: `market-${Date.now()}`,
              timestamp: new Date().toISOString(),
              message: payload.note,
              tone: payload.note.toLowerCase().includes("limited") ? "warning" : "neutral",
            },
            ...prev.slice(0, 49),
          ]);
        }
      } catch (error) {
        if (process.env.NODE_ENV !== "production") {
          console.error("symbol stream error", error);
        }
      }
    };

    source.onerror = () => {
      source.close();
      setCoachingLog((prev) => [
        {
          id: `symbol-error-${Date.now()}`,
          timestamp: new Date().toISOString(),
          message: "Price stream unavailable. Check Polygon/Finnhub credentials.",
          tone: "warning",
        },
        ...prev,
      ]);
    };

    return () => {
      source.close();
    };
  }, [upperSymbol]);

  const targets = useMemo(() => {
    if (structured?.targets?.length) return structured.targets;
    if (plan.targets?.length) return plan.targets;
    return [];
  }, [structured?.targets, plan.targets]);

  const summaryCards = [
    {
      label: "Status",
      value: planState.status?.replace("_", " ").toUpperCase() ?? "N/A",
      accent: "from-emerald-400/40 to-emerald-500/20",
    },
    {
      label: "Next step",
      value: planState.nextStep ?? "Monitor execution",
      accent: "from-sky-400/40 to-sky-500/20",
    },
    {
      label: "R:R to TP1",
      value: planState.rr ? planState.rr.toFixed(2) : "—",
      accent: "from-amber-400/40 to-amber-500/20",
    },
  ];

  return (
    <div className="space-y-8 px-6 py-10 sm:px-10">
      <header className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <div className="text-sm uppercase tracking-[0.3em] text-neutral-400">Plan console</div>
          <h1 className="mt-2 text-3xl font-semibold text-white sm:text-4xl">
            {upperSymbol} · {plan.style ?? structured?.style ?? "Unknown"} plan
          </h1>
          <p className="mt-2 max-w-xl text-sm text-neutral-400">
            Session banner:{" "}
            <span className="text-neutral-200">{plan.session_state?.banner ?? "Live data mode"}</span>
            {plan.session_state?.as_of ? (
              <span className="text-neutral-500">
                {" "}
                — as of {new Date(plan.session_state.as_of).toLocaleString()}
              </span>
            ) : null}
          </p>
        </div>
        {plan.chart_url ? (
          <Link
            href={plan.chart_url}
            target="_blank"
            className="inline-flex items-center gap-2 rounded-full border border-emerald-400/60 bg-emerald-400/10 px-4 py-2 text-sm font-semibold text-emerald-200 transition hover:bg-emerald-400/20"
          >
            View legacy chart
          </Link>
        ) : null}
      </header>

      <div className="grid gap-6 lg:grid-cols-[2fr_1fr]">
        <section className="rounded-3xl border border-neutral-800/80 bg-neutral-900/50 p-6 backdrop-blur">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-sm font-medium text-neutral-400">Live price</div>
              <div className="mt-1 text-3xl font-semibold text-emerald-300">
                {planState.lastPrice ? planState.lastPrice.toFixed(2) : "—"}
              </div>
            </div>
            <div className="text-right text-sm text-neutral-400">
              <div>Entry {structured?.entry?.level?.toFixed(2) ?? plan.entry?.toFixed(2) ?? "—"}</div>
              <div>Stop {planState.trailingStop?.toFixed(2) ?? plan.stop?.toFixed(2) ?? "—"}</div>
            </div>
          </div>
          <div className="mt-6 overflow-hidden rounded-2xl border border-neutral-800/70 bg-neutral-950/40">
            <PriceChart data={priceSeries} lastPrice={planState.lastPrice ?? undefined} />
          </div>
        </section>

        <aside className="space-y-4">
          {summaryCards.map((card) => (
            <div
              key={card.label}
              className={clsx(
                "rounded-2xl border border-neutral-800/60 bg-gradient-to-br p-5 backdrop-blur",
                card.accent,
              )}
            >
              <div className="text-xs uppercase tracking-[0.3em] text-neutral-200/70">{card.label}</div>
              <div className="mt-2 text-lg font-semibold text-white">{card.value}</div>
            </div>
          ))}

          <div className="rounded-2xl border border-neutral-800/60 bg-neutral-900/40 p-5 backdrop-blur">
            <div className="text-xs uppercase tracking-[0.3em] text-neutral-400">Targets</div>
            <ul className="mt-3 space-y-2 text-sm text-neutral-200">
              {targets.length === 0 ? (
                <li className="text-neutral-500">No targets published.</li>
              ) : (
                targets.map((target, index) => (
                  <li key={target} className="flex items-center justify-between rounded-xl border border-neutral-800/70 bg-neutral-900/60 px-3 py-2">
                    <span className="text-neutral-400">TP{index + 1}</span>
                    <span className="font-semibold text-emerald-300">{target.toFixed(2)}</span>
                  </li>
                ))
              )}
            </ul>
          </div>
        </aside>
      </div>

      <div className="grid gap-6 lg:grid-cols-[2fr_1fr]">
        <section className="rounded-3xl border border-neutral-800/80 bg-neutral-900/40 p-6 backdrop-blur">
          <header className="flex items-center justify-between">
            <h2 className="text-sm font-semibold uppercase tracking-[0.3em] text-neutral-400">
              Coaching timeline
            </h2>
            <span className="text-xs text-neutral-500">Latest {Math.min(coachingLog.length, 6)} updates</span>
          </header>
          <ol className="mt-4 space-y-4">
            {coachingLog.length === 0 ? (
              <li className="rounded-2xl border border-neutral-800/70 bg-neutral-900/60 p-4 text-sm text-neutral-400">
                Waiting for the first live update…
              </li>
            ) : (
              coachingLog.slice(0, 6).map((event) => (
                <li
                  key={event.id}
                  className={clsx(
                    "rounded-2xl border px-4 py-4 text-sm",
                    event.tone === "positive" && "border-emerald-400/50 bg-emerald-400/10 text-emerald-100",
                    event.tone === "warning" && "border-amber-400/60 bg-amber-400/10 text-amber-100",
                    event.tone === "neutral" && "border-neutral-800/70 bg-neutral-900/60 text-neutral-200",
                  )}
                >
                  <div className="text-xs uppercase tracking-[0.2em] text-neutral-400/80">
                    {new Date(event.timestamp).toLocaleTimeString()}
                  </div>
                  <p className="mt-2 leading-relaxed">{event.message}</p>
                </li>
              ))
            )}
          </ol>
        </section>

        <section className="space-y-4 rounded-3xl border border-neutral-800/80 bg-neutral-900/40 p-6 backdrop-blur">
          <div>
            <div className="text-xs uppercase tracking-[0.3em] text-neutral-400">Plan metadata</div>
            <dl className="mt-3 space-y-2 text-sm text-neutral-300">
              <div className="flex items-center justify-between">
                <dt className="text-neutral-400">Plan ID</dt>
                <dd className="font-mono text-xs text-neutral-300">{plan.plan_id}</dd>
              </div>
              <div className="flex items-center justify-between">
                <dt className="text-neutral-400">Version</dt>
                <dd>{plan.structured_plan?.plan_id === plan.plan_id ? "Structured" : "Legacy"}</dd>
              </div>
              <div className="flex items-center justify-between">
                <dt className="text-neutral-400">Confidence</dt>
                <dd>{plan.confidence ? `${(plan.confidence * 100).toFixed(0)}%` : "—"}</dd>
              </div>
              <div className="flex items-center justify-between">
                <dt className="text-neutral-400">RR to TP1</dt>
                <dd>{planState.rr ? planState.rr.toFixed(2) : "—"}</dd>
              </div>
            </dl>
          </div>

          <div>
            <div className="text-xs uppercase tracking-[0.3em] text-neutral-400">Key levels</div>
            <ul className="mt-2 space-y-1 text-sm text-neutral-300">
              {Object.entries(summaryLevels)
                .slice(0, 6)
                .map(([level, value]) => (
                  <li key={level} className="flex items-center justify-between text-neutral-400">
                    <span>{level.replace(/_/g, " ")}</span>
                    <span className="text-neutral-200">{typeof value === "number" ? value.toFixed(2) : value}</span>
                  </li>
                ))}
              {Object.keys(summaryLevels).length === 0 && (
                <li className="text-neutral-500">No keyed levels in snapshot.</li>
              )}
            </ul>
          </div>
        </section>
      </div>
    </div>
  );
}
