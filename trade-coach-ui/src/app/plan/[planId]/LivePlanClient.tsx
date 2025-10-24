'use client';

import clsx from "clsx";
import { useCallback, useEffect, useMemo, useReducer, useState } from "react";
import type { LineData } from "lightweight-charts";
import WebviewShell from "@/components/webview/WebviewShell";
import StatusStrip, { type StatusToken } from "@/components/webview/StatusStrip";
import ActionsDock, { type ActionKey } from "@/components/webview/ActionsDock";
import PlanPanel from "@/components/webview/PlanPanel";
import ChartContainer from "@/components/webview/ChartContainer";
import type { PlanDeltaEvent, PlanSnapshot, StructuredPlan, SymbolTickEvent } from "@/lib/types";
import { API_BASE_URL, WS_BASE_URL } from "@/lib/env";
import {
  normalizeChartParams,
  parsePrice,
  parseSupportingLevels,
  parseTargets,
  parseUiState,
  type ParsedUiState,
  type SupportingLevel,
} from "@/lib/chart";

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
  const structured = (initialSnapshot.plan.structured_plan ?? null) as StructuredPlan | null;

  const [planState, dispatchPlan] = useReducer(planReducer, deriveInitialState(initialSnapshot));
  const [coachingLog, setCoachingLog] = useState<CoachingEvent[]>(() => bootstrapCoachingLog(initialSnapshot));
  const [priceSeries, setPriceSeries] = useState<LineData[]>([]);
  const [nextPlanId, setNextPlanId] = useState<string | null>(null);
  const [wsStatus, setWsStatus] = useState<StatusToken>("connecting");
  const [priceStatus, setPriceStatus] = useState<StatusToken>("connecting");
  const [dataAgeSeconds, setDataAgeSeconds] = useState<number | null>(() => computeDataAge(initialSnapshot.plan.session_state?.as_of));
  const [showSupportingLevels, setShowSupportingLevels] = useState<boolean>(() => {
    const params = normalizeChartParams(initialSnapshot.plan.charts?.params ?? initialSnapshot.plan.charts_params ?? null);
    return params.supportingLevels !== "0";
  });
  const [uiTheme, setUiTheme] = useState<"dark" | "light">(() => (chartParamsRaw.theme === "light" ? "light" : "dark"));
  const [highlightedLevel, setHighlightedLevel] = useState<SupportingLevel | null>(null);
  const [sheetOpen, setSheetOpen] = useState(false);
  const initStartRef = useRef<number | null>(now());
  const initLoggedRef = useRef(false);
  const wsStartRef = useRef<number | null>(null);
  const analysisStartRef = useRef<number | null>(now());
  const analysisMeasuredRef = useRef(false);

  const plan = initialSnapshot.plan;
  const chartParamsRaw = useMemo(
    () => normalizeChartParams(plan.charts?.params ?? plan.charts_params ?? null),
    [plan.charts?.params, plan.charts_params],
  );
  const chartUiState: ParsedUiState = useMemo(() => parseUiState(chartParamsRaw.ui_state ?? null), [chartParamsRaw]);
  const supportingLevels = useMemo(() => parseSupportingLevels(chartParamsRaw.levels), [chartParamsRaw.levels]);

  const entryFromChart = parsePrice(chartParamsRaw.entry);
  const stopFromChart = parsePrice(chartParamsRaw.stop);
  const targetsFromChart = parseTargets(chartParamsRaw.tp);

  const entryLevel = entryFromChart ?? structured?.entry?.level ?? plan.entry ?? null;
  const baseStop = stopFromChart ?? plan.stop ?? structured?.stop ?? null;
  const targets = targetsFromChart.length ? targetsFromChart : structured?.targets ?? plan.targets ?? [];

  useEffect(() => {
    setDataAgeSeconds(computeDataAge(plan.session_state?.as_of));
    const timer = window.setInterval(() => setDataAgeSeconds(computeDataAge(plan.session_state?.as_of)), 30_000);
    return () => window.clearInterval(timer);
  }, [plan.session_state?.as_of]);

  useEffect(() => {
    if (!initLoggedRef.current && initStartRef.current != null) {
      const end = now();
      if (end != null) {
        logBudget("webview:init", end - initStartRef.current, 400);
        initLoggedRef.current = true;
      }
    }
    const latency = extractPlanLatency(plan);
    if (latency != null) logBudget("plan:initial", latency, 800);
  }, [plan]);

  useEffect(() => {
    const wsUrl = `${WS_BASE_URL}/ws/plans/${encodeURIComponent(planId)}`;
    wsStartRef.current = now();
    const socket = new WebSocket(wsUrl);

    socket.onopen = () => {
      setWsStatus("connected");
      if (wsStartRef.current != null) {
        const end = now();
        if (end != null) logBudget("ws:connect", end - wsStartRef.current, 500);
        wsStartRef.current = null;
      }
    };
    socket.onclose = () => setWsStatus("disconnected");
    socket.onerror = () => setWsStatus("disconnected");

    socket.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data) as PlanDeltaEvent;
        if (payload.t !== "plan_delta") return;
        dispatchPlan(payload.changes);
        const nextPlan = (payload.changes as Record<string, unknown>).next_plan_id;
        if (typeof nextPlan === "string" && nextPlan) setNextPlanId(nextPlan);
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
        if (process.env.NODE_ENV !== "production") console.error("plan ws parse error", error);
      }
    };

    return () => {
      socket.close();
    };
  }, [planId]);

  useEffect(() => {
    if (!upperSymbol) return;
    const streamUrl = `${API_BASE_URL}/stream/${encodeURIComponent(upperSymbol)}`;
    const source = new EventSource(streamUrl);

    source.onopen = () => setPriceStatus("connected");
    source.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data) as SymbolTickEvent;
        if (payload.t === "tick") {
          if (!analysisMeasuredRef.current && analysisStartRef.current != null) {
            const end = now();
            if (end != null) logBudget("analysis:first-tick", end - analysisStartRef.current, 1200);
            analysisMeasuredRef.current = true;
          }
          const timestamp = Math.floor(new Date(payload.ts).getTime() / 1000);
          setPriceSeries((prev) => {
            const next = [...prev, { time: timestamp, value: payload.p }];
            if (next.length > MAX_POINTS) next.splice(0, next.length - MAX_POINTS);
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
        if (process.env.NODE_ENV !== "production") console.error("symbol stream error", error);
      }
    };

    source.onerror = () => {
      setPriceStatus("disconnected");
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

  useEffect(() => {
    if (!showSupportingLevels) setHighlightedLevel(null);
  }, [showSupportingLevels]);

  useEffect(() => {
    setShowSupportingLevels(chartParamsRaw.supportingLevels !== "0");
    setUiTheme(chartParamsRaw.theme === "light" ? "light" : "dark");
  }, [chartParamsRaw.supportingLevels, chartParamsRaw.theme]);

  useEffect(() => {
    analysisStartRef.current = now();
    analysisMeasuredRef.current = false;
  }, [planId]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    if (!highlightedLevel) return;
    if (window.innerWidth < 1024) setSheetOpen(true);
  }, [highlightedLevel]);

  const interactiveUrl = plan.charts?.interactive ?? plan.chart_url ?? initialSnapshot.chart_url ?? null;

  const handleThemeToggle = useCallback(() => {
    setUiTheme((prev) => (prev === "light" ? "dark" : "light"));
  }, []);

  const handleActionsDock = useCallback(
    (action: ActionKey) => {
      if (action === "share" && interactiveUrl && typeof window !== "undefined") {
        window.open(interactiveUrl, "_blank", "noopener");
      }
      if (action === "size" || action === "coach") {
        setSheetOpen(true);
      }
      if (action === "size" || action === "validate") {
        console.info(`[perf] ${actionLabel(action)} triggered — downstream call budget 800ms`);
      }
      setCoachingLog((prev) => [
        {
          id: `dock-${action}-${Date.now()}`,
          timestamp: new Date().toISOString(),
          message: `${actionLabel(action)} triggered · routed to automation queue.`,
          tone: action === "validate" ? "warning" : "neutral",
        },
        ...prev.slice(0, 49),
      ]);
    },
    [interactiveUrl],
  );

  const statusStrip = (
    <StatusStrip
      wsStatus={wsStatus}
      priceStatus={priceStatus}
      dataAgeSeconds={dataAgeSeconds}
      riskBanner={plan.risk_block && typeof plan.risk_block === "object" ? (plan.risk_block as Record<string, string | undefined>).risk_note ?? null : null}
      sessionBanner={plan.session_state?.banner ?? null}
      theme={uiTheme}
      onToggleTheme={handleThemeToggle}
    />
  );

  const actionsDock = <ActionsDock onAction={handleActionsDock} theme={uiTheme} />;

  const planPanelProps = {
    plan,
    structured,
    badges: plan.badges ?? structured?.badges ?? [],
    confidence: plan.confidence ?? chartUiState.confidence,
    supportingLevels,
    highlightedLevel,
    onSelectLevel: (level: SupportingLevel | null) => setHighlightedLevel(level),
    theme: uiTheme,
  };

  const desktopPlanPanel = <PlanPanel {...planPanelProps} />;

  const mobileSheet = (
    <MobilePlanSheet
      open={sheetOpen}
      onToggle={() => setSheetOpen((prev) => !prev)}
      actions={<ActionsDock layout="horizontal" onAction={handleActionsDock} theme={uiTheme} />}
      theme={uiTheme}
    >
      <PlanPanel {...planPanelProps} />
    </MobilePlanSheet>
  );

  return (
    <WebviewShell
      statusStrip={statusStrip}
      chartPanel={
        <ChartContainer
          symbol={upperSymbol}
          interval={chartParamsRaw.interval}
      chartTheme={chartParamsRaw.theme}
      uiTheme={uiTheme}
          uiState={chartUiState}
          priceSeries={priceSeries}
          lastPrice={planState.lastPrice ?? undefined}
          entry={entryLevel ?? undefined}
          stop={planState.trailingStop ?? baseStop ?? undefined}
          trailingStop={planState.trailingStop ?? undefined}
          targets={targets}
          supportingLevels={supportingLevels}
          showSupportingLevels={showSupportingLevels}
          onToggleSupportingLevels={() => setShowSupportingLevels((prev) => !prev)}
          onHighlightLevel={setHighlightedLevel}
          highlightedLevel={highlightedLevel}
          interactiveUrl={interactiveUrl}
        >
          <CoachingTimeline events={coachingLog} theme={uiTheme} />
          {nextPlanId ? (
            <p className="mt-3 text-xs uppercase tracking-[0.25em] text-neutral-500">
              Next best action available · <a className="text-emerald-200 underline-offset-4 hover:underline" href={`/plan/${encodeURIComponent(nextPlanId)}`}>Switch to {nextPlanId}</a>
            </p>
          ) : null}
        </ChartContainer>
      }
      planPanel={desktopPlanPanel}
      actionsDock={actionsDock}
      mobileSheet={mobileSheet}
      theme={uiTheme}
    />
  );
}

function bootstrapCoachingLog(snapshot: PlanSnapshot): CoachingEvent[] {
  const banner = snapshot.plan.session_state?.banner;
  const note = snapshot.plan.notes;
  const items: CoachingEvent[] = [];
  if (banner) {
    items.push({
      id: `banner-${Date.now()}`,
      timestamp: snapshot.plan.session_state?.as_of ?? new Date().toISOString(),
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
}

function computeDataAge(asOf?: string | null): number | null {
  if (!asOf) return null;
  const asOfDate = new Date(asOf);
  if (Number.isNaN(asOfDate.getTime())) return null;
  return (Date.now() - asOfDate.getTime()) / 1000;
}

function CoachingTimeline({ events, theme }: { events: CoachingEvent[]; theme: "dark" | "light" }) {
  const isLight = theme === "light";
  return (
    <section
      className={clsx(
        "mt-6 space-y-2 rounded-3xl border p-5 text-sm",
        isLight ? "border-slate-200 bg-white text-slate-700" : "border-neutral-800/60 bg-neutral-900/40 text-neutral-200",
      )}
      aria-live="polite"
    >
      <header className="flex items-center justify-between">
        <h3 className={clsx("text-xs font-semibold uppercase tracking-[0.3em]", isLight ? "text-slate-500" : "text-neutral-400")}>Coaching timeline</h3>
        <span className={clsx("text-[0.65rem] uppercase tracking-[0.3em]", isLight ? "text-slate-400" : "text-neutral-500")}>
          Latest {Math.min(events.length, 5)} events
        </span>
      </header>
      <ol className="space-y-2">
        {events.length === 0 ? (
          <li
            className={clsx(
              "rounded-xl border px-4 py-3",
              isLight ? "border-slate-200 bg-white text-slate-500" : "border-neutral-800/70 bg-neutral-900/60 text-neutral-400",
            )}
          >
            Waiting for live updates…
          </li>
        ) : (
          events.slice(0, 5).map((event) => (
            <li
              key={event.id}
              className={clsx(
                "rounded-xl border px-4 py-3",
                event.tone === "positive" &&
                  (isLight ? "border-emerald-500/40 bg-emerald-400/20 text-emerald-700" : "border-emerald-500/40 bg-emerald-500/10 text-emerald-100"),
                event.tone === "warning" &&
                  (isLight ? "border-amber-500/40 bg-amber-400/20 text-amber-700" : "border-amber-500/50 bg-amber-500/10 text-amber-100"),
                event.tone === "neutral" &&
                  (isLight ? "border-slate-200 bg-white text-slate-700" : "border-neutral-800/70 bg-neutral-900/70 text-neutral-200"),
              )}
            >
              <div className={clsx("text-xs uppercase tracking-[0.2em]", isLight ? "text-slate-500" : "text-neutral-400/80")}>
                {new Date(event.timestamp).toLocaleTimeString()}
              </div>
              <p className={clsx("mt-1 leading-relaxed", isLight ? "text-slate-700" : "text-neutral-200")}>{event.message}</p>
            </li>
          ))
        )}
      </ol>
    </section>
  );
}

function MobilePlanSheet({
  open,
  onToggle,
  actions,
  children,
  theme,
}: {
  open: boolean;
  onToggle: () => void;
  actions?: React.ReactNode;
  children: React.ReactNode;
  theme: "dark" | "light";
}) {
  const isLight = theme === "light";
  return (
    <div className="lg:hidden">
      <button
        type="button"
        onClick={onToggle}
        className={clsx(
          "fixed bottom-6 left-1/2 z-40 -translate-x-1/2 rounded-full border px-6 py-3 text-xs font-semibold uppercase tracking-[0.3em] shadow-lg backdrop-blur",
          isLight ? "border-emerald-500/50 bg-emerald-400/20 text-emerald-700 shadow-emerald-400/40" : "border-emerald-500/50 bg-emerald-500/20 text-emerald-100 shadow-emerald-500/30",
        )}
      >
        {open ? "Close Plan" : "Plan Summary"}
      </button>
      <div
        className={clsx(
          "fixed inset-x-0 bottom-0 z-30 transform rounded-t-3xl border px-4 pb-8 pt-4 shadow-2xl backdrop-blur transition-transform duration-300",
          isLight ? "border-slate-200 bg-white/95 text-slate-700 shadow-slate-200/80" : "border-neutral-900/80 bg-neutral-950/95 text-neutral-200 shadow-black/60",
          open ? "translate-y-0" : "translate-y-[calc(100%-4.5rem)]",
        )}
      >
        <div
          className={clsx("mx-auto h-1 w-16 rounded-full", isLight ? "bg-slate-300" : "bg-neutral-700/70")}
          onClick={onToggle}
          role="presentation"
        />
        <div className="mt-4 space-y-4">
          {actions ? <div className="flex flex-wrap justify-center gap-2">{actions}</div> : null}
          <div className="max-h-[65vh] overflow-y-auto pr-1">{children}</div>
        </div>
      </div>
    </div>
  );
}

function actionLabel(action: "size" | "alerts" | "validate" | "coach" | "share"): string {
  switch (action) {
    case "size":
      return "Sizing request";
    case "alerts":
      return "Alerts request";
    case "validate":
      return "Plan validation";
    case "coach":
      return "Coach question";
    case "share":
      return "Share chart";
    default:
      return "Action";
  }
}

function logBudget(name: string, duration: number, budgetMs: number): void {
  const message = `[perf] ${name} ${duration.toFixed(1)}ms (budget ${budgetMs}ms)`;
  if (duration > budgetMs) console.warn(`${message} ⚠️`);
  else console.info(`${message} ✅`);
}

function extractPlanLatency(plan: PlanSnapshot["plan"]): number | null {
  const maybe = (value: unknown) => (typeof value === "number" && Number.isFinite(value) ? value : null);
  const base = plan as Record<string, unknown>;
  const meta = base.meta as Record<string, unknown> | undefined;
  const metrics = base.metrics as Record<string, unknown> | undefined;
  const candidates: Array<number | null> = [
    maybe(base["latency_ms"]),
    maybe(base["plan_latency_ms"]),
    maybe(base["plan_ms"]),
    meta ? maybe(meta["latency_ms"]) ?? maybe(meta["plan_ms"]) : null,
    metrics ? maybe(metrics["latency_ms"]) ?? maybe(metrics["plan_ms"]) : null,
  ];
  return candidates.find((value): value is number => value != null) ?? null;
}

function now(): number | null {
  return typeof performance !== "undefined" ? performance.now() : null;
}
