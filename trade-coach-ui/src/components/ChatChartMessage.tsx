'use client';

import { useEffect, useMemo, useState } from "react";
import Image from "next/image";
import { API_BASE_URL, PUBLIC_BASE_URL, withAuthHeaders } from "@/lib/env";

type PlanDirection = "long" | "short";

type ChatPlan = {
  entry: number;
  stop: number;
  tps: number[];
  direction: PlanDirection;
  ema?: number[];
  levels?: Record<string, number>;
};

type ChatChartMessageProps = {
  symbol: string;
  interval: "1m" | "5m" | "15m" | "1h" | "1D";
  plan: ChatPlan;
  focus?: "plan";
  centerTime?: "latest" | number;
  theme?: "light" | "dark";
  scalePlan?: "auto" | "off";
};

type ChartLinksResponse = {
  interactive: string;
  png?: string | null;
};

type LoadState =
  | { status: "idle" | "loading" }
  | { status: "ready"; liveUrl: string; pngUrl: string }
  | { status: "error"; message: string };

const NUMBER_FORMAT = new Intl.NumberFormat(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });

function clampToFinite(value: number | undefined | null): number | null {
  if (!Number.isFinite(value)) {
    return null;
  }
  return Number(value);
}

function encodeLevels(levels: Record<string, number> | undefined): string | undefined {
  if (!levels) return undefined;
  const entries = Object.entries(levels)
    .filter(([, value]) => Number.isFinite(value))
    .sort(([a], [b]) => a.localeCompare(b));
  if (!entries.length) return undefined;
  return entries.map(([label, value]) => `${value}|${label}`).join(";");
}

function computeRiskReward(entry: number | null, stop: number | null, target: number | null, direction: PlanDirection): number | null {
  if (!Number.isFinite(entry) || !Number.isFinite(stop) || !Number.isFinite(target)) {
    return null;
  }
  const e = Number(entry);
  const s = Number(stop);
  const t = Number(target);
  const risk = direction === "long" ? e - s : s - e;
  const reward = direction === "long" ? t - e : e - t;
  if (risk <= 0 || reward <= 0) {
    return null;
  }
  return reward / risk;
}

function buildPngUrl(params: URLSearchParams): string {
  const base = PUBLIC_BASE_URL || API_BASE_URL;
  const trimmed = base.replace(/\/$/, "");
  const query = params.toString();
  return query ? `${trimmed}/charts/png?${query}` : `${trimmed}/charts/png`;
}

export default function ChatChartMessage({ symbol, interval, plan, focus, centerTime, theme = "dark", scalePlan = "auto" }: ChatChartMessageProps) {
  const [state, setState] = useState<LoadState>({ status: "idle" });
  const [retryCounter, setRetryCounter] = useState(0);

  const normalized = useMemo(() => {
    const upperSymbol = symbol.toUpperCase();
    const entry = clampToFinite(plan.entry);
    const stop = clampToFinite(plan.stop);
    const filteredTps = (plan.tps || []).filter((value) => Number.isFinite(value)).map((value) => Number(value));
    const emaSpans = (plan.ema || []).filter((value) => Number.isFinite(value)).map((value) => Math.trunc(Number(value)));
    const levelsToken = encodeLevels(plan.levels);
    const tpCsv = filteredTps.length ? filteredTps.map((value) => NUMBER_FORMAT.format(value)).join(",") : undefined;
    const emaCsv = emaSpans.length ? emaSpans.join(",") : undefined;
    const centerToken =
      typeof centerTime === "number" && Number.isFinite(centerTime)
        ? Math.trunc(Number(centerTime)).toString()
        : typeof centerTime === "string"
          ? centerTime
          : undefined;

    const body: Record<string, unknown> = {
      symbol: upperSymbol,
      interval,
      direction: plan.direction,
      entry,
      stop,
      tp: tpCsv,
      ema: emaCsv,
      focus,
      center_time: centerToken,
      scale_plan: scalePlan,
      theme,
    };
    if (levelsToken) {
      body.levels = levelsToken;
    }

    Object.keys(body).forEach((key) => {
      if (body[key] === undefined || body[key] === null || body[key] === "") {
        delete body[key];
      }
    });

    const pngParams = new URLSearchParams();
    pngParams.set("symbol", upperSymbol);
    pngParams.set("interval", interval);
    if (Number.isFinite(entry)) {
      pngParams.set("entry", NUMBER_FORMAT.format(Number(entry)));
    }
    if (Number.isFinite(stop)) {
      pngParams.set("stop", NUMBER_FORMAT.format(Number(stop)));
    }
    if (tpCsv) {
      pngParams.set("tp", tpCsv);
    }
    if (emaCsv) {
      pngParams.set("ema", emaCsv);
    }
    if (levelsToken) {
      pngParams.set("levels", levelsToken);
    }
    if (focus) {
      pngParams.set("focus", focus);
    }
    if (centerToken) {
      pngParams.set("center_time", centerToken);
    }
    if (scalePlan) {
      pngParams.set("scale_plan", scalePlan);
    }
    pngParams.set("theme", theme);

    const signature = JSON.stringify({
      body,
      png: Array.from(pngParams.entries()),
    });

    return {
      body,
      pngParams,
      signature,
      upperSymbol,
      emaSpans,
      levelsToken,
      entry,
      stop,
      tp1: filteredTps.length ? filteredTps[0] : null,
    };
  }, [symbol, interval, plan.entry, plan.stop, plan.tps, plan.direction, plan.ema, plan.levels, focus, centerTime, theme, scalePlan]);

  useEffect(() => {
    let aborted = false;
    const controller = new AbortController();

    const run = async () => {
      setState({ status: "loading" });
      try {
        const headers = withAuthHeaders({
          "Content-Type": "application/json",
        });
        const response = await fetch(`${API_BASE_URL}/gpt/chart-url`, {
          method: "POST",
          headers,
          body: JSON.stringify(normalized.body),
          signal: controller.signal,
        });
        if (!response.ok) {
          throw new Error(`chart-url request failed (${response.status})`);
        }
        const data: ChartLinksResponse = await response.json();
        if (aborted) return;
        const liveUrl = data.interactive;
        if (!liveUrl) {
          throw new Error("chart-url response missing interactive link");
        }
        const pngUrl = data.png && data.png.length > 0 ? data.png : buildPngUrl(normalized.pngParams);
        setState({ status: "ready", liveUrl, pngUrl });
      } catch (error) {
        if (aborted) return;
        const message = error instanceof Error ? error.message : "Unknown chart error";
        setState({ status: "error", message });
      }
    };

    run();

    return () => {
      aborted = true;
      controller.abort();
    };
  }, [normalized.signature, normalized.body, normalized.pngParams, retryCounter]);

  const retry = () => setRetryCounter((value) => value + 1);

  const riskReward = useMemo(
    () => computeRiskReward(normalized.entry, normalized.stop, normalized.tp1, plan.direction),
    [normalized.entry, normalized.stop, normalized.tp1, plan.direction],
  );

  const emaBadge = normalized.emaSpans.length ? `EMA${normalized.emaSpans.join("/")}` : null;

  return (
    <div className="flex flex-col gap-2 text-sm text-slate-200">
      {state.status === "ready" && (
        <>
          <a
            href={state.liveUrl}
            target="_blank"
            rel="noreferrer"
            className="block overflow-hidden rounded-lg border border-slate-700/80 bg-slate-950/40 transition hover:border-sky-500/70 hover:shadow-lg hover:shadow-sky-900/40"
          >
            <Image
              src={state.pngUrl}
              alt={`${normalized.upperSymbol} ${interval} trading plan`}
              className="h-auto w-full"
              width={1280}
              height={720}
              unoptimized
            />
          </a>
          <div className="flex flex-wrap items-center gap-2 text-xs text-slate-300">
            <span className="font-semibold text-slate-100">
              {normalized.upperSymbol} · {interval.toUpperCase()}
            </span>
            <span className="rounded-full bg-slate-800/80 px-2 py-0.5 text-slate-200">{plan.direction.toUpperCase()}</span>
            {riskReward && (
              <span className="rounded-full bg-emerald-900/40 px-2 py-0.5 text-emerald-200">
                R:R {NUMBER_FORMAT.format(riskReward)}
              </span>
            )}
            {emaBadge && <span className="rounded-full bg-slate-800/80 px-2 py-0.5 text-slate-200">{emaBadge}</span>}
            {plan.levels &&
              Object.keys(plan.levels)
                .slice(0, 4)
                .map((label) => (
                  <span key={label} className="rounded-full bg-slate-800/60 px-2 py-0.5 text-slate-300">
                    {label}
                  </span>
                ))}
          </div>
          <p className="text-xs text-slate-400">
            Click to open the live chart with focus on the trade plan. Updates stream in via `/tv`.
          </p>
        </>
      )}
      {state.status === "loading" && <div className="text-xs text-slate-400">Loading chart preview…</div>}
      {state.status === "error" && (
        <div className="rounded-md border border-red-500/40 bg-red-950/40 p-3 text-xs text-red-200">
          <p className="mb-2 font-semibold">Unable to load chart preview.</p>
          <p className="mb-3 opacity-80">{state.message}</p>
          <button
            type="button"
            onClick={retry}
            className="rounded-md bg-red-500/20 px-3 py-1 font-medium text-red-100 transition hover:bg-red-500/40"
          >
            Retry
          </button>
        </div>
      )}
    </div>
  );
}
