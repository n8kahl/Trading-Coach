"use client";

import { useCallback, useEffect, useMemo, useRef } from "react";
import { ensureCanonicalChartUrl } from "@/lib/chartUrl";
import { API_BASE_URL, WATCHLIST_LIMIT, WATCHLIST_STYLE, WATCHLIST_UNIVERSE, withAuthHeaders } from "@/lib/env";
import type { WatchlistItem } from "@/store/useStore";
import { useStore } from "@/store/useStore";

const REFRESH_INTERVAL_MS = 60_000;

function toNumber(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string") {
    const parsed = Number.parseFloat(value);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }
  return null;
}

type RawScanEntry = Record<string, any>;

function extractPlanId(entry: RawScanEntry): string | null {
  return (
    entry.plan_id ??
    entry.plan?.plan_id ??
    entry.plan?.plan?.plan_id ??
    entry.plan?.plan?.plan?.plan_id ??
    null
  );
}

function extractSymbol(entry: RawScanEntry, fallback: string): string {
  return (
    entry.symbol ??
    entry.plan?.symbol ??
    entry.plan?.plan?.symbol ??
    entry.plan?.plan?.plan?.symbol ??
    fallback.toUpperCase()
  );
}

function extractChartsParams(entry: RawScanEntry): Record<string, unknown> | null {
  const params =
    entry.charts?.params ??
    entry.plan?.charts_params ??
    entry.plan?.plan?.charts_params ??
    entry.plan?.plan?.plan?.charts_params ??
    null;
  return params && typeof params === "object" ? params : null;
}

async function resolveChartUrl(entry: RawScanEntry, signal: AbortSignal): Promise<string | null> {
  const existing = ensureCanonicalChartUrl(
    entry.chart_url ??
      entry.trade_detail ??
      entry.charts?.interactive ??
      entry.plan?.chart_url ??
      entry.plan?.trade_detail ??
      entry.plan?.charts?.interactive ??
      null,
  );
  if (existing) return existing;

  const params = extractChartsParams(entry);
  if (!params || !API_BASE_URL) return null;

  try {
    const response = await fetch(`${API_BASE_URL}/gpt/chart-url`, {
      method: "POST",
      headers: withAuthHeaders({ "Content-Type": "application/json", Accept: "application/json" }),
      body: JSON.stringify(params),
      cache: "no-store",
      signal,
    });
    if (!response.ok) {
      return null;
    }
    const payload = await response.json();
    return ensureCanonicalChartUrl(payload?.interactive ?? null);
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      return null;
    }
    if (process.env.NODE_ENV !== "production") {
      console.warn("[watchlist] chart-url failed", error);
    }
    return null;
  }
}

function buildWatchlistItem(entry: RawScanEntry, chartUrl: string | null): WatchlistItem | null {
  const planId = extractPlanId(entry);
  if (!planId) return null;

  const meta = entry.meta ?? entry.plan?.meta ?? entry.plan?.plan?.meta ?? {};
  const actionableSoonValue =
    entry.actionable_soon ??
    entry.plan?.actionable_soon ??
    entry.plan?.plan?.actionable_soon ??
    meta?.actionable_soon ??
    false;

  const entryDistancePct =
    toNumber(entry.entry_distance_pct) ??
    toNumber(entry.plan?.entry_distance_pct) ??
    toNumber(meta.entry_distance_pct);

  const entryDistanceAtr =
    toNumber(entry.entry_distance_atr) ??
    toNumber(entry.plan?.entry_distance_atr) ??
    toNumber(meta.entry_distance_atr);

  const barsToTrigger =
    toNumber(entry.bars_to_trigger) ??
    toNumber(entry.plan?.bars_to_trigger) ??
    toNumber(meta.bars_to_trigger);

  return {
    plan_id: planId,
    symbol: extractSymbol(entry, ""),
    style: entry.style ?? entry.plan?.style ?? entry.plan?.plan?.style ?? null,
    plan_url: `/plan/${encodeURIComponent(planId)}`,
    chart_url: chartUrl,
    actionable_soon: Boolean(actionableSoonValue),
    entry_distance_pct: entryDistancePct,
    entry_distance_atr: entryDistanceAtr,
    bars_to_trigger: barsToTrigger,
    meta: typeof meta === "object" ? meta : undefined,
    raw: entry,
  };
}

function sortWatchlist(items: WatchlistItem[]): WatchlistItem[] {
  return [...items].sort((a, b) => {
    const aSoon = a.actionable_soon ? 0 : 1;
    const bSoon = b.actionable_soon ? 0 : 1;
    if (aSoon !== bSoon) return aSoon - bSoon;

    const aDist = a.entry_distance_pct ?? Number.POSITIVE_INFINITY;
    const bDist = b.entry_distance_pct ?? Number.POSITIVE_INFINITY;
    if (aDist !== bDist) return aDist - bDist;

    return a.symbol.localeCompare(b.symbol);
  });
}

async function fetchWatchlist(signal: AbortSignal): Promise<WatchlistItem[]> {
  if (!API_BASE_URL) return [];
  const body = {
    universe: WATCHLIST_UNIVERSE,
    style: WATCHLIST_STYLE,
    limit: WATCHLIST_LIMIT,
  };
  const response = await fetch(`${API_BASE_URL}/gpt/scan`, {
    method: "POST",
    headers: withAuthHeaders({ "Content-Type": "application/json", Accept: "application/json" }),
    body: JSON.stringify(body),
    cache: "no-store",
    signal,
  });
  if (!response.ok) {
    throw new Error(`watchlist fetch failed (${response.status})`);
  }
  const payload = await response.json();
  const rows: RawScanEntry[] = Array.isArray(payload?.plans)
    ? payload.plans
    : Array.isArray(payload?.results)
      ? payload.results
      : [];
  const enriched = await Promise.all(
    rows.map(async (entry) => {
      const chartUrl = await resolveChartUrl(entry, signal);
      return buildWatchlistItem(entry, chartUrl);
    }),
  );
  return sortWatchlist(enriched.filter((item): item is WatchlistItem => item !== null));
}

export function useWatchlist(autoRefresh = true) {
  const watchlist = useStore((state) => state.watchlist);
  const setWatchlist = useStore((state) => state.setWatchlist);
  const setWatchlistStatus = useStore((state) => state.setWatchlistStatus);
  const inFlightRef = useRef<AbortController | null>(null);

  const refresh = useCallback(async () => {
    if (inFlightRef.current) {
      inFlightRef.current.abort();
    }
    const controller = new AbortController();
    inFlightRef.current = controller;
    try {
      setWatchlistStatus("loading");
      const items = await fetchWatchlist(controller.signal);
      setWatchlist(items);
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") {
        return;
      }
      const message = error instanceof Error ? error.message : "Watchlist request failed";
      setWatchlistStatus("error", message);
    } finally {
      inFlightRef.current = null;
    }
  }, [setWatchlist, setWatchlistStatus]);

  useEffect(() => {
    refresh().catch(() => {});
    if (!autoRefresh) return undefined;
    const interval = window.setInterval(() => {
      refresh().catch(() => {});
    }, REFRESH_INTERVAL_MS);
    return () => window.clearInterval(interval);
  }, [autoRefresh, refresh]);

  useEffect(
    () => () => {
      if (inFlightRef.current) {
        inFlightRef.current.abort();
        inFlightRef.current = null;
      }
    },
    [],
  );

  return useMemo(
    () => ({
      ...watchlist,
      refresh,
    }),
    [watchlist, refresh],
  );
}
