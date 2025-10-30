import { useCallback, useEffect, useMemo, useState } from "react";
import type { CandlestickData } from "lightweight-charts";
import { API_BASE_URL } from "@/lib/env";

type Status = "idle" | "loading" | "ready" | "error";

export type PriceBar = CandlestickData & { volume?: number };

const MAX_LOOKBACK_SECONDS = 60 * 60 * 24 * 45; // ~45 days safeguard for higher timeframes
const MAX_BARS = 1800;

function resolutionToSeconds(resolution: string): number {
  const token = (resolution || "").toString().trim().toUpperCase();
  if (!token) return 60;
  if (token.endsWith("D")) {
    const days = Number.parseInt(token.replace("D", ""), 10);
    return Number.isFinite(days) && days > 0 ? days * 24 * 60 * 60 : 24 * 60 * 60;
  }
  if (token.endsWith("W")) {
    const weeks = Number.parseInt(token.replace("W", ""), 10);
    return Number.isFinite(weeks) && weeks > 0 ? weeks * 7 * 24 * 60 * 60 : 7 * 24 * 60 * 60;
  }
  if (token.endsWith("H")) {
    const hours = Number.parseInt(token.replace("H", ""), 10);
    return Number.isFinite(hours) && hours > 0 ? hours * 60 * 60 : 60 * 60;
  }
  const minutes = Number.parseInt(token, 10);
  if (!Number.isFinite(minutes) || minutes <= 0) {
    return 60;
  }
  return minutes * 60;
}

function sanitizeSymbol(symbol: string | null | undefined): string | null {
  if (!symbol) return null;
  const token = symbol.trim();
  return token ? token : null;
}

function clampLookback(resolutionSeconds: number): { from: number; to: number } {
  const nowSec = Math.floor(Date.now() / 1000);
  const spanByResolution = resolutionSeconds * MAX_BARS;
  const span = Math.min(Math.max(spanByResolution, resolutionSeconds * 200), MAX_LOOKBACK_SECONDS);
  const from = Math.max(nowSec - span, 0);
  return { from, to: nowSec };
}

function toNumeric(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const parsed = Number.parseFloat(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return null;
}

function normalizeBars(payload: Record<string, unknown>): PriceBar[] {
  const times = Array.isArray(payload.t) ? payload.t : [];
  const opens = Array.isArray(payload.o) ? payload.o : [];
  const highs = Array.isArray(payload.h) ? payload.h : [];
  const lows = Array.isArray(payload.l) ? payload.l : [];
  const closes = Array.isArray(payload.c) ? payload.c : [];
  const volumes = Array.isArray(payload.v) ? payload.v : [];
  const bars: PriceBar[] = [];
  const length = Math.min(times.length, opens.length, highs.length, lows.length, closes.length);
  for (let idx = 0; idx < length; idx += 1) {
    const rawTime = toNumeric(times[idx]);
    const open = toNumeric(opens[idx]);
    const high = toNumeric(highs[idx]);
    const low = toNumeric(lows[idx]);
    const close = toNumeric(closes[idx]);
    if (rawTime == null || open == null || high == null || low == null || close == null) {
      continue;
    }
    const normalizedTime = rawTime > 1_000_000_000_000 ? Math.trunc(rawTime / 1000) : Math.trunc(rawTime);
    bars.push({
      time: normalizedTime as PriceBar["time"],
      open,
      high,
      low,
      close,
      volume: toNumeric(volumes[idx]) ?? undefined,
    });
  }
  return bars.sort((a, b) => Number(a.time) - Number(b.time));
}

function mergeBars(existing: PriceBar[], incoming: PriceBar[]): PriceBar[] {
  if (existing.length === 0) return incoming;
  if (incoming.length === 0) return existing;
  const merged = new Map<number, PriceBar>();
  existing.forEach((bar) => merged.set(Number(bar.time), bar));
  incoming.forEach((bar) => merged.set(Number(bar.time), bar));
  return Array.from(merged.values()).sort((a, b) => Number(a.time) - Number(b.time));
}

export function usePriceSeries(symbol: string | null | undefined, resolution: string, deps: unknown[] = []) {
  const sanitizedSymbol = sanitizeSymbol(symbol);
  const [bars, setBars] = useState<PriceBar[]>([]);
  const [status, setStatus] = useState<Status>(sanitizedSymbol ? "loading" : "idle");
  const [error, setError] = useState<Error | null>(null);
  const [reloadToken, setReloadToken] = useState(0);
  const debug = typeof window !== "undefined" && new URLSearchParams(window.location.search).get("dev") !== null;

  const reload = useCallback(() => {
    setReloadToken((token) => token + 1);
    setStatus((prev) => (prev === "idle" ? "loading" : prev));
  }, []);

  useEffect(() => {
    if (!sanitizedSymbol) {
      setBars([]);
      return;
    }
    setBars([]);
  }, [sanitizedSymbol, resolution]);

  useEffect(() => {
    if (!sanitizedSymbol) {
      setBars([]);
      setStatus("idle");
      setError(null);
      return;
    }
    let cancelled = false;
    const controller = new AbortController();
    const { from, to } = clampLookback(resolutionToSeconds(resolution));

    async function load() {
      try {
        setStatus("loading");
        setError(null);
        const qs = new URLSearchParams({
          symbol: sanitizedSymbol,
          resolution,
          from: String(from),
          to: String(to),
        });
        const url = `${API_BASE_URL}/tv-api/bars?${qs.toString()}`;
        if (debug) {
          // eslint-disable-next-line no-console
          console.log("[usePriceSeries] request", { url, symbol: sanitizedSymbol, resolution, from, to });
        }
        const response = await fetch(url, {
          cache: "no-store",
          signal: controller.signal,
          headers: {
            "Cache-Control": "no-store",
          },
        });
        if (!response.ok) {
          throw new Error(`bars request failed (${response.status})`);
        }
        const payload = await response.json();
        if (cancelled) return;
        if (!payload || payload.s !== "ok") {
          setBars([]);
          setStatus("ready");
          if (debug) {
            // eslint-disable-next-line no-console
            console.log("[usePriceSeries] payload not ok", payload);
          }
          return;
        }
        const nextBars = normalizeBars(payload);
        if (debug) {
          const n = nextBars.length;
          const first = n ? Number(nextBars[0].time) : null;
          const last = n ? Number(nextBars[n - 1].time) : null;
          // eslint-disable-next-line no-console
          console.log("[usePriceSeries] normalized", { count: n, first, last });
        }
        setBars((prev) => mergeBars(prev, nextBars));
        setStatus("ready");
     } catch (err) {
        if (cancelled || (err instanceof DOMException && err.name === "AbortError")) {
          return;
        }
        setError(err instanceof Error ? err : new Error("Failed to load price series"));
        setStatus("error");
        if (debug) {
          // eslint-disable-next-line no-console
          console.log("[usePriceSeries] error", err);
        }
      }
    }

    load();

    return () => {
      cancelled = true;
      controller.abort();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sanitizedSymbol, resolution, reloadToken, ...deps]);

  return useMemo(
    () => ({
      bars,
      status,
      error,
      reload,
    }),
    [bars, status, error, reload],
  );
}

export type UsePriceSeriesReturn = ReturnType<typeof usePriceSeries>;
export type { PriceBar as PriceSeriesCandle };
