import { useEffect, useMemo, useState } from "react";
import type { LineData } from "lightweight-charts";
import { API_BASE_URL } from "@/lib/env";

type Status = "idle" | "loading" | "ready" | "error";

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

export function usePriceSeries(symbol: string | null | undefined, resolution: string, deps: unknown[] = []) {
  const sanitizedSymbol = sanitizeSymbol(symbol);
  const [series, setSeries] = useState<LineData[]>([]);
  const [status, setStatus] = useState<Status>(sanitizedSymbol ? "loading" : "idle");
  const [error, setError] = useState<Error | null>(null);
  const [reloadToken, setReloadToken] = useState(0);

  useEffect(() => {
    if (!sanitizedSymbol) {
      setSeries([]);
      setStatus("idle");
      setError(null);
      return;
    }
    let cancelled = false;
    const secondsPerBar = resolutionToSeconds(resolution);
    const nowSec = Math.floor(Date.now() / 1000);
    const spanSeconds = Math.min(secondsPerBar * 1200, 60 * 60 * 24 * 30);
    const from = Math.max(nowSec - spanSeconds, 0);

    const controller = new AbortController();
    async function load() {
      try {
        setStatus("loading");
        setError(null);
        const qs = new URLSearchParams({
          symbol: sanitizedSymbol,
          resolution,
          from: String(from),
          to: String(nowSec),
        });
        const response = await fetch(`${API_BASE_URL}/tv-api/bars?${qs.toString()}`, {
          cache: "no-store",
          signal: controller.signal,
        });
        if (!response.ok) {
          throw new Error(`bars request failed (${response.status})`);
        }
        const payload = await response.json();
        if (cancelled) return;
        if (!payload || payload.s !== "ok" || !Array.isArray(payload.t)) {
          setSeries([]);
          setStatus("ready");
          return;
        }
        const next: LineData[] = payload.t
          .map((time: unknown, idx: number) => {
            const t = typeof time === "number" ? time : Number(time);
            if (!Number.isFinite(t)) return null;
            const close = Array.isArray(payload.c) ? payload.c[idx] : undefined;
            const value = typeof close === "number" ? close : Number(close);
            if (!Number.isFinite(value)) return null;
            return { time: t as unknown as LineData["time"], value };
          })
          .filter((item): item is LineData => item !== null);
        setSeries(next);
        setStatus("ready");
      } catch (err) {
        if (cancelled || (err instanceof DOMException && err.name === "AbortError")) return;
        setError(err instanceof Error ? err : new Error("Failed to load price series"));
        setStatus("error");
      }
    }
    load();
    return () => {
      cancelled = true;
      controller.abort();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sanitizedSymbol, resolution, reloadToken, ...deps]);

  const reload = () => setReloadToken((token) => token + 1);

  return useMemo(
    () => ({
      series,
      status,
      error,
      reload,
    }),
    [series, status, error],
  );
}

export type UsePriceSeriesReturn = ReturnType<typeof usePriceSeries>;
