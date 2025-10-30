import { API_BASE_URL } from "@/lib/env";

export type TVBar = {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
};

type HistoryCallbackMeta = {
  noData: boolean;
  nextTime?: number;
};

type HistoryCallback = (bars: TVBar[], meta: HistoryCallbackMeta) => void;
type ErrorCallback = (reason?: string) => void;

type ResolveCallback = (symbolInfo: TVSymbolInfo) => void;

type SubscribeCallback = (bar: TVBar) => void;

export type TVSymbolInfo = {
  name: string;
  ticker: string;
  description: string;
  session: string;
  timezone: string;
  pricescale: number;
  minmov: number;
  supported_resolutions: string[];
  has_intraday: boolean;
  has_no_volume: boolean;
  has_weekly_and_monthly: boolean;
  data_status: string;
  type?: string;
  exchange?: string;
  volume_precision?: number;
};

export type TradingViewDatafeedOptions = {
  baseUrl?: string;
  onBatchLoaded?: (symbol: string, resolution: string, bars: TVBar[]) => void;
  onRealtimeBar?: (symbol: string, resolution: string, bar: TVBar) => void;
};

type SubscriberRecord = {
  timer: number;
  callback: SubscribeCallback;
  symbol: string;
  resolution: string;
};

type UdfBarsResponse = {
  s: "ok" | "no_data" | "error";
  t?: Array<number | string>;
  o?: Array<number | string>;
  h?: Array<number | string>;
  l?: Array<number | string>;
  c?: Array<number | string>;
  v?: Array<number | string>;
  nextTime?: number | string;
  errmsg?: string;
  symbol?: string;
  resolution?: string;
};

const DEFAULT_RESOLUTIONS = ["1", "3", "5", "15", "30", "60", "120", "240", "1D"] as const;

function resolutionToSeconds(resolution: string): number {
  const token = (resolution || "").toString().trim().toUpperCase();
  if (!token) return 60;
  if (token.endsWith("D")) {
    const days = Number.parseInt(token.replace("D", ""), 10) || 1;
    return days * 24 * 60 * 60;
  }
  if (token.endsWith("W")) {
    const weeks = Number.parseInt(token.replace("W", ""), 10) || 1;
    return weeks * 7 * 24 * 60 * 60;
  }
  const minutes = Number.parseInt(token, 10);
  if (!Number.isFinite(minutes) || minutes <= 0) {
    return 60;
  }
  return minutes * 60;
}

function buildQuery(params: Record<string, string | number | null | undefined>): string {
  return Object.entries(params)
    .filter(([, value]) => value !== undefined && value !== null && value !== "")
    .map(([key, value]) => `${encodeURIComponent(key)}=${encodeURIComponent(String(value))}`)
    .join("&");
}

function coerceNumber(value: unknown, fallback = 0): number {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string") {
    const parsed = Number.parseFloat(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return fallback;
}

function parseUdfBars(payload: UdfBarsResponse): TVBar[] {
  const times = Array.isArray(payload.t) ? payload.t : [];
  const opens = Array.isArray(payload.o) ? payload.o : [];
  const highs = Array.isArray(payload.h) ? payload.h : [];
  const lows = Array.isArray(payload.l) ? payload.l : [];
  const closes = Array.isArray(payload.c) ? payload.c : [];
  const volumes = Array.isArray(payload.v) ? payload.v : [];

  const length = times.length;
  if (!length) {
    return [];
  }

  const bars: TVBar[] = [];
  for (let idx = 0; idx < length; idx += 1) {
    const rawTime = coerceNumber(times[idx], NaN);
    if (!Number.isFinite(rawTime)) {
      continue;
    }
    const timeMs = rawTime >= 10_000_000_000 ? Math.trunc(rawTime) : Math.trunc(rawTime * 1000);
    const bar: TVBar = {
      time: timeMs,
      open: coerceNumber(opens[idx], coerceNumber(opens[idx - 1], 0)),
      high: coerceNumber(highs[idx], coerceNumber(highs[idx - 1], 0)),
      low: coerceNumber(lows[idx], coerceNumber(lows[idx - 1], 0)),
      close: coerceNumber(closes[idx], coerceNumber(closes[idx - 1], 0)),
      volume: volumes.length ? coerceNumber(volumes[idx], 0) : 0,
    };
    bars.push(bar);
  }
  return bars;
}

export class TradingViewDatafeed {
  private readonly baseUrl: string;
  private readonly subscribers = new Map<string, SubscriberRecord>();
  private readonly onBatchLoaded?: TradingViewDatafeedOptions["onBatchLoaded"];
  private readonly onRealtimeBar?: TradingViewDatafeedOptions["onRealtimeBar"];

  constructor(options: TradingViewDatafeedOptions = {}) {
    this.baseUrl = (options.baseUrl || API_BASE_URL).replace(/\/$/, "");
    this.onBatchLoaded = options.onBatchLoaded;
    this.onRealtimeBar = options.onRealtimeBar;
  }

  onReady(callback: (config: Record<string, unknown>) => void): void {
    window.setTimeout(
      () =>
        callback({
          supports_search: true,
          supports_group_request: false,
          supports_marks: false,
          supports_timescale_marks: false,
          supports_time: true,
          supported_resolutions: DEFAULT_RESOLUTIONS,
          exchanges: [{ value: "", name: "TradingCoach", desc: "Trading Coach" }],
          symbols_types: [{ name: "All", value: "all" }],
        }),
      0,
    );
  }

  async resolveSymbol(symbolName: string, onResolve: ResolveCallback, onError: ErrorCallback): Promise<void> {
    try {
      const qs = buildQuery({ symbol: symbolName });
      const response = await fetch(`${this.baseUrl}/tv-api/symbols?${qs}`, { cache: "no-store" });
      if (!response.ok) throw new Error(`Symbol lookup failed (${response.status})`);
      const payload = (await response.json()) as TVSymbolInfo;
      if (!payload) {
        throw new Error("Empty resolve payload");
      }
      onResolve(payload);
    } catch (error) {
      console.error("[datafeed] resolveSymbol error", error);
      onError(error instanceof Error ? error.message : "resolve failed");
    }
  }

  async getBars(
    symbolInfo: TVSymbolInfo,
    resolution: string,
    periodParams: { from?: number; to?: number; countBack?: number },
    onResult: HistoryCallback,
    onError: ErrorCallback,
  ): Promise<void> {
    try {
      const now = Math.floor(Date.now() / 1000);
      const rangeSeconds = resolutionToSeconds(resolution) * (periodParams.countBack || 600);
      const from = periodParams.from ?? now - rangeSeconds;
      const to = periodParams.to ?? now;
      const qs = buildQuery({
        symbol: symbolInfo.ticker || symbolInfo.name,
        resolution,
        from,
        to,
      });
      const response = await fetch(`${this.baseUrl}/tv-api/bars?${qs}`, { cache: "no-store" });
      if (!response.ok) throw new Error(`Bars request failed (${response.status})`);
      const payload = (await response.json()) as UdfBarsResponse;
      if (!payload || !payload.s) {
        throw new Error("Malformed bars payload");
      }
      if (payload.s === "error") {
        throw new Error(payload.errmsg || "Bars request returned error");
      }
      const nextTimeToken =
        payload.nextTime !== undefined && payload.nextTime !== null
          ? coerceNumber(payload.nextTime, Number.NaN)
          : Number.NaN;
      const normalizedNextTime = Number.isFinite(nextTimeToken) ? Math.trunc(nextTimeToken) : undefined;
      if (payload.s === "no_data") {
        onResult([], { noData: true, nextTime: normalizedNextTime });
        return;
      }
      const bars = parseUdfBars(payload);
      this.onBatchLoaded?.(symbolInfo.ticker || symbolInfo.name, resolution, bars);
      onResult(bars, { noData: bars.length === 0, nextTime: normalizedNextTime });
    } catch (error) {
      console.error("[datafeed] getBars error", error);
      onError(error instanceof Error ? error.message : "bars failed");
    }
  }

  subscribeBars(
    symbolInfo: TVSymbolInfo,
    resolution: string,
    onRealtimeCallback: SubscribeCallback,
    subscriberUID: string,
  ): void {
    const intervalMs = Math.min(Math.max(resolutionToSeconds(resolution) * 1000, 5000), 60_000);

    const poll = async (): Promise<void> => {
      try {
        const now = Math.floor(Date.now() / 1000);
        const from = now - resolutionToSeconds(resolution) * 10;
        const qs = buildQuery({
          symbol: symbolInfo.ticker || symbolInfo.name,
          resolution,
          from,
          to: now,
        });
        const response = await fetch(`${this.baseUrl}/tv-api/bars?${qs}`, { cache: "no-store" });
        if (!response.ok) return;
        const payload = (await response.json()) as UdfBarsResponse;
        if (!payload || !payload.s || payload.s !== "ok") {
          return;
        }
        const bars = parseUdfBars(payload);
        if (!bars.length) return;
        const lastBar = bars[bars.length - 1];
        this.onRealtimeBar?.(symbolInfo.ticker || symbolInfo.name, resolution, lastBar);
        onRealtimeCallback(lastBar);
      } catch (error) {
        console.warn("[datafeed] subscribeBars poll error", error);
      }
    };

    poll();
    const timer = window.setInterval(poll, intervalMs);
    this.subscribers.set(subscriberUID, {
      timer,
      callback: onRealtimeCallback,
      symbol: symbolInfo.ticker || symbolInfo.name,
      resolution,
    });
  }

  unsubscribeBars(subscriberUID: string): void {
    const record = this.subscribers.get(subscriberUID);
    if (!record) return;
    window.clearInterval(record.timer);
    this.subscribers.delete(subscriberUID);
  }
}

export type TradingViewDatafeedInstance = InstanceType<typeof TradingViewDatafeed>;
