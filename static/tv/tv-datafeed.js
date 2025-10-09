(function () {
  const CONFIG = {
    supported_resolutions: ["1", "3", "5", "15", "30", "60", "120", "240", "1D"],
    exchanges: [{ value: "", name: "TradingCoach" }],
    symbols_types: [{ name: "All", value: "all" }],
  };

  const toQueryString = (params) =>
    Object.entries(params)
      .filter(([, value]) => value !== undefined && value !== null)
      .map(([key, value]) => `${encodeURIComponent(key)}=${encodeURIComponent(value)}`)
      .join("&");

  const resolutionToSeconds = (resolution) => {
    const token = String(resolution || "").trim().toUpperCase();
    if (token.endsWith("D")) {
      const days = parseInt(token, 10) || 1;
      return days * 24 * 60 * 60;
    }
    const minutes = parseInt(token, 10);
    if (!Number.isFinite(minutes) || minutes <= 0) {
      return 60;
    }
    return minutes * 60;
  };

  class TradingCoachDataFeed {
    constructor(baseUrl) {
      this.baseUrl = baseUrl;
      this.subscribers = new Map();
    }

    onReady(callback) {
      window.setTimeout(() => callback(CONFIG), 0);
    }

    async resolveSymbol(symbolName, onResolve, onError) {
      try {
        const qs = toQueryString({ symbol: symbolName });
        const response = await fetch(`${this.baseUrl}/tv-api/symbols?${qs}`);
        if (!response.ok) throw new Error(`Symbol lookup failed (${response.status})`);
        const payload = await response.json();
        if (!payload || payload.error) throw new Error(payload?.error || "Unknown symbol");
        onResolve(payload);
      } catch (err) {
        console.error("[tv-datafeed] resolveSymbol error", err);
        if (onError) onError("Symbol not found");
      }
    }

    async getBars(symbolInfo, resolution, periodParams, onResult, onError) {
      try {
        const now = Math.floor(Date.now() / 1000);
        const rangeSeconds = resolutionToSeconds(resolution) * (periodParams.countBack || 600);
        const from = periodParams.from || now - rangeSeconds;
        const to = periodParams.to || now;
        const qs = toQueryString({
          symbol: symbolInfo.ticker || symbolInfo.name,
          resolution,
          from,
          to,
        });
        const response = await fetch(`${this.baseUrl}/tv-api/bars?${qs}`);
        if (!response.ok) throw new Error(`Bars request failed (${response.status})`);
        const payload = await response.json();
        if (payload.s !== "ok") {
          onResult([], { noData: true });
          return;
        }
        const bars = payload.t.map((time, idx) => ({
          time: time * 1000,
          open: payload.o[idx],
          high: payload.h[idx],
          low: payload.l[idx],
          close: payload.c[idx],
          volume: payload.v[idx],
        }));
        onResult(bars, { noData: bars.length === 0 });
      } catch (err) {
        console.error("[tv-datafeed] getBars error", err);
        if (onError) onError(err);
      }
    }

    subscribeBars(symbolInfo, resolution, onRealtimeCallback, subscriberUID) {
      const intervalMs = Math.min(Math.max(resolutionToSeconds(resolution) * 1000, 5000), 60000);
      const poll = async () => {
        try {
          const now = Math.floor(Date.now() / 1000);
          const from = now - resolutionToSeconds(resolution) * 10;
          const qs = toQueryString({
            symbol: symbolInfo.ticker || symbolInfo.name,
            resolution,
            from,
            to: now,
          });
          const response = await fetch(`${this.baseUrl}/tv-api/bars?${qs}`);
          if (!response.ok) return;
          const payload = await response.json();
          if (payload.s !== "ok" || !payload.t.length) return;
          const lastIndex = payload.t.length - 1;
          onRealtimeCallback({
            time: payload.t[lastIndex] * 1000,
            open: payload.o[lastIndex],
            high: payload.h[lastIndex],
            low: payload.l[lastIndex],
            close: payload.c[lastIndex],
            volume: payload.v[lastIndex],
          });
        } catch (err) {
          console.warn("[tv-datafeed] subscribe poll error", err);
        }
      };
      poll();
      const handle = window.setInterval(poll, intervalMs);
      this.subscribers.set(subscriberUID, handle);
    }

    unsubscribeBars(subscriberUID) {
      const handle = this.subscribers.get(subscriberUID);
      if (handle) {
        window.clearInterval(handle);
        this.subscribers.delete(subscriberUID);
      }
    }
  }

  window.TradingCoachDataFeed = TradingCoachDataFeed;
})();
