(function () {
  const params = new URLSearchParams(window.location.search);

  const normalizeResolution = (value) => {
    const token = (value || "").toString().trim().toLowerCase();
    if (!token) return "1";
    if (token.endsWith("m")) {
      return String(parseInt(token.replace("m", ""), 10) || 1);
    }
    if (token.endsWith("h")) {
      const hours = parseInt(token.replace("h", ""), 10) || 1;
      return String(hours * 60);
    }
    if (token === "d" || token === "1d") {
      return "1D";
    }
    return token.toUpperCase();
  };

  const resolutionToSeconds = (resolution) => {
    const token = (resolution || "").trim().toUpperCase();
    if (token.endsWith("D")) {
      const days = parseInt(token, 10) || 1;
      return days * 24 * 60 * 60;
    }
    const minutes = parseInt(token, 10);
    if (!Number.isFinite(minutes) || minutes <= 0) return 60;
    return minutes * 60;
  };

  const parseListParam = (value) =>
    (value || "")
      .split(",")
      .map((chunk) => chunk.trim())
      .filter(Boolean);

  const parseFloatList = (value) =>
    parseListParam(value)
      .map((item) => parseFloat(item))
      .filter((num) => Number.isFinite(num));

  const symbol = params.get("symbol") || "AAPL";
  const resolution = normalizeResolution(params.get("interval") || "1");
  const theme = params.get("theme") === "light" ? "light" : "dark";
  const emaInputs = parseListParam(params.get("ema")).map((n) => parseInt(n, 10)).filter((n) => Number.isFinite(n) && n > 0);
  const showVWAP = params.get("vwap") !== "0";
  const extraStudies = parseListParam(params.get("studies"));
  const range = params.get("range");

  const plan = {
    entry: params.get("entry"),
    stop: params.get("stop"),
    tps: parseFloatList(params.get("tp")),
    direction: params.get("direction"),
    strategy: params.get("strategy"),
    atr: params.get("atr"),
    notes: params.get("notes"),
  };

  const legendEl = document.getElementById("plan_legend");

  const updateLegend = () => {
    const rows = [];
    if (plan.entry) rows.push(["Entry", plan.entry]);
    if (plan.stop) rows.push(["Stop", plan.stop]);
    plan.tps.forEach((value, idx) => rows.push([`TP${idx + 1}`, value.toFixed(2)]));
    if (plan.atr) rows.push(["ATR", plan.atr]);

    const header =
      [plan.strategy, plan.direction ? plan.direction.toUpperCase() : ""]
        .filter(Boolean)
        .join(" · ") || "Trade Plan";

    if (!rows.length && !plan.notes && header === "Trade Plan") {
      legendEl.classList.remove("visible");
      legendEl.innerHTML = "";
      return;
    }

    const detailRows = rows.map(([label, value]) => `<dt>${label}</dt><dd>${value}</dd>`).join("");
    const notes = plan.notes ? `<div class="notes">${plan.notes}</div>` : "";
    legendEl.innerHTML = `<h2>${header}</h2><dl>${detailRows}</dl>${notes}`;
    legendEl.classList.add("visible");
  };

  const baseUrl = `${window.location.protocol}//${window.location.host}`;
  const datafeed = window.TradingCoachDataFeed ? new window.TradingCoachDataFeed(baseUrl) : null;

  const setVisibleRange = (chart) => {
    if (!range) return;
    const now = Math.floor(Date.now() / 1000);
    const span = resolutionToSeconds(normalizeResolution(range)) || resolutionToSeconds(resolution) * 200;
    chart.setVisibleRange({ from: now - span, to: now });
  };

  const initTradingViewWidget = () => {
    if (!window.TradingView || typeof window.TradingView.widget !== "function" || !datafeed) {
      return false;
    }

    const widget = new window.TradingView.widget({
      autosize: true,
      symbol,
      interval: resolution,
      theme,
      container_id: "tv_chart_container",
      library_path: "charting_library/",
      locale: "en",
      timezone: "America/New_York",
      datafeed,
      toolbar_bg: theme === "light" ? "#ffffff" : "#0b0f14",
      disabled_features: ["use_localstorage_for_settings", "timeframes_toolbar", "legend_widget"],
      enabled_features: ["study_templates"],
      overrides: {
        "paneProperties.background": theme === "light" ? "#ffffff" : "#0b0f14",
        "paneProperties.vertGridProperties.color": theme === "light" ? "#e8edf2" : "#1f2933",
        "paneProperties.horzGridProperties.color": theme === "light" ? "#e8edf2" : "#1f2933",
        "scalesProperties.textColor": theme === "light" ? "#1b2733" : "#e6edf3",
      },
    });

    const drawLevel = (chart, price, label, color) => {
      const level = parseFloat(price);
      if (!Number.isFinite(level)) return;
      chart.createShape(
        { price: level },
        {
          shape: "horizontal_line",
          text: label,
          lock: true,
          disableUndo: true,
          overrides: {
            color,
            linewidth: 2,
          },
        }
      );
    };

    widget.onChartReady(() => {
      const chart = widget.activeChart();
      setVisibleRange(chart);

      emaInputs.forEach((length) => {
        chart.createStudy("Moving Average Exponential", false, false, [length]);
      });
      if (showVWAP) {
        chart.createStudy("VWAP", false, false);
      }
      extraStudies.forEach((study) => {
        if (study) chart.createStudy(study, false, false);
      });

      if (plan.entry) drawLevel(chart, plan.entry, "Entry", "#facc15");
      if (plan.stop) drawLevel(chart, plan.stop, "Stop", "#ef4444");
      plan.tps.forEach((value, idx) => drawLevel(chart, value, `TP${idx + 1}`, "#22c55e"));

      updateLegend();
    });

    return true;
  };

  const computeEMA = (bars, length) => {
    if (!bars.length) return [];
    const multiplier = 2 / (length + 1);
    let ema = bars[0].close;
    return bars.map((bar, index) => {
      const close = bar.close;
      if (index === 0) {
        ema = close;
      } else {
        ema = (close - ema) * multiplier + ema;
      }
      return { time: Math.floor(bar.time / 1000), value: ema };
    });
  };

  const computeVWAP = (bars) => {
    let cumulativePV = 0;
    let cumulativeVolume = 0;
    return bars.map((bar) => {
      const typical = (bar.high + bar.low + bar.close) / 3;
      cumulativePV += typical * bar.volume;
      cumulativeVolume += bar.volume || 0;
      const vwap = cumulativeVolume > 0 ? cumulativePV / cumulativeVolume : typical;
      return { time: Math.floor(bar.time / 1000), value: vwap };
    });
  };

  const initLightweightCharts = async () => {
    if (!window.LightweightCharts) {
      return false;
    }

    const container = document.getElementById("tv_chart_container");
    container.innerHTML = "";

    const chart = LightweightCharts.createChart(container, {
      layout: {
        background: { type: "solid", color: theme === "light" ? "#ffffff" : "#0b0f14" },
        textColor: theme === "light" ? "#1b2733" : "#e6edf3",
      },
      grid: {
        vertLines: { color: theme === "light" ? "#e8edf2" : "#1f2933" },
        horzLines: { color: theme === "light" ? "#e8edf2" : "#1f2933" },
      },
      crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
      timeScale: { borderColor: theme === "light" ? "#d1d5db" : "#1f2933" },
      rightPriceScale: { borderColor: theme === "light" ? "#d1d5db" : "#1f2933" },
    });

    const candleSeries = chart.addCandlestickSeries({
      upColor: "#22c55e",
      downColor: "#ef4444",
      borderUpColor: "#22c55e",
      borderDownColor: "#ef4444",
      wickUpColor: "#22c55e",
      wickDownColor: "#ef4444",
    });

    try {
      const now = Math.floor(Date.now() / 1000);
      const span = resolutionToSeconds(normalizeResolution(range)) || resolutionToSeconds(resolution) * 500;
      const from = now - span;
      const qs = new URLSearchParams({ symbol, resolution, from: String(from), to: String(now) }).toString();
      const response = await fetch(`${baseUrl}/tv-api/bars?${qs}`);
      const payload = await response.json();
      if (payload.s !== "ok") throw new Error("No data");

      const bars = payload.t.map((time, idx) => ({
        time: time * 1000,
        open: payload.o[idx],
        high: payload.h[idx],
        low: payload.l[idx],
        close: payload.c[idx],
        volume: payload.v[idx] || 0,
      }));

      candleSeries.setData(
        bars.map((bar) => ({
          time: Math.floor(bar.time / 1000),
          open: bar.open,
          high: bar.high,
          low: bar.low,
          close: bar.close,
        }))
      );

      emaInputs.slice(0, 4).forEach((length, index) => {
        const emaSeries = chart.addLineSeries({
          color: ["#93c5fd", "#f97316", "#a855f7", "#14b8a6"][index % 4],
          lineWidth: 2,
        });
        emaSeries.setData(computeEMA(bars, length));
      });

      if (showVWAP) {
        const vwapSeries = chart.addLineSeries({ color: "#facc15", lineWidth: 2 });
        vwapSeries.setData(computeVWAP(bars));
      }

      const addPriceLine = (price, title, color) => {
        const level = parseFloat(price);
        if (!Number.isFinite(level)) return;
        candleSeries.createPriceLine({
          price: level,
          color,
          lineWidth: 2,
          axisLabelVisible: true,
          title,
        });
      };

      if (plan.entry) addPriceLine(plan.entry, "Entry", "#facc15");
      if (plan.stop) addPriceLine(plan.stop, "Stop", "#ef4444");
      plan.tps.forEach((value, idx) => addPriceLine(value, `TP${idx + 1}`, "#22c55e"));

      chart.timeScale().fitContent();
      updateLegend();
    } catch (err) {
      console.error("[tv] Lightweight fallback failed", err);
      container.innerHTML = "<p style=\"padding:24px;text-align:center;\">No market data available.</p>";
    }

    return true;
  };

  const bootstrap = async () => {
    if (initTradingViewWidget()) return;
    if (await initLightweightCharts()) return;
    console.error("No chart library available");
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bootstrap);
  } else {
    bootstrap();
  }
})();
