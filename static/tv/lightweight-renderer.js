(function () {
  if (!window.LightweightCharts) {
    return;
  }

  const params = new URLSearchParams(window.location.search);

  const normalizeResolution = (value) => {
    const token = (value || "").toString().trim().toLowerCase();
    if (!token) return "1";
    if (token.endsWith("m")) return token.replace("m", "");
    if (token.endsWith("h")) return String((parseInt(token, 10) || 1) * 60);
    if (token === "d" || token === "1d") return "1D";
    return token.toUpperCase();
  };

  const resolution = normalizeResolution(params.get("interval") || "1");
  const symbol = params.get("symbol") || "AAPL";
  const emaInputs = (params.get("ema") || "").split(",").map((s) => parseInt(s.trim(), 10)).filter((n) => Number.isFinite(n) && n > 0);
  const showVWAP = params.get("vwap") !== "0";
  const plan = {
    entry: parseFloat(params.get("entry")),
    stop: parseFloat(params.get("stop")),
    tps: (params.get("tp") || "").split(",").map((v) => parseFloat(v.trim())).filter((v) => Number.isFinite(v)),
  };
  const keyLevels = (params.get("levels") || "").split(",").map((v) => parseFloat(v.trim())).filter((v) => Number.isFinite(v));

  const resolutionToSeconds = (res) => {
    const token = (res || "").trim().toUpperCase();
    if (token.endsWith("D")) return (parseInt(token, 10) || 1) * 24 * 60 * 60;
    const minutes = parseInt(token, 10);
    if (!Number.isFinite(minutes) || minutes <= 0) return 60;
    return minutes * 60;
  };

  const computeEMA = (bars, length) => {
    if (!bars.length) return [];
    let ema = bars[0].close;
    const multiplier = 2 / (length + 1);
    return bars.map((bar, idx) => {
      if (idx === 0) {
        ema = bar.close;
      } else {
        ema = (bar.close - ema) * multiplier + ema;
      }
      return { time: bar.time, value: ema };
    });
  };

  const computeVWAP = (bars) => {
    let cumPV = 0;
    let cumVolume = 0;
    return bars.map((bar) => {
      const typical = (bar.high + bar.low + bar.close) / 3;
      cumPV += typical * bar.volume;
      cumVolume += bar.volume;
      const vwap = cumVolume > 0 ? cumPV / cumVolume : typical;
      return { time: bar.time, value: vwap };
    });
  };

  const drawPlanLevels = (candleSeries) => {
    const addPriceLine = (value, title, color, lineStyle) => {
      const level = parseFloat(value);
      if (!Number.isFinite(level)) return;
      candleSeries.createPriceLine({
        price: level,
        color,
        lineWidth: 2,
        lineStyle: lineStyle ?? LightweightCharts.LineStyle.Solid,
        axisLabelVisible: true,
        title,
      });
    };

    if (plan.entry) addPriceLine(plan.entry, "Entry", "#facc15");
    if (plan.stop) addPriceLine(plan.stop, "Stop", "#ef4444");
    plan.tps.forEach((value, idx) => addPriceLine(value, `TP${idx + 1}`, "#22c55e"));
    keyLevels.forEach((value, idx) => addPriceLine(value, `Level ${idx + 1}`, "#facc15", LightweightCharts.LineStyle.Dotted));
  };

  window.addEventListener("lightweight-data", (event) => {
    const { theme, bars } = event.detail;
    if (!Array.isArray(bars) || !bars.length) {
      document.getElementById("tv_chart_container").innerHTML = "<p style=\"padding:24px;text-align:center;\">No market data available.</p>";
      return;
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

    candleSeries.setData(
      bars.map((bar) => ({
        time: bar.time,
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
        title: `EMA ${length}`,
      });
      emaSeries.setData(computeEMA(bars, length));
    });

    if (showVWAP) {
      const vwapSeries = chart.addLineSeries({ color: "#facc15", lineWidth: 2, title: "VWAP" });
      vwapSeries.setData(computeVWAP(bars));
    }

    drawPlanLevels(candleSeries);

    chart.timeScale().fitContent();
  });
})();
