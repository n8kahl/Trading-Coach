(function () {
  const params = new URLSearchParams(window.location.search);
  const symbol = params.get("symbol") || "AAPL";
  const interval = params.get("interval") || "1";
  const theme = params.get("theme") === "light" ? "light" : "dark";
  const studies = (params.get("studies") || "").split(",").map((s) => s.trim()).filter(Boolean);
  const emaInputs = (params.get("ema") || "").split(",").map((s) => parseInt(s.trim(), 10)).filter((n) => Number.isFinite(n) && n > 0);
  const showVWAP = params.get("vwap") !== "0";
  const range = params.get("range") || null;
  const plan = {
    entry: params.get("entry"),
    stop: params.get("stop"),
    tps: (params.get("tp") || "").split(",").map((v) => parseFloat(v.trim())).filter((v) => Number.isFinite(v)),
    direction: params.get("direction"),
    strategy: params.get("strategy"),
    atr: params.get("atr"),
    notes: params.get("notes"),
  };

  const resolutionToSeconds = (resolution) => {
    const token = String(resolution || "").trim().toUpperCase();
    if (token.endsWith("D")) {
      const days = parseInt(token, 10) || 1;
      return days * 24 * 60 * 60;
    }
    const minutes = parseInt(token, 10);
    if (!Number.isFinite(minutes) || minutes <= 0) return 60;
    return minutes * 60;
  };

  const baseUrl = `${window.location.protocol}//${window.location.host}`;
  const datafeed = new window.TradingCoachDataFeed(baseUrl);

  const widget = new TradingView.widget({
    autosize: true,
    symbol,
    interval,
    theme,
    container_id: "tv_chart_container",
    library_path: "charting_library/",
    locale: "en",
    timezone: "America/New_York",
    datafeed,
    toolbar_bg: theme === "light" ? "#fff" : "#0b0f14",
    disabled_features: ["use_localstorage_for_settings", "timeframes_toolbar", "legend_widget"],
    enabled_features: ["study_templates"],
    studies_overrides: {},
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

  const updateLegend = () => {
    const legend = document.getElementById("plan_legend");
    const rows = [];
    if (plan.entry) rows.push(["Entry", plan.entry]);
    if (plan.stop) rows.push(["Stop", plan.stop]);
    plan.tps.forEach((value, idx) => rows.push([`TP${idx + 1}`, value.toFixed(2)]));
    if (plan.atr) rows.push(["ATR", plan.atr]);
    if (!rows.length && !plan.notes && !plan.strategy && !plan.direction) {
      legend.classList.remove("visible");
      legend.innerHTML = "";
      return;
    }
    const header =
      [plan.strategy, plan.direction ? plan.direction.toUpperCase() : ""]
        .filter(Boolean)
        .join(" · ") || "Trade Plan";
    const details = rows.map(([key, value]) => `<dt>${key}</dt><dd>${value}</dd>`).join("");
    const notesBlock = plan.notes ? `<div class="notes">${plan.notes}</div>` : "";
    legend.innerHTML = `<h2>${header}</h2><dl>${details}</dl>${notesBlock}`;
    legend.classList.add("visible");
  };

  widget.onChartReady(() => {
    const chart = widget.activeChart();

    if (range) {
      chart.setVisibleRange({
        from: Math.floor(Date.now() / 1000) - resolutionToSeconds(range),
        to: Math.floor(Date.now() / 1000),
      });
    }

    emaInputs.forEach((length) => {
      chart.createStudy("Moving Average Exponential", false, false, [length]);
    });
    if (showVWAP) {
      chart.createStudy("VWAP", false, false);
    }
    studies.forEach((study) => {
      if (!study) return;
      chart.createStudy(study, false, false);
    });

    if (plan.entry) drawLevel(chart, plan.entry, "Entry", "#facc15");
    if (plan.stop) drawLevel(chart, plan.stop, "Stop", "#ef4444");
    plan.tps.forEach((value, idx) => drawLevel(chart, value, `TP${idx + 1}`, "#22c55e"));

    updateLegend();
  });
})();
