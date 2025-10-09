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

  const parseNumber = (value) => {
    if (value === null || value === undefined || value === "") return null;
    const num = Number.parseFloat(value);
    return Number.isFinite(num) ? num : null;
  };

  const symbol = params.get("symbol") || "AAPL";
  const resolution = normalizeResolution(params.get("interval") || "1");
  const theme = params.get("theme") === "light" ? "light" : "dark";
  const emaInputs = parseListParam(params.get("ema")).map((n) => parseInt(n, 10)).filter((n) => Number.isFinite(n) && n > 0);
  const showVWAP = params.get("vwap") !== "0";
  const extraStudies = parseListParam(params.get("studies"));
  const range = params.get("range");
  const keyLevelsRaw = parseFloatList(params.get("levels"));
  let keyLevels = keyLevelsRaw.slice();
  const debug = params.get("debug") === "1";
  const scalePlanParam = (params.get("scale_plan") || "").trim();

  const plan = {
    entry: parseNumber(params.get("entry")),
    stop: parseNumber(params.get("stop")),
    tps: parseFloatList(params.get("tp")),
    direction: params.get("direction"),
    strategy: params.get("strategy"),
    atr: parseNumber(params.get("atr")),
    notes: params.get("notes"),
  };
  const planOriginal = {
    entry: Number.isFinite(plan.entry) ? plan.entry : null,
    stop: Number.isFinite(plan.stop) ? plan.stop : null,
    tps: plan.tps.slice(),
    atr: Number.isFinite(plan.atr) ? plan.atr : null,
  };
  let planScale = {
    mode: null,
    factor: 1,
    applied: false,
    reference: null,
    lastClose: null,
  };

  const legendEl = document.getElementById("plan_legend");
  const debugEl = document.getElementById("debug_banner");
  const dbg = (msg) => {
    if (!debug) return;
    debugEl.style.display = "block";
    debugEl.textContent += (debugEl.textContent ? "\n" : "") + msg;
  };

  const formatPlanValue = (raw, scaled, decimals = 2) => {
    if (!Number.isFinite(scaled)) return null;
    if (planScale.applied && Number.isFinite(raw)) {
      const tolerance = Math.max(Math.pow(10, -decimals) * 0.5, Math.abs(raw) * 0.0005);
      if (Math.abs(raw - scaled) > tolerance) {
        return `${raw.toFixed(decimals)} → ${scaled.toFixed(decimals)}`;
      }
    }
    return scaled.toFixed(decimals);
  };

  const updateLegend = () => {
    const rows = [];
    const entryText = formatPlanValue(planOriginal.entry, plan.entry);
    if (entryText) rows.push(["Entry", entryText]);
    const stopText = formatPlanValue(planOriginal.stop, plan.stop);
    if (stopText) rows.push(["Stop", stopText]);
    plan.tps.forEach((value, idx) => {
      const raw = planOriginal.tps[idx];
      const formatted = formatPlanValue(raw, value);
      if (formatted) rows.push([`TP${idx + 1}`, formatted]);
    });
    const atrText = formatPlanValue(planOriginal.atr, plan.atr, 4);
    if (atrText) rows.push(["ATR", atrText]);
    if (planScale.applied) {
      let scaleLabel = `${planScale.factor.toFixed(4)}×`;
      if (planScale.mode === "auto" && Number.isFinite(planScale.reference) && Number.isFinite(planScale.lastClose)) {
        scaleLabel += ` (${planScale.reference.toFixed(2)}→${planScale.lastClose.toFixed(2)})`;
      } else if (planScale.mode === "auto") {
        scaleLabel += " (auto)";
      }
      rows.push(["Scale", scaleLabel]);
    }
    const indicatorTokens = [];
    if (emaInputs.length) indicatorTokens.push(`EMA(${emaInputs.join("/")})`);
    if (showVWAP) indicatorTokens.push("VWAP");
    indicatorTokens.push(`Resolution ${resolution}`);
    rows.push(["Indicators", indicatorTokens.join(" · ")]);

    const header =
      [symbol.toUpperCase(), resolution.toUpperCase()]
        .filter(Boolean)
        .join(" · ");
    const subHeader =
      [plan.strategy, plan.direction ? plan.direction.toUpperCase() : ""]
        .filter(Boolean)
        .join(" · ");

    if (!rows.length && !plan.notes && !subHeader) {
      legendEl.classList.remove("visible");
      legendEl.innerHTML = "";
      return;
    }

    const detailRows = rows
      .filter(([, value]) => value)
      .map(([label, value]) => `<dt>${label}</dt><dd>${value}</dd>`)
      .join("");
    const notes = plan.notes ? `<div class="notes">${plan.notes}</div>` : "";
    const sub = subHeader ? `<p style="margin:4px 0 12px;opacity:0.75;">${subHeader}</p>` : "";
    legendEl.innerHTML = `<h2>${header}</h2>${sub}<dl>${detailRows}</dl>${notes}`;
    legendEl.classList.add("visible");
  };

  const baseUrl = `${window.location.protocol}//${window.location.host}`;
  const datafeed = window.TradingCoachDataFeed ? new window.TradingCoachDataFeed(baseUrl) : null;
  const resetPlanToOriginal = () => {
    plan.entry = Number.isFinite(planOriginal.entry) ? planOriginal.entry : null;
    plan.stop = Number.isFinite(planOriginal.stop) ? planOriginal.stop : null;
    plan.tps = planOriginal.tps.slice();
    plan.atr = Number.isFinite(planOriginal.atr) ? planOriginal.atr : null;
    keyLevels = keyLevelsRaw.slice();
  };

  const applyPlanScale = (scaleInfo) => {
    resetPlanToOriginal();
    if (!scaleInfo || !scaleInfo.applied || !Number.isFinite(scaleInfo.factor) || scaleInfo.factor <= 0) {
      planScale = {
        mode: null,
        factor: 1,
        applied: false,
        reference: null,
        lastClose: null,
      };
      return;
    }

    planScale = {
      mode: scaleInfo.mode || null,
      factor: scaleInfo.factor,
      applied: true,
      reference: Number.isFinite(scaleInfo.reference) ? scaleInfo.reference : null,
      lastClose: Number.isFinite(scaleInfo.lastClose) ? scaleInfo.lastClose : null,
    };

    const scaleValue = (value) => {
      if (!Number.isFinite(value)) return null;
      const scaled = value * planScale.factor;
      return Number.isFinite(scaled) ? scaled : null;
    };

    plan.entry = scaleValue(planOriginal.entry);
    plan.stop = scaleValue(planOriginal.stop);
    plan.tps = planOriginal.tps.map((value) => scaleValue(value));
    plan.atr = scaleValue(planOriginal.atr);
    keyLevels = keyLevelsRaw.map((value) => scaleValue(value)).filter((value) => Number.isFinite(value));
  };

  const evaluateScalePlan = (mode, lastClose, { autoWhenEmpty = false } = {}) => {
    let normalized = (mode || "").trim().toLowerCase();
    if (!normalized && autoWhenEmpty) {
      normalized = "auto";
    }
    if (!normalized) {
      return { applied: false, factor: 1 };
    }
    if (["off", "none", "false", "0"].includes(normalized)) {
      return { applied: false, factor: 1, mode: "off" };
    }
    if (normalized === "auto") {
      if (!Number.isFinite(lastClose)) {
        return { applied: false, factor: 1 };
      }
      const candidates = [
        planOriginal.entry,
        planOriginal.stop,
        ...planOriginal.tps,
        ...keyLevelsRaw,
      ].filter((value) => Number.isFinite(value) && value > 0);
      const reference = candidates.length ? candidates[0] : null;
      if (!Number.isFinite(reference) || reference <= 0) {
        return { applied: false, factor: 1 };
      }
      const factor = lastClose / reference;
      if (!Number.isFinite(factor) || factor <= 0) {
        return { applied: false, factor: 1 };
      }
      const drift = Math.abs(1 - factor);
      if (drift < 0.02 || factor < 0.05 || factor > 20) {
        return { applied: false, factor: 1 };
      }
      return {
        applied: true,
        factor,
        mode: "auto",
        reference,
        lastClose,
      };
    }
    const explicit = Number.parseFloat(normalized);
    if (Number.isFinite(explicit) && explicit > 0 && Math.abs(1 - explicit) > 0.0001) {
      return {
        applied: true,
        factor: explicit,
        mode: "manual",
      };
    }
    return { applied: false, factor: 1 };
  };

  // Apply explicit numeric scaling immediately so legend + labels match.
  const initialScaleInfo = evaluateScalePlan(scalePlanParam, null, { autoWhenEmpty: false });
  if (initialScaleInfo.applied && initialScaleInfo.mode === "manual") {
    applyPlanScale(initialScaleInfo);
  } else {
    applyPlanScale({ applied: false, factor: 1 });
  }

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

      if (Number.isFinite(plan.entry)) drawLevel(chart, plan.entry, "Entry", "#facc15");
      if (Number.isFinite(plan.stop)) drawLevel(chart, plan.stop, "Stop", "#ef4444");
      plan.tps.forEach((value, idx) => {
        if (Number.isFinite(value)) drawLevel(chart, value, `TP${idx + 1}`, "#22c55e");
      });

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
      rightPriceScale: { borderColor: theme === "light" ? "#d1d5db" : "#1f2933", scaleMargins: { top: 0.08, bottom: 0.25 } },
      leftPriceScale: { visible: true, borderColor: theme === "light" ? "#d1d5db" : "#1f2933", scaleMargins: { top: 0.8, bottom: 0.02 } },
    });

    const candleSeries = chart.addCandlestickSeries({
      upColor: "#22c55e",
      downColor: "#ef4444",
      borderUpColor: "#22c55e",
      borderDownColor: "#ef4444",
      wickUpColor: "#22c55e",
      wickDownColor: "#ef4444",
    });

    // Volume histogram overlay on left scale
    const volumeSeries = chart.addHistogramSeries({
      priceScaleId: 'left',
      priceFormat: { type: 'volume' },
      color: theme === 'light' ? '#94a3b8' : '#334155',
      base: 0,
    });

    try {
      const now = Math.floor(Date.now() / 1000);
      const resNorm = normalizeResolution(range) || resolution;
      const span = resolutionToSeconds(resNorm) || resolutionToSeconds(resolution) * 500;
      const from = now - span;
      const qs = new URLSearchParams({ symbol, resolution, from: String(from), to: String(now) }).toString();
      dbg(`request: /tv-api/bars?${qs}`);
      const response = await fetch(`${baseUrl}/tv-api/bars?${qs}`);
      const payload = await response.json();
      dbg(`response: status=${payload.s} bars=${(payload.t||[]).length}`);
      if (payload.s !== "ok") throw new Error("No data");

      const bars = payload.t.map((time, idx) => ({
        time: time * 1000,
        open: payload.o[idx],
        high: payload.h[idx],
        low: payload.l[idx],
        close: payload.c[idx],
        volume: payload.v[idx] || 0,
      }));

      const candleData =
        bars.map((bar) => ({
          time: Math.floor(bar.time / 1000),
          open: bar.open,
          high: bar.high,
          low: bar.low,
          close: bar.close,
        }));
      candleSeries.setData(candleData);

      // Map volume with up/down colors
      const volData = bars.map((bar) => ({
        time: Math.floor(bar.time / 1000),
        value: bar.volume || 0,
        color: bar.close >= bar.open ? '#22c55e88' : '#ef444488',
      }));
      volumeSeries.setData(volData);

      const lastClose = bars.length ? bars[bars.length - 1].close : null;
      const allowAutoDefault = scalePlanParam === "";
      const scaleInfo = evaluateScalePlan(scalePlanParam, lastClose, { autoWhenEmpty: allowAutoDefault });
      if (scaleInfo.applied) {
        const label = scaleInfo.mode === "auto" && allowAutoDefault ? "auto*default" : scaleInfo.mode;
        dbg(`scale_plan applied mode=${label} factor=${scaleInfo.factor.toFixed(4)}`);
      } else if (scalePlanParam) {
        dbg(`scale_plan skipped mode=${scalePlanParam}`);
      }
      applyPlanScale(scaleInfo);

      emaInputs.slice(0, 4).forEach((length, index) => {
        const emaSeries = chart.addLineSeries({
          color: ["#93c5fd", "#f97316", "#a855f7", "#14b8a6"][index % 4],
          lineWidth: 2,
          lastValueVisible: true,
          priceLineVisible: false,
          crosshairMarkerVisible: false,
        });
        emaSeries.setData(computeEMA(bars, length));
      });

      if (showVWAP) {
        const vwapSeries = chart.addLineSeries({ color: "#facc15", lineWidth: 2, lastValueVisible: true, priceLineVisible: false, crosshairMarkerVisible: false });
        vwapSeries.setData(computeVWAP(bars));
      }

      const addPriceLine = (price, title, color, lineStyle) => {
        const level = parseFloat(price);
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

      if (Number.isFinite(plan.entry)) addPriceLine(plan.entry, "Entry", "#facc15");
      if (Number.isFinite(plan.stop)) addPriceLine(plan.stop, "Stop", "#ef4444");
      plan.tps.forEach((value, idx) => {
        if (Number.isFinite(value)) addPriceLine(value, `TP${idx + 1}`, "#22c55e");
      });
      keyLevels.forEach((value) => addPriceLine(value, "Key", "#facc15", LightweightCharts.LineStyle.Dotted));

      const levelValues = [
        ...(Number.isFinite(plan.entry) ? [plan.entry] : []),
        ...(Number.isFinite(plan.stop) ? [plan.stop] : []),
        ...plan.tps.filter((value) => Number.isFinite(value)),
        ...keyLevels,
      ].filter((v) => Number.isFinite(v));

      if (levelValues.length) {
        const dataMin = Math.min(...bars.map((b) => b.low));
        const dataMax = Math.max(...bars.map((b) => b.high));
        const minValue = Math.min(dataMin, ...levelValues);
        const maxValue = Math.max(dataMax, ...levelValues);
        const hasAutoScale = typeof candleSeries.setAutoscaleInfoProvider === "function";
        if (hasAutoScale) {
          try {
            candleSeries.setAutoscaleInfoProvider(() => ({ priceRange: { minValue, maxValue } }));
          } catch (e) {
            dbg(`autoscale provider not available: ${e?.message || e}`);
          }
        }
        if (!hasAutoScale) {
          // Fallback: add a transparent helper series with min/max so autoscale includes levels
          const helper = chart.addLineSeries({
            color: theme === "light" ? "#00000000" : "#00000000",
            lineWidth: 1,
            lastValueVisible: false,
            priceLineVisible: false,
            crosshairMarkerVisible: false,
          });
          const firstTs = Math.floor(bars[0].time / 1000);
          const lastTs = Math.floor(bars[bars.length - 1].time / 1000);
          helper.setData([
            { time: firstTs, value: minValue },
            { time: lastTs, value: maxValue },
          ]);
        }
      }

      chart.timeScale().fitContent();
      updateLegend();
    } catch (err) {
      console.error("[tv] Lightweight fallback failed", err);
      dbg(`error: ${err?.message || err}`);
      // If we have bars but something failed after fetch, still try to render baseline chart
      try {
        if (typeof bars !== "undefined" && Array.isArray(bars) && bars.length) {
          const failsafe = document.getElementById("tv_chart_container");
          failsafe.innerHTML = "";
          const chart = LightweightCharts.createChart(failsafe, { layout: { background: { type: "solid", color: theme === "light" ? "#ffffff" : "#0b0f14" }, textColor: theme === "light" ? "#1b2733" : "#e6edf3" } });
          const cs = chart.addCandlestickSeries();
          cs.setData(bars.map((b) => ({ time: Math.floor(b.time / 1000), open: b.open, high: b.high, low: b.low, close: b.close })));
          chart.timeScale().fitContent();
          return;
        }
      } catch {}
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
