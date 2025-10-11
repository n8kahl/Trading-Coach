(function () {
  const params = new URLSearchParams(window.location.search);

  const normalizeResolution = (value) => {
    const token = (value || '').toString().trim().toLowerCase();
    if (!token) return '1';
    if (token.endsWith('m')) return String(parseInt(token.replace('m', ''), 10) || 1);
    if (token.endsWith('h')) return String((parseInt(token.replace('h', ''), 10) || 1) * 60);
    if (token === 'd' || token === '1d') return '1D';
    return token.toUpperCase();
  };

  const resolutionToSeconds = (resolution) => {
    const token = (resolution || '').trim().toUpperCase();
    if (token.endsWith('D')) {
      const days = parseInt(token, 10) || 1;
      return days * 24 * 60 * 60;
    }
    const minutes = parseInt(token, 10);
    if (!Number.isFinite(minutes) || minutes <= 0) return 60;
    return minutes * 60;
  };

  const parseList = (value) =>
    (value || '')
      .split(',')
      .map((chunk) => chunk.trim())
      .filter(Boolean);

  const parseFloatList = (value) =>
    parseList(value)
      .map((item) => parseFloat(item))
      .filter((num) => Number.isFinite(num));

  const parseNamedLevels = (value) => {
    if (!value) return [];
    const chunks = value.includes(';') ? value.split(';') : value.split(',');
    return chunks
      .map((raw) => {
        const token = raw.trim();
        if (!token) return null;
        const [pricePart, labelPart] = token.split('|').map((part) => part.trim());
        const price = parseFloat(pricePart);
        if (!Number.isFinite(price)) return null;
        return { price, label: labelPart || null };
      })
      .filter(Boolean);
  };

  const parseNumber = (value) => {
    if (value === null || value === undefined || value === '') return null;
    const num = Number.parseFloat(value);
    return Number.isFinite(num) ? num : null;
  };

  const symbol = (params.get('symbol') || 'AAPL').toUpperCase();
  const resolution = normalizeResolution(params.get('interval') || '15');
  const theme = params.get('theme') === 'light' ? 'light' : 'dark';
  const baseUrl = `${window.location.protocol}//${window.location.host}`;

  const plan = {
    entry: parseNumber(params.get('entry')),
    stop: parseNumber(params.get('stop')),
    tps: parseFloatList(params.get('tp')),
    direction: (params.get('direction') || '').toLowerCase(),
    strategy: params.get('strategy'),
    atr: parseNumber(params.get('atr')),
    notes: params.get('notes'),
    title: params.get('title'),
  };
  const emaTokens = parseList(params.get('ema'));
  let keyLevels = parseNamedLevels(params.get('levels'));
  let overlayValues = [];
  const scalePlanToken = (params.get('scale_plan') || 'auto').toLowerCase();
  let emaSeries = [];

  const container = document.getElementById('tv_chart_container');
  const legendEl = document.getElementById('plan_legend');
  const debugEl = document.getElementById('debug_banner');

  const dbg = (msg) => {
    debugEl.style.display = 'block';
    debugEl.textContent += (debugEl.textContent ? '\n' : '') + msg;
  };

  if (!window.LightweightCharts) {
    dbg('LightweightCharts not available');
    return;
  }

  const chart = LightweightCharts.createChart(container, {
    layout: {
      background: { type: 'solid', color: theme === 'light' ? '#ffffff' : '#0b0f14' },
      textColor: theme === 'light' ? '#1b2733' : '#e6edf3',
    },
    grid: {
      vertLines: { color: theme === 'light' ? '#e5e9f0' : '#1f2933' },
      horzLines: { color: theme === 'light' ? '#e5e9f0' : '#1f2933' },
    },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    timeScale: { borderColor: theme === 'light' ? '#d1d5db' : '#1f2933' },
    rightPriceScale: { borderColor: theme === 'light' ? '#d1d5db' : '#1f2933', scaleMargins: { top: 0.1, bottom: 0.25 } },
    leftPriceScale: { visible: true, borderColor: theme === 'light' ? '#d1d5db' : '#1f2933', scaleMargins: { top: 0.8, bottom: 0.02 } },
  });

  const candleSeries = chart.addCandlestickSeries({
    upColor: '#22c55e',
    downColor: '#ef4444',
    borderUpColor: '#22c55e',
    borderDownColor: '#ef4444',
    wickUpColor: '#22c55e',
    wickDownColor: '#ef4444',
  });

  const volumeSeries = chart.addHistogramSeries({
    priceScaleId: 'left',
    color: theme === 'light' ? '#94a3b8' : '#334155',
    priceFormat: { type: 'volume' },
    scaleMargins: { top: 0.7, bottom: 0 },
  });

  const emaPalette = ['#38bdf8', '#a855f7', '#facc15', '#f97316'];
  emaSeries = emaTokens.reduce((acc, token, idx) => {
    const span = parseInt(token, 10);
    if (!Number.isFinite(span) || span <= 0) return acc;
    const series = chart.addLineSeries({
      lineWidth: 2,
      color: emaPalette[idx % emaPalette.length],
      title: `EMA${span}`,
    });
    acc.push({ span, series });
    return acc;
  }, []);

  const priceLines = [];
  const clearPriceLines = () => {
    while (priceLines.length) {
      const line = priceLines.pop();
      candleSeries.removePriceLine(line);
    }
  };

  const addPriceLine = (price, title, color, style = LightweightCharts.LineStyle.Solid, width = 2) => {
    if (!Number.isFinite(price)) return;
    const line = candleSeries.createPriceLine({
      price,
      color,
      title,
      lineWidth: width,
      lineStyle: style,
    });
    priceLines.push(line);
  };

  const renderLegend = (lastPrice) => {
    const rows = [];
    const pushRow = (label, value) => {
      if (value === null || value === undefined) return;
      rows.push(`<dt>${label}</dt><dd>${value}</dd>`);
    };
    pushRow('Symbol', symbol);
    pushRow('Resolution', resolution);
    if (Number.isFinite(lastPrice)) pushRow('Last', lastPrice.toFixed(2));
    if (Number.isFinite(plan.entry)) pushRow('Entry', plan.entry.toFixed(2));
    if (Number.isFinite(plan.stop)) pushRow('Stop', plan.stop.toFixed(2));
    plan.tps.forEach((tp, idx) => {
      if (Number.isFinite(tp)) pushRow(`TP${idx + 1}`, tp.toFixed(2));
    });
    if (Number.isFinite(plan.atr)) pushRow('ATR', plan.atr.toFixed(4));
    if (plan.strategy) pushRow('Setup', plan.strategy);
    legendEl.innerHTML = `
      <h2>${plan.title || `${symbol} · ${resolution}`}</h2>
      <dl>${rows.join('')}</dl>
      ${plan.notes ? `<p class="notes">${plan.notes}</p>` : ''}
    `;
    legendEl.classList.add('visible');
  };

  const fetchBars = async () => {
    try {
      const now = Math.floor(Date.now() / 1000);
      const span = Math.min(resolutionToSeconds(resolution) * 600, 60 * 60 * 24 * 365);
      const from = now - span;
      const qs = new URLSearchParams({
        symbol,
        resolution,
        from: String(from),
        to: String(now),
      }).toString();
      const response = await fetch(`${baseUrl}/tv-api/bars?${qs}`);
      if (!response.ok) throw new Error(`bars request failed (${response.status})`);
      const payload = await response.json();
      if (payload.s !== 'ok' || !payload.t?.length) throw new Error('No data');

      const bars = payload.t.map((time, idx) => ({
        time: payload.t[idx],
        open: payload.o[idx],
        high: payload.h[idx],
        low: payload.l[idx],
        close: payload.c[idx],
        volume: payload.v[idx] || 0,
      }));

      const candleData = bars.map((bar) => ({
        time: bar.time,
        open: bar.open,
        high: bar.high,
        low: bar.low,
        close: bar.close,
      }));
      candleSeries.setData(candleData);

      const volumeData = bars.map((bar) => ({
        time: bar.time,
        value: bar.volume,
        color: bar.close >= bar.open ? '#22c55e88' : '#ef444488',
      }));
      volumeSeries.setData(volumeData);

      const lastPrice = bars[bars.length - 1]?.close ?? null;
      let scaleMultiplier = 1;
      if (scalePlanToken !== 'off') {
        if (scalePlanToken === 'auto') {
          if (Number.isFinite(plan.entry) && Number.isFinite(lastPrice) && plan.entry && lastPrice) {
            const ratio = lastPrice / plan.entry;
            if (ratio > 1.2 || ratio < 0.8) {
              scaleMultiplier = ratio;
            }
          }
        } else {
          const manual = parseFloat(scalePlanToken);
          if (Number.isFinite(manual) && manual > 0) {
            scaleMultiplier = manual;
          }
        }
      }

      if (scaleMultiplier !== 1 && Number.isFinite(scaleMultiplier) && scaleMultiplier > 0) {
        if (Number.isFinite(plan.entry)) plan.entry *= scaleMultiplier;
        if (Number.isFinite(plan.stop)) plan.stop *= scaleMultiplier;
        plan.tps = plan.tps.map((tp) => (Number.isFinite(tp) ? tp * scaleMultiplier : tp));
        keyLevels = keyLevels.map((lvl) => (Number.isFinite(lvl.price) ? { ...lvl, price: lvl.price * scaleMultiplier } : lvl));
      }

      overlayValues = [
        plan.entry,
        plan.stop,
        ...plan.tps,
        ...keyLevels.map((lvl) => lvl.price),
      ].filter((value) => Number.isFinite(value));

      const vwapRequested = (params.get('vwap') || '').toLowerCase() !== 'false';
      let vwapSeries = null;
      if (vwapRequested) {
        vwapSeries = chart.addLineSeries({
          lineWidth: 2,
          color: '#ffffff',
          title: 'VWAP',
        });
      }
      emaSeries.forEach(({ span, series }) => {
        const values = [];
        const weight = 2 / (span + 1);
        let emaValue = null;
        for (let i = 0; i < candleData.length; i += 1) {
          const close = candleData[i].close;
          if (close == null) continue;
          if (emaValue === null) {
            emaValue = close;
          } else {
            emaValue = close * weight + emaValue * (1 - weight);
          }
          values.push({ time: candleData[i].time, value: emaValue });
        }
        series.setData(values);
      });
      if (vwapSeries) {
        const values = [];
        let cumulativePV = 0;
        let cumulativeVol = 0;
        for (let i = 0; i < candleData.length; i += 1) {
          const bar = candleData[i];
          const vol = volumeData[i]?.value || 0;
          const typical = (bar.high + bar.low + bar.close) / 3;
          cumulativePV += typical * vol;
          cumulativeVol += vol;
          const vwapValue = cumulativeVol > 0 ? cumulativePV / cumulativeVol : typical;
          values.push({ time: bar.time, value: vwapValue });
        }
        vwapSeries.setData(values);
      }

      clearPriceLines();
      addPriceLine(plan.entry, 'Entry', '#facc15', LightweightCharts.LineStyle.Solid, 3);
      addPriceLine(plan.stop, 'Stop', '#ef4444', LightweightCharts.LineStyle.Solid, 3);
      plan.tps.forEach((tp, idx) => addPriceLine(tp, `TP${idx + 1}`, '#7CFC00', LightweightCharts.LineStyle.Solid, 3));
      [...keyLevels]
        .filter((level) => Number.isFinite(level.price))
        .sort((a, b) => b.price - a.price)
        .forEach((level, idx) => {
          const label = level.label ? level.label : `Level ${idx + 1}`;
          addPriceLine(level.price, label, '#94a3b8', LightweightCharts.LineStyle.Dotted);
        });

      renderLegend(lastPrice);
      const priceScale = chart.priceScale('right');
      if (overlayValues.length) {
        const dataLows = bars.map((bar) => bar.low).filter((val) => Number.isFinite(val));
        const dataHighs = bars.map((bar) => bar.high).filter((val) => Number.isFinite(val));
        const dataMin = dataLows.length ? Math.min(...dataLows) : null;
        const dataMax = dataHighs.length ? Math.max(...dataHighs) : null;
        const overlayMin = Math.min(...overlayValues);
        const overlayMax = Math.max(...overlayValues);
        const combinedMin = [overlayMin, dataMin].filter((val) => val !== null && val !== undefined).reduce((acc, val) => Math.min(acc, val));
        const combinedMax = [overlayMax, dataMax].filter((val) => val !== null && val !== undefined).reduce((acc, val) => Math.max(acc, val));
        if (Number.isFinite(combinedMin) && Number.isFinite(combinedMax) && combinedMax > combinedMin) {
          const padding = Math.max((combinedMax - combinedMin) * 0.1, 0.01);
          priceScale.setAutoScale(false);
          priceScale.setPriceRange({ minValue: combinedMin - padding, maxValue: combinedMax + padding });
        } else {
          priceScale.setAutoScale(true);
        }
      } else {
        priceScale.setAutoScale(true);
      }
      debugEl.style.display = 'none';
      debugEl.textContent = '';
    } catch (err) {
      dbg(`Error loading data: ${err.message}`);
    }
  };

  fetchBars();
  window.addEventListener('resize', () => {
    chart.resize(container.clientWidth, container.clientHeight);
  });
  window.setInterval(fetchBars, 60000);
})();
