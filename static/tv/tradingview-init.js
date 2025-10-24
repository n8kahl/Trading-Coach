(async function () {
  const params = new URLSearchParams(window.location.search);

  const normalizeResolution = (value) => {
    const token = (value || '').toString().trim().toLowerCase();
    if (!token) return '1';
    if (token.endsWith('m')) return String(parseInt(token.replace('m', ''), 10) || 1);
    if (token.endsWith('h')) return String((parseInt(token.replace('h', ''), 10) || 1) * 60);
    if (token.endsWith('w')) return '1W';
    if (token === 'd' || token === '1d') return '1D';
    return token.toUpperCase();
  };

  const resolutionToSeconds = (resolution) => {
    const token = (resolution || '').trim().toUpperCase();
    if (token.endsWith('D')) {
      const days = parseInt(token, 10) || 1;
      return days * 24 * 60 * 60;
    }
    if (token.endsWith('W')) {
      const weeks = parseInt(token, 10) || 1;
      return weeks * 7 * 24 * 60 * 60;
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

  let planMeta = {};
  try {
    const rawPlanMeta = params.get('plan_meta');
    planMeta = rawPlanMeta ? JSON.parse(rawPlanMeta) : {};
  } catch {
    planMeta = {};
  }

  const planMetaSymbol =
    typeof planMeta.symbol === 'string' && planMeta.symbol.trim()
      ? planMeta.symbol.trim().toUpperCase()
      : typeof planMeta.plan === 'object' && typeof planMeta.plan.symbol === 'string'
        ? planMeta.plan.symbol.trim().toUpperCase()
        : null;

  const planIdSymbol = (() => {
    const planId = params.get('plan_id');
    if (!planId) return null;
    const token = planId.split('-')[0];
    return token ? token.trim().toUpperCase() : null;
  })();

  const rawSymbolParam = params.get('symbol');
  const resolvedSymbol = (rawSymbolParam || planMetaSymbol || planIdSymbol || 'AAPL').toUpperCase();
  if (!rawSymbolParam) {
    params.set('symbol', resolvedSymbol);
    const newUrl = `${window.location.pathname}?${params.toString()}`;
    window.history.replaceState({}, '', newUrl);
  }

  const symbol = resolvedSymbol;
  const marketStatusParam = (params.get('market_status') || '').trim().toLowerCase();
  const sessionStatusParam = (params.get('session_status') || '').trim().toLowerCase();
  const sessionPhaseParam = (params.get('session_phase') || '').trim().toLowerCase();
  const sessionBannerParamRaw = params.get('session_banner');
  const sessionBannerParam = typeof sessionBannerParamRaw === 'string' ? sessionBannerParamRaw.trim() : '';
  const liveFlag = params.get('live') === '1';
  let currentResolution = normalizeResolution(params.get('interval') || '15');
  const theme = params.get('theme') === 'light' ? 'light' : 'dark';
  const baseUrl = `${window.location.protocol}//${window.location.host}`;
  const initialPlanId = (params.get('plan_id') || '').trim() || null;
  const initialPlanVersion = (params.get('plan_version') || '').trim() || null;
  let currentPlanId = initialPlanId;
  let currentPlanVersion = initialPlanVersion;
  let planLayers = null;
  let planLayersMeta = {};
  let planZones = [];
  const MAX_PRIMARY_LEVELS = 6;
  const SUPPORTING_LEVEL_LIMIT = 40;
  let levelGroups = { primary: [], supplemental: [] };
  const supportingLevelsParam = params.get('supportingLevels');
  let showAllLevels = supportingLevelsParam !== '0';
  let allKeyLevels = [];
  let levelsToggleEl = null;
  let pendingAutoReplan = null;
  let autoReplanInFlight = false;
  let keyLevels = [];
  const AUTO_REARM_STORAGE_KEY = 'tc_auto_rearm';
  let autoRearmEnabled = (() => {
    try {
      return window.localStorage.getItem(AUTO_REARM_STORAGE_KEY) === '1';
    } catch {
      return false;
    }
  })();
  if (currentPlanId) {
    try {
      const layersResponse = await fetch(`${baseUrl}/api/v1/gpt/chart-layers?plan_id=${encodeURIComponent(currentPlanId)}`, {
        cache: 'no-store',
      });
      if (layersResponse.ok) {
        planLayers = await layersResponse.json();
        if (planLayers && Array.isArray(planLayers.zones)) {
          planZones = planLayers.zones;
        }
        if (planLayers && typeof planLayers === 'object' && planLayers.meta && typeof planLayers.meta === 'object') {
          planLayersMeta = planLayers.meta;
        }
      }
    } catch (error) {
      console.warn('chart-layers fetch failed', error);
    }
  }

  const toNumber = (value) => {
    const num = Number(value);
    return Number.isFinite(num) ? num : null;
  };

  let basePlan = {
    entry: parseNumber(params.get('entry')),
    stop: parseNumber(params.get('stop')),
    tps: parseFloatList(params.get('tp')),
    direction: (params.get('direction') || '').toLowerCase(),
    strategy: params.get('strategy'),
    atr: parseNumber(params.get('atr')),
    notes: params.get('notes'),
    title: params.get('title'),
    tpMeta: (() => {
      try {
        return JSON.parse(params.get('tp_meta') || '[]');
      } catch {
        return [];
      }
    })(),
    runner: (() => {
      try {
        return JSON.parse(params.get('runner') || 'null');
      } catch {
        return null;
      }
    })(),
  };

  const clonePlan = () => JSON.parse(JSON.stringify(basePlan));
  let currentPlan = clonePlan();

  const collectInitialLevels = () => {
    const collected = [];
    const fromParams = parseNamedLevels(params.get('levels'));
    if (Array.isArray(fromParams) && fromParams.length) {
      fromParams.forEach((item) => {
        if (!item || !Number.isFinite(item.price)) return;
        collected.push({ price: Number(item.price), label: item.label || null });
      });
    }
    if (planLayers && Array.isArray(planLayers.levels)) {
      planLayers.levels.forEach((item) => {
        if (!item || !Number.isFinite(item.price)) return;
        collected.push({ price: Number(item.price), label: item.label || null });
      });
    }
    if (planZones.length) {
      planZones.forEach((zone, index) => {
        if (!zone || (!Number.isFinite(zone.high) && !Number.isFinite(zone.low))) {
          return;
        }
        const baseLabel = zone.label || zone.kind || `Zone ${index + 1}`;
        if (Number.isFinite(zone.high)) {
          collected.push({ price: Number(zone.high), label: `${baseLabel} High` });
        }
        if (Number.isFinite(zone.low)) {
          collected.push({ price: Number(zone.low), label: `${baseLabel} Low` });
        }
      });
    }
    const keyLevelSource = mergedPlanMeta.key_levels;
    if (Array.isArray(keyLevelSource)) {
      keyLevelSource.forEach((item) => {
        if (!item || !Number.isFinite(item.price)) return;
        const label = typeof item.label === 'string' && item.label.trim() ? item.label.trim() : null;
        collected.push({ price: Number(item.price), label });
      });
    } else if (keyLevelSource && typeof keyLevelSource === 'object') {
      Object.entries(keyLevelSource).forEach(([name, value]) => {
        const numeric = toNumber(value);
        if (!Number.isFinite(numeric)) return;
        const label = name
          .replace(/_/g, ' ')
          .replace(/\b([a-z])/g, (match) => match.toUpperCase());
        collected.push({ price: numeric, label });
      });
    }
    return dedupeLevels(collected);
  };

  const paramConfidence = parseNumber(params.get('confidence'));
  const mergedPlanMeta = {
    symbol,
    style: planMeta.style || params.get('style'),
    style_display: planMeta.style_display || null,
    bias: planMeta.bias || basePlan.direction || null,
    confidence: toNumber(planMeta.confidence) ?? paramConfidence,
    risk_reward: toNumber(planMeta.risk_reward ?? planMeta.rr_to_t1),
    notes: planMeta.notes || params.get('notes') || null,
    warnings: Array.isArray(planMeta.warnings) ? planMeta.warnings : [],
    runner: planMeta.runner || basePlan.runner,
    targets: Array.isArray(planMeta.targets) && planMeta.targets.length ? planMeta.targets : basePlan.tps,
    target_meta: Array.isArray(planMeta.target_meta) && planMeta.target_meta.length ? planMeta.target_meta : basePlan.tpMeta,
    entry: toNumber(planMeta.entry) ?? basePlan.entry,
    stop: toNumber(planMeta.stop) ?? basePlan.stop,
    atr: toNumber(planMeta.atr) ?? basePlan.atr,
    strategy: planMeta.strategy_label || basePlan.strategy,
    expected_move: toNumber(planMeta.expected_move),
    horizon_minutes: toNumber(planMeta.horizon_minutes),
    key_levels: planMeta.key_levels || null,
    confluence: Array.isArray(planMeta.confluence_tags) ? planMeta.confluence_tags : [],
    confidence_factors: Array.isArray(planMeta.confidence_factors) ? planMeta.confidence_factors : [],
  };

  const supportingLevelsFromParams = dedupeLevels(parseNamedLevels(params.get('levels')));
  allKeyLevels = collectInitialLevels();
  setLevelGroupsFromMeta(planLayersMeta, allKeyLevels, { silent: true });
  if (supportingLevelsFromParams.length) {
    levelGroups.supplemental = dedupeLevels([
      ...supportingLevelsFromParams,
      ...levelGroups.supplemental,
    ]);
  }
  updateLevelVisibility({ silent: true });
  setOrDeleteParam('supportingLevels', showAllLevels ? '1' : '0');

  const dataSourceRaw = (params.get('data_source') || '').trim();
  const dataModeToken = (params.get('data_mode') || '').trim().toLowerCase();
  const dataAgeMsParam = Number(params.get('data_age_ms') || '');
  const lastUpdateRaw = (params.get('last_update') || '').trim();
  const STALE_FEED_THRESHOLD_MS = 120000;
  const STREAM_GOOD_THRESHOLD_MS = 15000;
  const STREAM_WARN_THRESHOLD_MS = 60000;
  let dataLastUpdateDate = parseIsoDate(lastUpdateRaw);
  let dataAgeMs = Number.isFinite(dataAgeMsParam) ? dataAgeMsParam : null;
  let lastHeartbeatMs = dataLastUpdateDate ? dataLastUpdateDate.getTime() : null;
  let followLive = true;
  let suppressFollowDetection = false;
  let forceAutoFocus = true;
  const isFeedDegraded = () =>
    dataModeToken === 'degraded' || (Number.isFinite(dataAgeMs) && dataAgeMs > STALE_FEED_THRESHOLD_MS);
  let lastDegradedState = isFeedDegraded();
  let streamingStatusState = dataLastUpdateDate ? (dataAgeMs !== null && dataAgeMs > STREAM_WARN_THRESHOLD_MS ? 'stale' : dataAgeMs > STREAM_GOOD_THRESHOLD_MS ? 'warn' : 'good') : 'idle';

  const focusToken = (params.get('focus') || '').toLowerCase();
  const centerTimeParamRaw = params.get('center_time');
  const centerTimeToken = typeof centerTimeParamRaw === 'string' ? centerTimeParamRaw.trim() : '';
  const centerTimeIsLatest = centerTimeToken && centerTimeToken.toLowerCase() === 'latest';
  const centerTimeNumeric = !centerTimeIsLatest && centerTimeToken ? Number(centerTimeToken) : null;
  let hasAppliedFocusRange = false;
  let hasAppliedTimeCenter = false;
  let hasAppliedTimeWindow = false;
const headerSymbolEl = document.getElementById('header_symbol');
  const headerStrategyEl = document.getElementById('header_strategy');
  const headerBiasEl = document.getElementById('header_bias');
  const headerConfidenceEl = document.getElementById('header_confidence');
  const headerRREl = document.getElementById('header_rr');
  const headerDurationEl = document.getElementById('header_duration');
  const headerLastPriceEl = document.getElementById('header_lastprice');
  const headerPlanStatusEl = document.getElementById('header_planstatus');
  const headerMarketEl = document.getElementById('header_market');
  const headerLastUpdateEl = document.getElementById('header_lastupdate');
  const headerDataSourceEl = document.getElementById('header_datasource');
  const streamingStatusEl = document.getElementById('streaming_status');
const followLiveToggleEl = document.getElementById('follow_live_toggle');
levelsToggleEl = document.getElementById('levels_toggle');
  const planStatusNoteEl = document.getElementById('plan_status_note');
  const timeframeSwitcherEl = document.getElementById('timeframe_switcher');
  const planPanelEl = document.getElementById('plan_panel');
  const planPanelBodyEl = document.getElementById('plan_panel_body');
  const debugEl = document.getElementById('debug_banner');
  const toastContainerEl = document.getElementById('toast_container');

  const headerSymbolMetaEl = (() => {
    if (!headerSymbolEl) return null;
    const parent = headerSymbolEl.parentElement;
    if (!parent) return null;
    const node = document.createElement('div');
    node.id = 'header_symbol_meta';
    node.className = 'plan-header__meta';
    parent.appendChild(node);
    return node;
  })();

  const planLogEntries = [];
  const PLAN_LOG_LIMIT = 50;
  const PLAN_LOG_VISIBLE_LIMIT = 5;
  let planLogListEl = null;
  let planLogEmptyEl = null;

  if (headerLastUpdateEl) {
    headerLastUpdateEl.style.whiteSpace = 'nowrap';
    headerLastUpdateEl.style.display = 'inline-block';
  }

  const setSymbolMeta = (text, alert = false) => {
    if (!headerSymbolMetaEl) return;
    headerSymbolMetaEl.textContent = text || '';
    if (alert && text) {
      headerSymbolMetaEl.classList.add('plan-header__meta--alert');
    } else {
      headerSymbolMetaEl.classList.remove('plan-header__meta--alert');
    }
  };
  setSymbolMeta('');

  const MONTH_NAMES = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  const etSessionPartsFormatter = new Intl.DateTimeFormat('en-US', {
    timeZone: 'America/New_York',
    hour12: false,
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  });

  const getEtParts = (timestampSeconds) => {
    const parts = etSessionPartsFormatter.formatToParts(new Date(timestampSeconds * 1000));
    const map = {};
    for (const part of parts) {
      if (part.type !== 'literal') {
        map[part.type] = part.value;
      }
    }
    const monthNumber = Number(map.month);
    return {
      year: Number(map.year),
      month: monthNumber,
      monthName: MONTH_NAMES[monthNumber - 1] || map.month,
      day: map.day,
      hour: Number(map.hour),
      minute: Number(map.minute),
    };
  };

  const classifySession = (minutes) => {
    if (minutes >= 4 * 60 && minutes < 9 * 60 + 30) return 'premarket';
    if (minutes >= 9 * 60 + 30 && minutes < 16 * 60) return 'regular';
    if (minutes >= 16 * 60 && minutes < 20 * 60) return 'afterhours';
    return 'overnight';
  };

  const SESSION_CLASSNAMES = {
    premarket: 'session-band session-premarket',
    regular: 'session-band session-regular',
    afterhours: 'session-band session-afterhours',
    overnight: 'session-band',
  };

  let sessionOverlayEl = null;
  let sessionSegments = [];
  let lastSecondsPerBar = Math.max(resolutionToSeconds(currentResolution), 60);

  const escapeHtml = (value) =>
    typeof value === 'string'
      ? value
          .replace(/&/g, '&amp;')
          .replace(/</g, '&lt;')
          .replace(/>/g, '&gt;')
          .replace(/"/g, '&quot;')
      : value;

  function parseIsoDate(value) {
    if (!value) return null;
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? null : date;
  }

  function dedupeLevels(levels) {
    if (!Array.isArray(levels)) return [];
    const seen = new Set();
    const result = [];
    levels.forEach((item) => {
      if (!item || !Number.isFinite(item.price)) return;
      const label = typeof item.label === 'string' && item.label.trim() ? item.label.trim() : null;
      const token = `${label || ''}:${Number(item.price).toFixed(4)}`;
      if (seen.has(token)) return;
      seen.add(token);
      result.push({ price: Number(item.price), label });
    });
    return result;
  }

  function sanitizeLevelList(levels) {
    if (!Array.isArray(levels)) return [];
    return dedupeLevels(
      levels
        .map((item) => {
          if (!item) return null;
          if (Number.isFinite(item.price)) {
            return { price: Number(item.price), label: item.label || null };
          }
          if (Number.isFinite(item.level)) {
            return { price: Number(item.level), label: item.label || null };
          }
          return null;
        })
        .filter(Boolean),
    );
  }

  function setLevelGroups(primary, supplemental, { silent = false } = {}) {
    levelGroups = {
      primary: dedupeLevels(primary),
      supplemental: dedupeLevels(supplemental),
    };
    updateLevelVisibility({ silent });
  }

  function setLevelGroupsFromMeta(meta, fallbackLevels, { silent = false } = {}) {
    const fallback = dedupeLevels(Array.isArray(fallbackLevels) ? fallbackLevels : []);
    const metaGroups = (meta && meta.level_groups) || {};
    let primary = sanitizeLevelList(metaGroups.primary);
    let supplemental = sanitizeLevelList(metaGroups.supplemental);
    if (!primary.length && fallback.length) {
      primary = fallback.slice(0, MAX_PRIMARY_LEVELS);
    }
    if (!supplemental.length && fallback.length) {
      const start = primary.length;
      supplemental = fallback.slice(start, fallback.length);
    }
    setLevelGroups(primary, supplemental, { silent });
  }

  function updateLevelVisibility({ silent = false } = {}) {
    const combined = showAllLevels
      ? [...levelGroups.primary, ...levelGroups.supplemental]
      : [...levelGroups.primary];
    keyLevels = dedupeLevels(combined);
    updateLevelsToggleUi();
    if (!silent) {
      refreshLevelLines();
    }
  }

  function updateLevelsToggleUi() {
    if (!levelsToggleEl) return;
    if (!levelGroups.supplemental.length) {
      levelsToggleEl.disabled = true;
      levelsToggleEl.classList.remove('levels-toggle--active');
      levelsToggleEl.setAttribute('aria-pressed', 'false');
      levelsToggleEl.textContent = 'Show Supporting Levels';
      levelsToggleEl.title = 'Supporting levels unavailable';
      return;
    }
    levelsToggleEl.disabled = false;
    levelsToggleEl.classList.toggle('levels-toggle--active', showAllLevels);
    levelsToggleEl.setAttribute('aria-pressed', showAllLevels ? 'true' : 'false');
    levelsToggleEl.textContent = showAllLevels ? 'Hide Supporting Levels' : 'Show Supporting Levels';
    levelsToggleEl.title = showAllLevels ? 'Hide supporting chart levels' : 'Display supporting chart levels';
  }

  const etDateFormatter = new Intl.DateTimeFormat('en-US', {
    timeZone: 'America/New_York',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
  const etTimeFormatter = new Intl.DateTimeFormat('en-US', {
    timeZone: 'America/New_York',
    hour: '2-digit',
    minute: '2-digit',
  });
  const etClockFormatter = new Intl.DateTimeFormat('en-US', {
    timeZone: 'America/New_York',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });

  const formatRelativeDuration = (ms) => {
    if (!Number.isFinite(ms) || ms < 0) return '';
    const seconds = Math.round(ms / 1000);
    if (seconds < 60) return `${seconds}s`;
    const minutes = Math.round(seconds / 60);
    if (minutes < 60) return `${minutes}m`;
    const hours = Math.round(minutes / 60);
    if (hours < 24) return `${hours}h`;
    const days = Math.round(hours / 24);
    return `${days}d`;
  };
  const friendlySourceLabel = (token) => {
    const normalized = (token || '').trim().toLowerCase();
    if (!normalized) return '';
    if (normalized === 'polygon') return 'Polygon';
    if (normalized === 'polygon_cached') return 'Polygon (cached)';
    if (normalized === 'polygon_stale') return 'Polygon (stale)';
    return (token || '').replace(/_/g, ' ').replace(/\b\w/g, (char) => char.toUpperCase());
  };

  const formatLogTime = (timestampSeconds) => {
    if (!Number.isFinite(timestampSeconds)) return '';
    const date = new Date(timestampSeconds * 1000);
    return `${etTimeFormatter.format(date)} ET`;
  };

  const updateSessionBands = (candles, secondsPerBar) => {
    if (!Array.isArray(candles) || !candles.length) {
      sessionSegments = [];
      renderSessionBands();
      return;
    }
    const segments = [];
    let current = null;
    candles.forEach((bar) => {
      if (!bar || !Number.isFinite(bar.time)) return;
      const info = getEtParts(bar.time);
      const dateKey = `${info.year}-${String(info.month).padStart(2, '0')}-${info.day}`;
      const minutes = info.hour * 60 + info.minute;
      const session = classifySession(minutes);
      if (!current || current.session !== session || current.dateKey !== dateKey) {
        if (current) {
          segments.push(current);
        }
        current = {
          session,
          start: bar.time,
          end: bar.time,
          dateKey,
        };
      }
      current.end = bar.time + secondsPerBar;
    });
    if (current) {
      segments.push(current);
    }
    sessionSegments = segments;
    renderSessionBands();
  };

  const renderSessionBands = () => {
    if (!sessionOverlayEl) return;
    const timeScale = chart ? chart.timeScale() : null;
    if (!timeScale) return;
    sessionOverlayEl.innerHTML = '';
    if (!sessionSegments.length) return;
    sessionSegments.forEach((segment) => {
      const left = timeScale.timeToCoordinate(segment.start);
      const right = timeScale.timeToCoordinate(segment.end);
      if (left == null || right == null || right <= left) {
        return;
      }
      const band = document.createElement('div');
      band.className = SESSION_CLASSNAMES[segment.session] || SESSION_CLASSNAMES.overnight;
      band.style.left = `${left}px`;
      band.style.width = `${right - left}px`;
      sessionOverlayEl.appendChild(band);
    });
  };

  const renderPlanLog = () => {
    if (!planLogListEl || !planLogEmptyEl) return;
    if (!planLogEntries.length) {
      planLogListEl.innerHTML = '';
      planLogEmptyEl.style.display = 'block';
      return;
    }
    planLogEmptyEl.style.display = 'none';
    const itemsHtml = planLogEntries
      .map((entry) => {
        const severityClass = entry.severity === 'alert' ? ' plan-log__entry--alert' : '';
        const timeLabel = formatLogTime(entry.ts);
        const safeMessage = escapeHtml(entry.message);
        return `<li class="plan-log__entry${severityClass}"><span class="plan-log__time">${timeLabel}</span><span class="plan-log__message">${safeMessage}</span></li>`;
      })
      .join('');
    planLogListEl.innerHTML = itemsHtml;
  };

  const appendPlanLogEntry = (message, timestampSeconds = Math.floor(Date.now() / 1000), severity = 'info') => {
    if (!message && message !== 0) return;
    const raw = String(message).trim();
    if (!raw) return;
    const ts = Number.isFinite(timestampSeconds) ? Number(timestampSeconds) : Math.floor(Date.now() / 1000);
    if (planLogEntries.length && planLogEntries[0].message === raw && planLogEntries[0].severity === severity) {
      // Avoid spamming identical consecutive lines.
      return;
    }
    const duplicateIndex = planLogEntries.findIndex((entry) => entry.message === raw);
    if (duplicateIndex !== -1) {
      planLogEntries.splice(duplicateIndex, 1);
    }
    planLogEntries.unshift({ message: raw, ts, severity });
    if (planLogEntries.length > PLAN_LOG_LIMIT) {
      planLogEntries.length = PLAN_LOG_LIMIT;
    }
    renderPlanLog();
  };

  const showToast = (message, { tone = 'success', duration = 3200 } = {}) => {
    if (!toastContainerEl || !message) return;
    const toast = document.createElement('div');
    toast.className = `toast ${tone === 'warn' ? 'toast--warn' : 'toast--success'}`;
    toast.textContent = message;
    toastContainerEl.appendChild(toast);
    window.setTimeout(() => {
      toast.classList.add('toast--hide');
      window.setTimeout(() => {
        if (toast.parentElement === toastContainerEl) {
          toastContainerEl.removeChild(toast);
        }
      }, 320);
    }, Math.max(1200, duration));
  };

  const updateFollowLiveToggle = () => {
    if (!followLiveToggleEl) return;
    followLiveToggleEl.classList.toggle('follow-toggle--active', followLive);
    followLiveToggleEl.setAttribute('aria-pressed', followLive ? 'true' : 'false');
    followLiveToggleEl.title = followLive ? 'Auto-scroll to latest bar' : 'Auto-scroll paused — click to resume';
  };

  const setFollowLiveState = (value, { scroll = true } = {}) => {
    const next = Boolean(value);
    if (followLive === next) return;
    followLive = next;
    updateFollowLiveToggle();
    if (followLive && scroll) {
      ensureScrollToRealTime();
    }
  };

  const pauseFollowLive = () => {
    if (!followLive) return;
    followLive = false;
    updateFollowLiveToggle();
  };

  const updateAutoRearmToggle = () => {
    const btn = document.getElementById('auto_rearm_toggle');
    if (!btn) return;
    if (autoRearmEnabled) {
      btn.classList.add('plan-replay__button--active');
      btn.setAttribute('aria-pressed', 'true');
      btn.textContent = 'Auto Rearm: On';
      btn.title = 'Automatically reload the next plan when it becomes available';
    } else {
      btn.classList.remove('plan-replay__button--active');
      btn.setAttribute('aria-pressed', 'false');
      btn.textContent = 'Auto Rearm: Off';
      btn.title = 'Leave auto reload off — you will see a prompt when a new plan is ready';
    }
  };

  const setAutoRearmEnabled = (value) => {
    autoRearmEnabled = Boolean(value);
    try {
      window.localStorage.setItem(AUTO_REARM_STORAGE_KEY, autoRearmEnabled ? '1' : '0');
    } catch {
      // ignore storage errors
    }
    updateAutoRearmToggle();
    if (autoRearmEnabled && pendingAutoReplan && !autoReplanInFlight) {
      adoptPendingReplan({ viaAuto: true });
    }
  };

  const renderAutoReplanBanner = () => {
    const section = document.getElementById('auto_replan_section');
    const messageEl = document.getElementById('auto_replan_message');
    const applyBtn = document.getElementById('auto_replan_apply');
    if (!section || !messageEl || !applyBtn) return;
    if (!pendingAutoReplan) {
      section.hidden = true;
      applyBtn.disabled = true;
      return;
    }
    section.hidden = false;
    const note = pendingAutoReplan.note ? ` — ${escapeHtml(String(pendingAutoReplan.note))}` : '';
    messageEl.innerHTML = `Next plan ${escapeHtml(String(pendingAutoReplan.planId))}${note}`;
    applyBtn.disabled = autoReplanInFlight;
  };

  const adoptPendingReplan = async ({ viaAuto = false } = {}) => {
    if (!pendingAutoReplan || autoReplanInFlight) return;
    autoReplanInFlight = true;
    renderAutoReplanBanner();
    try {
      if (!viaAuto) {
        pauseFollowLive();
      }
      const payload = await fetchPlanIdea(pendingAutoReplan.planId);
      await applyPlanSnapshot(payload, {
        logMessage: viaAuto ? 'Auto replan applied.' : 'Next plan loaded.',
        fallbackPlanId: pendingAutoReplan.planId,
        fallbackPlanVersion: pendingAutoReplan.planVersion,
      });
      pendingAutoReplan = null;
      renderAutoReplanBanner();
      showToast(viaAuto ? 'Auto replan applied.' : 'Plan reloaded.');
    } catch (error) {
      console.error('Auto replan adoption failed', error);
      showToast('Failed to load next plan', { tone: 'warn' });
    } finally {
      autoReplanInFlight = false;
      renderAutoReplanBanner();
    }
  };

  const streamingStateForAge = (ageMs) => {
    if (ageMs === null || ageMs === undefined) {
      return dataLastUpdateDate ? 'warn' : 'idle';
    }
    if (ageMs <= STREAM_GOOD_THRESHOLD_MS) return 'good';
    if (ageMs <= STREAM_WARN_THRESHOLD_MS) return 'warn';
    return 'stale';
  };

  const applyStreamingStatus = () => {
    if (!streamingStatusEl) return;
    const age = dataLastUpdateDate ? (Number.isFinite(dataAgeMs) ? dataAgeMs : Date.now() - dataLastUpdateDate.getTime()) : null;
    const nextState = dataLastUpdateDate ? streamingStateForAge(age) : 'idle';
    streamingStatusState = nextState;
    streamingStatusEl.classList.remove('streaming-pill--idle', 'streaming-pill--good', 'streaming-pill--warn', 'streaming-pill--stale');
    streamingStatusEl.classList.add(`streaming-pill--${nextState}`);
    let title = 'Awaiting first heartbeat';
    if (dataLastUpdateDate) {
      const rel = Number.isFinite(age) ? formatRelativeDuration(age) : '';
      const statusLabel =
        nextState === 'good'
          ? 'Heartbeat fresh'
          : nextState === 'warn'
            ? 'Heartbeat delayed'
            : nextState === 'stale'
              ? 'Heartbeat stale'
              : 'Awaiting stream';
      const parts = [statusLabel];
      if (rel) parts.push(`${rel} ago`);
      if (dataSourceRaw) parts.push(`Source ${friendlySourceLabel(dataSourceRaw)}`);
      title = parts.join(' · ');
    }
    streamingStatusEl.title = title;
    streamingStatusEl.setAttribute('aria-label', title);
  };

  const setDataLastUpdate = (timestampSeconds) => {
    if (!Number.isFinite(timestampSeconds)) return;
    dataLastUpdateDate = new Date(timestampSeconds * 1000);
    lastHeartbeatMs = dataLastUpdateDate.getTime();
    dataAgeMs = 0;
    const degraded = isFeedDegraded();
    if (degraded !== lastDegradedState) {
      lastDegradedState = degraded;
      applyMarketStatus(currentMarketPhase || 'closed');
    }
    streamingStatusState = 'good';
    updateDataMetadata();
  };

  const updateDataMetadata = () => {
    if (headerLastUpdateEl) {
      if (!dataLastUpdateDate) {
        headerLastUpdateEl.textContent = 'Awaiting first heartbeat';
        headerLastUpdateEl.classList.remove('plan-header__meta--alert');
      } else {
        const age = Number.isFinite(dataAgeMs) ? dataAgeMs : Date.now() - dataLastUpdateDate.getTime();
        const rel = Number.isFinite(age) ? formatRelativeDuration(age) : '';
        const formatted = etDateFormatter.format(dataLastUpdateDate);
        headerLastUpdateEl.textContent = `Last update ${formatted} ET${rel ? ` (${rel} ago)` : ''}`;
        headerLastUpdateEl.classList.toggle('plan-header__meta--alert', streamingStatusState === 'stale');
      }
    }
    if (headerDataSourceEl) {
      const sourceLabel = friendlySourceLabel(dataSourceRaw);
      const parts = [];
      const degraded = isFeedDegraded();
      if (sourceLabel) parts.push(`Data: ${sourceLabel}`);
      if (degraded) parts.push('Degraded');
      headerDataSourceEl.textContent = parts.join(' · ');
      headerDataSourceEl.classList.toggle('plan-header__meta--alert', degraded);
    }
    applyStreamingStatus();
  };

  const isAlertSeverity = (token) => {
    if (!token) return false;
    const normalized = token.toLowerCase();
    return normalized !== 'intact';
  };

  updateDataMetadata();
  updateFollowLiveToggle();

  if (followLiveToggleEl) {
    followLiveToggleEl.addEventListener('click', () => {
      setFollowLiveState(!followLive, { scroll: !followLive });
    });
  }

  if (levelsToggleEl) {
    levelsToggleEl.addEventListener('click', () => {
      if (!levelGroups.supplemental.length) return;
      showAllLevels = !showAllLevels;
      setOrDeleteParam('supportingLevels', showAllLevels ? '1' : '0');
      updateLevelVisibility();
      const newUrl = `${window.location.pathname}?${params.toString()}`;
      window.history.replaceState({}, '', newUrl);
    });
  }

  window.setInterval(() => {
    if (!dataLastUpdateDate) return;
    dataAgeMs = Date.now() - dataLastUpdateDate.getTime();
    const degraded = isFeedDegraded();
    if (degraded !== lastDegradedState) {
      lastDegradedState = degraded;
      applyMarketStatus(currentMarketPhase || 'closed');
    }
    updateDataMetadata();
  }, 1000);

  const showError = (message) => {
    if (!debugEl) return;
    debugEl.style.display = 'block';
    debugEl.textContent = message;
  };

  window.addEventListener('error', (event) => {
    showError(`Runtime error: ${event.message}`);
  });

  window.addEventListener('unhandledrejection', (event) => {
    showError(`Unhandled promise rejection: ${event.reason}`);
  });

  const TIMEFRAMES = [
    { label: '1m', resolution: '1' },
    { label: '5m', resolution: '5' },
    { label: '10m', resolution: '10' },
    { label: '30m', resolution: '30' },
    { label: '1H', resolution: '60' },
    { label: '1D', resolution: '1D' },
    { label: '1W', resolution: '1W' },
  ];
  let activeTimeframe =
    TIMEFRAMES.find((tf) => normalizeResolution(tf.resolution) === currentResolution) || TIMEFRAMES[0];

  const layoutBase = {
    background: { type: 'solid', color: theme === 'light' ? '#ffffff' : '#0b0f14' },
    textColor: theme === 'light' ? '#1b2733' : '#e6edf3',
  };
  const gridBase = {
    vertLines: { color: theme === 'light' ? '#e5e9f0' : '#1f2933' },
    horzLines: { color: theme === 'light' ? '#e5e9f0' : '#1f2933' },
  };

  const TickMarkType = LightweightCharts.TickMarkType;

  const formatTickMark = (time, tickMarkType) => {
    const info = getEtParts(time);
    const timeLabel = `${String(info.hour).padStart(2, '0')}:${String(info.minute).padStart(2, '0')}`;
    const dayLabel = `${info.monthName} ${info.day}`;
    switch (tickMarkType) {
      case TickMarkType.Time:
      case TickMarkType.TimeWithSeconds:
        return timeLabel;
      case TickMarkType.TimeAndDate:
      case TickMarkType.DayOfMonth:
        return dayLabel;
      case TickMarkType.Month:
        return `${info.monthName} ${info.year}`;
      case TickMarkType.Year:
        return `${info.year}`;
      default:
        return dayLabel;
    }
  };

  const container = document.getElementById('tv_chart_container');
  const chart = LightweightCharts.createChart(container, {
    layout: layoutBase,
    grid: gridBase,
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    timeScale: {
      borderColor: theme === 'light' ? '#d1d5db' : '#1f2933',
      timeVisible: true,
      secondsVisible: false,
      tickMarkFormatter: formatTickMark,
    },
    rightPriceScale: {
      borderColor: theme === 'light' ? '#d1d5db' : '#1f2933',
      scaleMargins: { top: 0.1, bottom: 0.25 },
    },
    leftPriceScale: {
      visible: true,
      borderColor: theme === 'light' ? '#d1d5db' : '#1f2933',
      scaleMargins: { top: 0.8, bottom: 0.02 },
    },
    watermark: {
      visible: true,
      fontSize: 22,
      horzAlign: 'left',
      vertAlign: 'bottom',
      color: theme === 'light' ? 'rgba(15,23,42,0.08)' : 'rgba(148,163,184,0.08)',
      text: '',
    },
  });

  sessionOverlayEl = document.createElement('div');
  sessionOverlayEl.className = 'session-overlay';
  container.appendChild(sessionOverlayEl);
  chart.timeScale().subscribeVisibleTimeRangeChange((visibleRange) => {
    renderSessionBands();
    if (suppressFollowDetection || !followLive) {
      return;
    }
    if (!visibleRange || !latestCandleData.length) return;
    const lastBar = latestCandleData[latestCandleData.length - 1];
    if (!lastBar || !Number.isFinite(lastBar.time)) return;
    const rangeTo = Number(visibleRange.to);
    if (!Number.isFinite(rangeTo)) return;
    const toleranceSeconds = Math.max(lastSecondsPerBar, 60);
    if (rangeTo < lastBar.time - toleranceSeconds * 0.5) {
      pauseFollowLive();
    }
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
  const emaTokensInput = parseList(params.get('ema'));
  const emaTokens = emaTokensInput.length ? emaTokensInput : ['9', '21', '50'];
  const emaSeries = emaTokens.reduce((acc, token, idx) => {
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

  let vwapSeries = null;
  const scalePlanToken = (params.get('scale_plan') || 'auto').toLowerCase();

  let lastKnownPrice = null;
  let fetchToken = 0;
  let currentPlanStatus = 'intact';
  let latestPlanNote = 'Plan intact. Risk profile unchanged.';
  let latestNextStep = 'hold_plan';
  let latestMarketNote = null;
  if (isFeedDegraded()) {
    latestMarketNote = 'Data feed degraded — using last Polygon bars.';
  }
  let currentMarketPhase = null;
  let streamSource = null;
  let latestCandleData = [];
  let latestVolumeData = [];
  const DEFAULT_REPLAY_MINUTES = 10;
  const REPLAY_MAX_MINUTES = 180;
  const REPLAY_STEP_MS = 750;
  let isReplaying = false;
  let replayQueue = [];
  let replayIndex = 0;
  let replayTimer = null;

  let planFocusSeries = null;
  let loggedInitialPlan = false;

  const ensurePlanFocusSeries = () => {
    if (planFocusSeries) {
      return planFocusSeries;
    }
    planFocusSeries = chart.addLineSeries({
      color: 'rgba(0,0,0,0)',
      lineWidth: 0,
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: false,
      visible: false,
    });
    return planFocusSeries;
  };

  const collectPlanFocusValues = (plan) => {
    const values = [];
    if (plan) {
      const pushIfFinite = (val) => {
        if (Number.isFinite(val)) {
          values.push(val);
        }
      };
      pushIfFinite(plan.entry);
      pushIfFinite(plan.stop);
      (plan.tps || []).forEach((tp) => pushIfFinite(tp));
      if (plan.runner) {
        pushIfFinite(plan.runner.anchor);
      }
    }
    if (Array.isArray(keyLevels) && keyLevels.length) {
      keyLevels.forEach((level) => {
        if (level && Number.isFinite(level.price)) {
          values.push(level.price);
        }
      });
    }
    return values;
  };

  const applyPlanPriceFocus = (plan, { force = false } = {}) => {
    const focusSeries = ensurePlanFocusSeries();
    if (focusToken !== 'plan') {
      focusSeries.setData([]);
      focusSeries.applyOptions({ visible: false });
      chart.priceScale('right').applyOptions({ autoScale: true });
      hasAppliedFocusRange = false;
      return;
    }
    if (!force && hasAppliedFocusRange) {
      return;
    }
    const values = collectPlanFocusValues(plan);
    if (!values.length) {
      chart.priceScale('right').applyOptions({ autoScale: true });
      hasAppliedFocusRange = false;
      return;
    }
    const minPrice = Math.min(...values);
    const maxPrice = Math.max(...values);
    const span = Math.max(0.01, maxPrice - minPrice);
    const padPct = 0.008;
    const baselinePad = Math.max(span * padPct, span * 0.05);
    const atrPad = Number.isFinite(plan?.atr) && plan.atr > 0 ? plan.atr * 0.5 : 0;
    const pad = Math.max(baselinePad, atrPad);
    const startTime = latestCandleData.length ? latestCandleData[0].time : undefined;
    const endTime = latestCandleData.length ? latestCandleData[latestCandleData.length - 1].time : undefined;
    if (startTime == null || endTime == null) {
      focusSeries.setData([]);
      focusSeries.applyOptions({ visible: false });
      chart.priceScale('right').applyOptions({ autoScale: true });
      hasAppliedFocusRange = false;
      return;
    }
    focusSeries.setData([
      { time: startTime, value: minPrice - pad },
      { time: endTime, value: maxPrice + pad },
    ]);
    focusSeries.applyOptions({ visible: true });
    chart.priceScale('right').applyOptions({ autoScale: true });
    hasAppliedFocusRange = true;
  };

  const applyTimeCentering = ({ force = false } = {}) => {
    if (!centerTimeToken) {
      return;
    }
    if (!force && hasAppliedTimeCenter) {
      return;
    }
    if (!latestCandleData.length) {
      return;
    }
    const timeScale = chart.timeScale();
    if (centerTimeIsLatest) {
      timeScale.scrollToPosition(0, true);
      hasAppliedTimeCenter = true;
      return;
    }
    if (!Number.isFinite(centerTimeNumeric)) {
      return;
    }
    let closestIndex = -1;
    let smallestDiff = Number.POSITIVE_INFINITY;
    latestCandleData.forEach((bar, idx) => {
      if (!bar || !Number.isFinite(bar.time)) return;
      const diff = Math.abs(bar.time - centerTimeNumeric);
      if (diff < smallestDiff) {
        smallestDiff = diff;
        closestIndex = idx;
      }
    });
    if (closestIndex === -1) {
      return;
    }
    const logicalRange = timeScale.getVisibleLogicalRange();
    const halfSpan = logicalRange
      ? Math.max(5, (logicalRange.to - logicalRange.from) / 2)
      : Math.max(5, Math.floor(latestCandleData.length / 4) || 20);
    try {
      timeScale.setVisibleLogicalRange({
        from: closestIndex - halfSpan,
        to: closestIndex + halfSpan,
      });
      hasAppliedTimeCenter = true;
    } catch (err) {
      console.warn('center_time adjustment failed', err);
    }
  };

  const applyPlanTimeWindow = ({ force = false } = {}) => {
    if (focusToken !== 'plan') {
      hasAppliedTimeWindow = false;
      return;
    }
    if (!force && hasAppliedTimeWindow) {
      return;
    }
    const totalBars = latestCandleData.length;
    if (!totalBars) {
      return;
    }
    const windowSize = Math.min(Math.max(120, Math.round(totalBars * 0.4)), 360);
    const fromIndex = Math.max(totalBars - windowSize, 0);
    const toIndex = Math.max(totalBars - 1, fromIndex + 1);
    try {
      chart.timeScale().setVisibleLogicalRange({
        from: fromIndex,
        to: toIndex,
      });
      hasAppliedTimeWindow = true;
    } catch (err) {
      console.warn('plan time window adjustment failed', err);
    }
  };
  let replayPrevPhase = null;
  let replayPrevNote = null;
  let replayHadStream = false;
  let replayFetchToken = 0;
  let replaySavedCandleData = [];
  let replaySavedVolumeData = [];
  let replaySavedVisibleRange = null;
  let replayStatusEl = null;
  let replayStartButton = null;
  let replayStopButton = null;
  let replayMinutesInput = null;
  let replayStatusMessage = '';
  const replayConfig = { minutes: DEFAULT_REPLAY_MINUTES };

  const PLAN_STATUS_META = {
    intact: { label: 'Plan Intact', className: 'status-pill--intact' },
    at_risk: { label: 'Plan At Risk', className: 'status-pill--risk' },
    invalidated: { label: 'Plan Invalidated', className: 'status-pill--invalid' },
    reversal: { label: 'Plan Reversal', className: 'status-pill--reversal' },
  };

  const statusLabel = (token) => {
    if (!token) return null;
    const normalized = token.toLowerCase();
    const meta = PLAN_STATUS_META[normalized];
    if (meta) return meta.label;
    return token.replace(/_/g, ' ').replace(/\b\w/g, (char) => char.toUpperCase());
  };

  const NEXT_STEP_LABELS = {
    hold_plan: 'Hold plan',
    tighten_stop: 'Tighten stop',
    plan_invalidated: 'Plan invalidated',
    consider_reversal: 'Consider reversal',
  };

  const MARKET_PHASE_META = {
    regular: { label: 'Market Open', className: 'status-pill--intact', note: 'Market open.' },
    premarket: { label: 'Pre-Market', className: 'status-pill--risk', note: 'Pre-market session — liquidity thinner.' },
    afterhours: { label: 'After Hours', className: 'status-pill--risk', note: 'After hours — liquidity thinner.' },
    closed: { label: 'Market Closed', className: 'status-pill--invalid', note: 'Market closed — live updates limited.' },
    replay: { label: 'Replay Mode', className: 'status-pill--risk', note: 'Replaying recent price action.' },
  };

  const formatNextStep = (token) => {
    if (!token) return null;
    const normalized = token.toLowerCase();
    if (NEXT_STEP_LABELS[normalized]) return NEXT_STEP_LABELS[normalized];
    return normalized.replace(/_/g, ' ').replace(/\b\w/g, (ch) => ch.toUpperCase());
  };

  const clampReplayMinutes = (value) => {
    const num = Number.parseInt(value, 10);
    if (!Number.isFinite(num)) return DEFAULT_REPLAY_MINUTES;
    const clamped = Math.min(Math.max(num, 1), REPLAY_MAX_MINUTES);
    return clamped;
  };

  const parseEventTimestamp = (payload) => {
    if (!payload) return null;
    const raw = payload.ts || payload.timestamp || payload.time || payload.t;
    if (raw === null || raw === undefined) return null;
    if (typeof raw === 'number') {
      if (raw > 10_000_000_000) {
        return Math.floor(raw / 1000);
      }
      return Math.floor(raw);
    }
    if (typeof raw === 'string') {
      const num = Number(raw);
      if (Number.isFinite(num)) {
        return num > 10_000_000_000 ? Math.floor(num / 1000) : Math.floor(num);
      }
      const date = new Date(raw);
      if (!Number.isNaN(date.getTime())) {
        return Math.floor(date.getTime() / 1000);
      }
    }
    if (raw instanceof Date && !Number.isNaN(raw.getTime())) {
      return Math.floor(raw.getTime() / 1000);
    }
    return null;
  };

  const updateStatusNote = () => {
    if (!planStatusNoteEl) return;
    const parts = [];
    if (latestPlanNote) parts.push(latestPlanNote);
    if (latestNextStep) {
      const label = formatNextStep(latestNextStep);
      if (label) parts.push(`Next: ${label}`);
    }
    if (latestMarketNote) parts.push(latestMarketNote);
    if (parts.length) {
      planStatusNoteEl.textContent = parts.join(' • ');
      planStatusNoteEl.classList.add('active');
    } else {
      planStatusNoteEl.textContent = '';
      planStatusNoteEl.classList.remove('active');
    }
  };

  const applyPlanStatus = (status, note, nextStep, rrValue) => {
    if (typeof status === 'string' && status) {
      currentPlanStatus = status.toLowerCase();
    }
    if (typeof note === 'string' && note.trim()) {
      latestPlanNote = note.trim();
    }
    if (typeof nextStep === 'string' && nextStep.trim()) {
      latestNextStep = nextStep.trim();
    }
    const meta = PLAN_STATUS_META[currentPlanStatus] || PLAN_STATUS_META.intact;
    if (headerPlanStatusEl) {
      headerPlanStatusEl.textContent = meta.label;
      headerPlanStatusEl.className = `status-pill ${meta.className}`;
    }
    if (typeof rrValue === 'number' && headerRREl) {
      headerRREl.textContent = `R:R (TP1): ${rrValue.toFixed(2)}`;
    }
    lastDegradedState = isFeedDegraded();
    updateStatusNote();
  };

  const applyMarketStatus = (phase, note) => {
    if (typeof phase === 'string' && phase.trim()) {
      currentMarketPhase = phase.trim().toLowerCase();
    } else if (!currentMarketPhase) {
      currentMarketPhase = 'closed';
    }
    const meta = MARKET_PHASE_META[currentMarketPhase] || MARKET_PHASE_META.closed;
    if (headerMarketEl) {
      headerMarketEl.textContent = meta.label;
      headerMarketEl.className = `status-pill ${meta.className}`;
    }
    if (note && typeof note === 'string' && note.trim()) {
      latestMarketNote = note.trim();
    } else {
      latestMarketNote = meta.note;
    }
    if (isFeedDegraded()) {
      const degradeCopy = 'Data feed degraded — using last Polygon bars.';
      if (latestMarketNote) {
        if (!latestMarketNote.toLowerCase().includes('degraded')) {
          latestMarketNote = `${latestMarketNote} · ${degradeCopy}`;
        }
      } else {
        latestMarketNote = degradeCopy;
      }
    }
    updateStatusNote();
  };

  const normalizeMarketPhase = (token) => {
    if (!token) return null;
    const normalized = token.toLowerCase();
    if (normalized.includes('pre')) return 'premarket';
    if (normalized.includes('after') || normalized.includes('post')) return 'afterhours';
    if (normalized.includes('closed')) return 'closed';
    if (
      normalized.includes('regular') ||
      normalized.includes('open') ||
      normalized.includes('rth') ||
      normalized.includes('midday') ||
      normalized.includes('session') ||
      normalized.includes('intraday')
    ) {
      return 'regular';
    }
    return null;
  };

  const initialMarketPhase =
    normalizeMarketPhase(sessionPhaseParam) ||
    normalizeMarketPhase(sessionStatusParam) ||
    normalizeMarketPhase(marketStatusParam) ||
    (marketStatusParam === 'open' ? 'regular' : null) ||
    (liveFlag ? 'regular' : null) ||
    'closed';

  const initialMarketNote = sessionBannerParam || null;

  const setCurrentPlanId = (planId, planVersion) => {
    currentPlanId = planId ? planId.trim() : null;
    currentPlanVersion = planVersion ? String(planVersion).trim() : null;
    if (currentPlanId) params.set('plan_id', currentPlanId);
    else params.delete('plan_id');
    if (currentPlanVersion) params.set('plan_version', currentPlanVersion);
    else params.delete('plan_version');
    const newUrl = `${window.location.pathname}?${params.toString()}`;
    window.history.replaceState({}, '', newUrl);
  };

  function setOrDeleteParam(key, value) {
    if (value === null || value === undefined || value === '') {
      params.delete(key);
      return;
    }
    params.set(key, String(value));
  }

  const numericArray = (value) => {
    if (!Array.isArray(value)) return [];
    return value
      .map((item) => Number(item))
      .filter((item) => Number.isFinite(item));
  };

  applyPlanStatus(currentPlanStatus, latestPlanNote, latestNextStep, null);
  applyMarketStatus(initialMarketPhase, initialMarketNote);

  const matchesPlan = (incomingPlanId) => {
    if (!currentPlanId) return true;
    return (incomingPlanId || '').trim() === currentPlanId;
  };

  const handlePlanStateEvent = (plans) => {
    if (!Array.isArray(plans) || plans.length === 0) return;
    let candidate = null;
    if (currentPlanId) {
      candidate = plans.find((item) => (item?.plan_id || '').trim() === currentPlanId);
    }
    if (!candidate) {
      candidate = plans[0];
    }
    if (!candidate) return;
    applyPlanStatus(candidate.status || currentPlanStatus, candidate.note, candidate.next_step, candidate.rr_to_t1);
  };

  const handlePlanDeltaEvent = (payload) => {
    if (!payload || !matchesPlan(payload.plan_id)) return;
    const changes = payload.changes || {};
    const statusToken = changes.status || payload.status || currentPlanStatus;
    const rrValue = Number.isFinite(changes.rr_to_t1) ? changes.rr_to_t1 : null;
    applyPlanStatus(statusToken, changes.note, changes.next_step, rrValue);
    const eventTs = parseEventTimestamp(payload);
    const logParts = [];
    if (changes.status) {
      logParts.push(`Status → ${statusLabel(changes.status) || changes.status}`);
    }
    if (changes.note) {
      logParts.push(changes.note);
    }
    if (changes.next_step) {
      const stepLabel = formatNextStep(changes.next_step);
      if (stepLabel) {
        logParts.push(`Next: ${stepLabel}`);
      }
    }
    if (logParts.length) {
      appendPlanLogEntry(logParts.join(' • '), eventTs, isAlertSeverity(statusToken) ? 'alert' : 'info');
    }
    if (Number.isFinite(changes.last_price)) {
      lastKnownPrice = changes.last_price;
      updateHeaderPricing(lastKnownPrice);
      renderPlanPanel(lastKnownPrice);
      updatePlanPanelLastPrice(lastKnownPrice);
    }
    if (changes.next_plan_id) {
      pendingAutoReplan = {
        planId: String(changes.next_plan_id).trim(),
        planVersion: changes.next_plan_version ? String(changes.next_plan_version).trim() : null,
        note: changes.note || null,
      };
      renderAutoReplanBanner();
      if (autoRearmEnabled) {
        adoptPendingReplan({ viaAuto: true });
      } else {
        showToast('Next plan ready — reload when you are set.');
      }
    }
    if (Number.isFinite(eventTs)) {
      setDataLastUpdate(eventTs);
    }
  };

  const handlePlanFullEvent = (payload) => {
    if (!payload) return;
    const planBlock = payload.plan || {};
    if (!matchesPlan(planBlock.plan_id)) return;
    applyPlanStatus('intact', 'Plan updated. Review levels.', 'hold_plan', planBlock.rr_to_t1);
    const eventTs = parseEventTimestamp(payload);
    appendPlanLogEntry('Plan refreshed — review latest guidance.', eventTs, 'info');
    if (Number.isFinite(eventTs)) {
      setDataLastUpdate(eventTs);
    }
  };

  const handleTickEvent = (payload) => {
    if (!payload || isReplaying) return;
    const eventTs = parseEventTimestamp(payload);
    const price = Number.isFinite(payload.p) ? payload.p : Number.isFinite(payload.close) ? payload.close : null;
    if (price === null) return;
    lastKnownPrice = price;
    updateHeaderPricing(price);
    updatePlanPanelLastPrice(price);
    updateRealtimeBar(price, payload);
    if (Number.isFinite(eventTs)) {
      setDataLastUpdate(eventTs);
    }
  };

  const connectStream = () => {
    if (isReplaying || typeof EventSource === 'undefined') return;
    const streamUrl = `${baseUrl}/stream/${symbol}`;
    if (streamSource) {
      streamSource.close();
      streamSource = null;
    }
    try {
      streamSource = new EventSource(streamUrl);
      streamSource.onmessage = (msg) => {
        try { pulsePlanLogIndicator(); } catch {}
        if (!msg?.data) return;
        let envelope;
        try {
          envelope = JSON.parse(msg.data);
        } catch (err) {
          console.warn('Failed to parse stream message', err);
          return;
        }
        const event = envelope?.event;
        if (!event || !event.t) return;
        switch (event.t) {
          case 'plan_state':
            handlePlanStateEvent(event.plans);
            break;
          case 'plan_delta':
            handlePlanDeltaEvent(event);
            break;
          case 'plan_full':
            handlePlanFullEvent(event.payload);
            break;
          case 'tick':
            handleTickEvent(event);
            break;
          case 'bar':
            handleTickEvent(event);
            break;
          case 'market_status':
            applyMarketStatus(event.phase, event.note);
            break;
          case 'heartbeat': {
            const ts = parseEventTimestamp(event) ?? Math.floor(Date.now() / 1000);
            setDataLastUpdate(ts);
            break;
          }
          default:
            break;
        }
      };
      streamSource.onerror = () => {
        if (streamSource) {
          streamSource.close();
          streamSource = null;
        }
        window.setTimeout(connectStream, 5000);
      };
    } catch (err) {
      console.error('Stream connection failed', err);
      window.setTimeout(connectStream, 5000);
    }
  };

  const priceLineMap = new Map();
  const buildLevelLineId = (label, price) => {
    const safeLabel = (label || '').trim().replace(/\s+/g, '_');
    return `level:${safeLabel}:${Number(price).toFixed(4)}`;
  };
  const formatLevelLabel = (level, idx) => {
    const label = typeof level.label === 'string' && level.label.trim() ? level.label.trim() : null;
    return label || `Level ${idx + 1}`;
  };
  const setPriceLine = (id, options) => {
    if (!Number.isFinite(options.price)) return;
    const existing = priceLineMap.get(id);
    if (existing) {
      existing.applyOptions(options);
      return existing;
    }
    const line = candleSeries.createPriceLine(options);
    priceLineMap.set(id, line);
    return line;
  };
  const prunePriceLines = (activeIds) => {
    for (const [id, line] of priceLineMap.entries()) {
      if (!activeIds.has(id)) {
        candleSeries.removePriceLine(line);
        priceLineMap.delete(id);
      }
    }
  };

  const refreshLevelLines = () => {
    if (!candleSeries) return;
    const desiredIds = new Set();
    const orderedLevels = [...keyLevels]
      .filter((level) => Number.isFinite(level.price))
      .sort((a, b) => b.price - a.price)
      .slice(0, SUPPORTING_LEVEL_LIMIT);
    orderedLevels.forEach((level, idx) => {
        const label = formatLevelLabel(level, idx);
        const id = buildLevelLineId(label, level.price);
        desiredIds.add(id);
        setPriceLine(id, {
          price: level.price,
          color: '#94a3b8',
          title: label,
          lineWidth: 1,
          lineStyle: LightweightCharts.LineStyle.Dotted,
        });
      });
    for (const [id, line] of priceLineMap.entries()) {
      if (id.startsWith('level:') && !desiredIds.has(id)) {
        candleSeries.removePriceLine(line);
        priceLineMap.delete(id);
      }
    }
  };

  const setWatermark = () => {
    const tfLabel = activeTimeframe ? activeTimeframe.label : currentResolution;
    chart.applyOptions({
      watermark: {
        text: `${symbol} · ${tfLabel}`,
      },
    });
  };

  const TIMEFRAME_REFRESH_MS = 60000;

  const initializeTimeframes = () => {
    timeframeSwitcherEl.innerHTML = '';
    TIMEFRAMES.forEach((tf) => {
      const button = document.createElement('button');
      button.className = 'timeframe-button';
      button.textContent = tf.label;
      button.dataset.resolution = tf.resolution;
      if (normalizeResolution(tf.resolution) === currentResolution) {
        button.classList.add('active');
        activeTimeframe = tf;
      }
      button.addEventListener('click', () => {
        if (normalizeResolution(tf.resolution) === currentResolution) return;
        currentResolution = normalizeResolution(tf.resolution);
        activeTimeframe = tf;
        params.set('interval', tf.label);
        const newUrl = `${window.location.pathname}?${params.toString()}`;
        window.history.replaceState({}, '', newUrl);
        Array.from(timeframeSwitcherEl.querySelectorAll('button')).forEach((btn) => btn.classList.remove('active'));
        button.classList.add('active');
        setWatermark();
        fetchBars();
      });
      timeframeSwitcherEl.appendChild(button);
    });
  };

  const formatPrice = (value) => (Number.isFinite(value) ? value.toFixed(2) : '—');
  const formatPercentage = (value) => {
    const num = toNumber(value);
    return num !== null ? `${(num * 100).toFixed(0)}%` : '—';
  };

  const formatProbability = (value) => {
    const num = toNumber(value);
    if (num === null) return null;
    return `${Math.round(num * 100)}% POT`;
  };

  const estimateDuration = () => {
    if (Number.isFinite(mergedPlanMeta.horizon_minutes)) {
      const minutes = mergedPlanMeta.horizon_minutes;
      if (minutes >= 1440) {
        const days = minutes / 1440;
        return `Stay in trade ≈ ${days.toFixed(days >= 2 ? 0 : 1)} day${days >= 2 ? 's' : ''}`;
      }
      if (minutes >= 60) {
        const hours = minutes / 60;
        return `Stay in trade ≈ ${hours.toFixed(hours >= 2 ? 0 : 1)} hour${hours >= 2 ? 's' : ''}`;
      }
      return `Stay in trade ≈ ${minutes.toFixed(0)} minutes`;
    }
    let styleToken = (mergedPlanMeta.style || '').toLowerCase();
    if (!styleToken) {
      const minutes = parseInt(currentResolution, 10);
      if (Number.isFinite(minutes)) {
        if (minutes <= 5) styleToken = 'scalp';
        else if (minutes <= 60) styleToken = 'intraday';
        else styleToken = 'swing';
      }
    }
    if (styleToken === 'scalp' || styleToken === '0dte') return 'Stay in trade ≈ 30–60 minutes';
    if (styleToken === 'intraday') return 'Stay in trade ≈ 2–4 hours';
    if (styleToken === 'swing') return 'Stay in trade ≈ 3–5 days';
    if (styleToken === 'leaps') return 'Stay in trade: multi-week campaign';
    return null;
  };

  const horizonShort = () => {
    if (Number.isFinite(mergedPlanMeta.horizon_minutes)) {
      const minutes = mergedPlanMeta.horizon_minutes;
      if (minutes >= 1440) {
        const days = minutes / 1440;
        return `${days.toFixed(days >= 2 ? 0 : 1)}d`;
      }
      if (minutes >= 60) {
        const hours = minutes / 60;
        return `${hours.toFixed(hours >= 2 ? 0 : 1)}h`;
      }
      return `${minutes.toFixed(0)}m`;
    }
    const styleToken = (mergedPlanMeta.style || '').toLowerCase();
    if (styleToken === 'scalp' || styleToken === '0dte') return '30–60m';
    if (styleToken === 'intraday') return '2–4h';
    if (styleToken === 'swing') return '3–5d';
    if (styleToken === 'leaps') return 'multi‑week';
    return null;
  };

  const renderHeader = () => {
    const bias = (mergedPlanMeta.bias || currentPlan.direction || '').toLowerCase();
    if (headerSymbolEl) headerSymbolEl.textContent = symbol;
    if (headerStrategyEl) {
      const styleTokenRaw = mergedPlanMeta.style_display || mergedPlanMeta.style || '';
      const styleLabel = styleTokenRaw ? String(styleTokenRaw).toUpperCase() : '';
      const strategyLabel = mergedPlanMeta.strategy || currentPlan.strategy || '';
      headerStrategyEl.textContent = [styleLabel, strategyLabel].filter(Boolean).join(' · ');
    }
    if (headerBiasEl) headerBiasEl.textContent = bias ? `Bias: ${bias === 'long' ? 'Long 🟢' : 'Short 🔴'}` : '';
    const rawConfidence =
      toNumber(mergedPlanMeta.confidence) ??
      toNumber(planMeta.confidence_score) ??
      toNumber(planMeta.plan_confidence) ??
      toNumber(planMeta.plan?.confidence) ??
      paramConfidence;
    const displayConfidence = rawConfidence !== null && rawConfidence > 0 ? rawConfidence : null;
    if (headerConfidenceEl) {
      headerConfidenceEl.textContent = displayConfidence !== null ? `Confidence: ${(displayConfidence * 100).toFixed(0)}%` : '';
    }
    const fallbackRRForHeader = (() => {
      const entry =
        Number.isFinite(currentPlan.entry) && Number.isFinite(currentPlan.stop)
          ? currentPlan.entry
          : mergedPlanMeta.entry;
      const stop =
        Number.isFinite(currentPlan.entry) && Number.isFinite(currentPlan.stop)
          ? currentPlan.stop
          : mergedPlanMeta.stop;
      const targetsSource =
        Number.isFinite(currentPlan.entry) && Number.isFinite(currentPlan.stop)
          ? currentPlan.tps
          : Array.isArray(mergedPlanMeta.targets)
            ? mergedPlanMeta.targets
            : [];
      const target = targetsSource?.find((value) => Number.isFinite(value));
      const direction = (mergedPlanMeta.bias || currentPlan.direction || 'long').toLowerCase();
      if (!Number.isFinite(entry) || !Number.isFinite(stop) || !Number.isFinite(target)) return null;
      const risk = direction === 'long' ? entry - stop : stop - entry;
      const reward = direction === 'long' ? target - entry : entry - target;
      if (risk <= 0 || reward <= 0) return null;
      return reward / risk;
    })();
    if (headerRREl) {
      const rrValue = toNumber(mergedPlanMeta.risk_reward);
      const rrDisplay = rrValue !== null && rrValue > 0 ? rrValue : fallbackRRForHeader;
      headerRREl.textContent = rrDisplay !== null ? `R:R (TP1): ${rrDisplay.toFixed(2)}` : '';
    }
    if (headerDurationEl) {
      headerDurationEl.textContent = estimateDuration() || '';
    }
    updateDataMetadata();
  };

  const renderPlanPanel = (lastPrice) => {
    if (!planPanelBodyEl) return;
    const targetsMeta = Array.isArray(mergedPlanMeta.target_meta) ? mergedPlanMeta.target_meta : [];
    const warnings = Array.isArray(mergedPlanMeta.warnings) ? mergedPlanMeta.warnings : [];

    const targetsList = currentPlan.tps
      .map((tp, idx) => {
        if (!Number.isFinite(tp)) return null;
        const meta = targetsMeta[idx] || {};
        const sequence = Number.isFinite(meta.sequence) ? meta.sequence : idx + 1;
        const label = sequence >= 3 ? 'Runner' : `TP${sequence}`;
        const rrVal = toNumber(meta.rr);
        const rr = rrVal !== null ? ` · R:R ${rrVal.toFixed(2)}` : '';
        const pot = formatProbability(meta.prob_touch);
        const emVal = toNumber(meta.em_fraction);
        const em = emVal !== null ? ` · ${emVal.toFixed(2)}×EM` : '';
        return `<li><strong>${label}:</strong> ${formatPrice(tp)}${em}${pot ? ` · ${pot}` : ''}${rr}</li>`;
      })
      .filter(Boolean)
      .join('');

    const strategyLabel = mergedPlanMeta.strategy || currentPlan.strategy || '';
    const warningsList = warnings
      .map((warning) => `<li>${warning}</li>`)
      .join('');
    const checklistHtml = (() => {
      if (warningsList) return warningsList;
      const lowerStrategy = strategyLabel.toLowerCase();
      const steps = [];
      if (/vwap/.test(lowerStrategy)) steps.push('Wait for breakout/retest above VWAP with confirming volume.');
      if (/orb|opening range/.test(lowerStrategy)) steps.push('Confirm opening range break and successful retest before entry.');
      if (/pullback|trend/.test(lowerStrategy)) steps.push('Enter on pullback to EMA stack with higher low and momentum rebuild.');
      if (/reversal/.test(lowerStrategy)) steps.push('Require double-top/bottom with neckline reclaim before entering.');
      if (!steps.length) steps.push('Confirm structure alignment and reclaim of entry trigger before acting.');
      return steps.map((step) => `<li>${escapeHtml(step)}</li>`).join('');
    })();

    const buildConfluence = () => {
      const biasToken = (mergedPlanMeta.bias || currentPlan.direction || '').toLowerCase();
      const COLOR_PRIORITY = { red: 0, yellow: 1, green: 2 };
      const itemsMap = new Map();

      const normalizeLabel = (value) =>
        value
          .replace(/_/g, ' ')
          .replace(/\b([a-z])/g, (match) => match.toUpperCase())
          .replace(/\s+/g, ' ')
          .trim();

      const pushItem = (item) => {
        if (!item || !item.id) return;
        const existing = itemsMap.get(item.id);
        if (!existing || COLOR_PRIORITY[item.color] > COLOR_PRIORITY[existing.color]) {
          itemsMap.set(item.id, item);
        }
      };

      const classifyToken = (rawToken) => {
        if (!rawToken && rawToken !== 0) return null;
        const original = String(rawToken).trim();
        if (!original) return null;
        const normalized = original.toLowerCase();

        const positiveForBias = (flag) => {
          if (!biasToken) return flag;
          if (biasToken === 'long') return flag;
          if (biasToken === 'short') return !flag;
          return flag;
        };

        if (/ema/.test(normalized)) {
          const isDown = /down/.test(normalized);
          const color = positiveForBias(!isDown) ? 'green' : 'red';
          return { id: 'ema', label: 'EMA Alignment', color };
        }
        if (/vwap|avwap/.test(normalized)) {
          const isBelow = /below/.test(normalized);
          const color = positiveForBias(!isBelow) ? 'green' : 'red';
          return { id: 'vwap', label: 'VWAP Context', color };
        }
        if (/mtf|multi-timeframe|htf/.test(normalized)) {
          return { id: 'mtf', label: 'MTF Alignment', color: 'green' };
        }
        if (/volume|poc|vah|val/.test(normalized)) {
          return { id: 'volume', label: 'Volume Profile', color: 'yellow' };
        }
        if (/liquidity|stoprun|liquidity_pool/.test(normalized)) {
          return { id: 'liquidity', label: 'Liquidity Pools', color: 'yellow' };
        }
        if (/fvg|imbalance/.test(normalized)) {
          return { id: 'fvg', label: 'Fair Value Gap', color: 'yellow' };
        }
        if (/gamma|dealer|oi_wall/.test(normalized)) {
          return { id: 'gamma', label: 'Gamma Magnet', color: 'yellow' };
        }
        if (/opening\s*range|orh|orl/.test(normalized)) {
          return { id: 'opening_range', label: 'Opening Range Levels', color: 'green' };
        }
        if (/session\s*(high|low)|daily/.test(normalized)) {
          return { id: 'session_levels', label: 'Session Levels', color: 'green' };
        }
        if (/prob|stat|quantile/.test(normalized)) {
          return { id: 'stats', label: 'Historical Stats', color: 'green' };
        }
        if (/trend/.test(normalized)) {
          const aligned = !/misaligned|weak/.test(normalized);
          return { id: 'trend', label: 'Trend Structure', color: aligned ? 'green' : 'yellow' };
        }
        if (/runner/.test(normalized)) {
          return { id: 'runner', label: 'Runner Management', color: 'green' };
        }
        const id = normalized.replace(/[^a-z0-9]+/g, '_');
        return { id, label: normalizeLabel(original), color: 'green' };
      };

      const registerStrings = (values) => {
        if (!Array.isArray(values)) return;
        values.forEach((value) => {
          const item = classifyToken(value);
          if (item) pushItem(item);
        });
      };

      registerStrings(planLayersMeta.confidence_factors);
      registerStrings(planLayersMeta.confluence);
      registerStrings(mergedPlanMeta.confidence_factors);
      registerStrings(mergedPlanMeta.confluence);

      if (planLayersMeta.features && typeof planLayersMeta.features === 'object') {
        const features = planLayersMeta.features;
        if (features.ema_trend_up === true) registerStrings(['EMA trend up']);
        if (features.ema_trend_down === true) registerStrings(['EMA trend down']);
        if (Number.isFinite(features.vwap) && Number.isFinite(lastKnownPrice)) {
          const above = lastKnownPrice >= features.vwap;
          registerStrings([above ? 'Above VWAP' : 'Below VWAP']);
        }
      }

      if (!itemsMap.size) {
        if (mergedPlanMeta.bias) {
          pushItem({
            id: 'trend_bias',
            label: `Trend ${mergedPlanMeta.bias.toString().toUpperCase()}`,
            color: 'green',
          });
        }
        if (strategyLabel) {
          pushItem({ id: 'strategy', label: `Strategy ${normalizeLabel(strategyLabel)}`, color: 'green' });
        }
        if (!itemsMap.size) {
          pushItem({ id: 'monitor', label: 'Monitor structure', color: 'yellow' });
        }
      }

      return Array.from(itemsMap.values())
        .sort((a, b) => COLOR_PRIORITY[b.color] - COLOR_PRIORITY[a.color])
        .slice(0, 5);
    };

    const confluenceItems = buildConfluence();

    const horizonCopy = horizonShort() || '—';
    const rrValue = toNumber(mergedPlanMeta.risk_reward);
    const rrFallback = (() => {
      const useScaled = Number.isFinite(currentPlan.entry) && Number.isFinite(currentPlan.stop);
      const entry = useScaled ? currentPlan.entry : mergedPlanMeta.entry;
      const stop = useScaled ? currentPlan.stop : mergedPlanMeta.stop;
      const targetPool = useScaled
        ? currentPlan.tps
        : Array.isArray(mergedPlanMeta.targets)
          ? mergedPlanMeta.targets
          : [];
      const target = targetPool?.find((value) => Number.isFinite(value));
      const direction = (mergedPlanMeta.bias || currentPlan.direction || 'long').toLowerCase();
      if (!Number.isFinite(entry) || !Number.isFinite(stop) || !Number.isFinite(target)) return null;
      const risk = direction === 'long' ? entry - stop : stop - entry;
      const reward = direction === 'long' ? target - entry : entry - target;
      if (risk <= 0 || reward <= 0) return null;
      return reward / risk;
    })();
    const rrCopy = rrValue !== null && rrValue > 0 ? rrValue.toFixed(2) : rrFallback !== null ? rrFallback.toFixed(2) : '—';
    const runnerNote = mergedPlanMeta.runner && mergedPlanMeta.runner.note ? mergedPlanMeta.runner.note : null;
    const quotesNotice = planLayersMeta.quotes_notice || mergedPlanMeta.quotes_notice || null;

    const lastPriceCopy = Number.isFinite(lastPrice) ? formatPrice(lastPrice) : '—';
    const replayMinutesValue = clampReplayMinutes(replayConfig.minutes);
    replayConfig.minutes = replayMinutesValue;
    const replayStatusText = replayStatusMessage || (isReplaying ? 'Replay in progress…' : '');
    const startDisabledAttr = isReplaying ? 'disabled' : '';
    const stopDisabledAttr = isReplaying ? '' : 'disabled';

    planPanelBodyEl.innerHTML = `
      <div class="plan-panel__section">
        <div class="plan-metrics">
          <span><small>Entry</small><strong>${formatPrice(mergedPlanMeta.entry)}</strong></span>
          <span><small>Stop</small><strong>${formatPrice(mergedPlanMeta.stop)}</strong></span>
          <span><small>Last Price</small><strong id="plan_last_price_value">${lastPriceCopy}</strong></span>
          <span><small>Horizon</small><strong>${horizonCopy}</strong></span>
          <span><small>R:R (TP1)</small><strong>${rrCopy}</strong></span>
        </div>
        ${
          strategyLabel
            ? `<div style="margin-top:6px;font-size:12px;letter-spacing:0.06em;text-transform:uppercase;color:rgba(148,163,184,0.78);">Strategy: ${escapeHtml(strategyLabel)}</div>`
            : ''
        }
      </div>
      <div class="plan-panel__section">
        <h3>Targets</h3>
        <ul class="plan-panel__targets">
          ${targetsList || '<li>No targets supplied.</li>'}
        </ul>
      </div>
      ${
        runnerNote
          ? `<div class="plan-panel__section">
              <h3>Runner Guidance</h3>
              <p>${runnerNote}</p>
            </div>`
          : ''
      }
      <div class="plan-panel__section">
        <h3>Pre-Entry Checklist</h3>
        <ul class="plan-panel__warnings">
          ${checklistHtml}
        </ul>
      </div>
      <div class="plan-panel__section">
        <h3>Confluence</h3>
        ${
          confluenceItems.length
            ? `<ul class="plan-panel__confluence">
                ${confluenceItems
                  .map(
                    (item) =>
                      `<li style="display:flex;align-items:center;gap:8px;font-size:13px;">` +
                      `<span style="display:inline-block;width:10px;height:10px;border-radius:50%;background:${
                        item.color === 'green'
                          ? '#4ade80'
                          : item.color === 'yellow'
                            ? '#facc15'
                            : '#f87171'
                      }"></span>${escapeHtml(item.label)}</li>`,
                  )
                  .join('')}
              </ul>`
            : '<p style="font-size:12px;color:rgba(148,163,184,0.7);margin:6px 0 0;">No confluence signals yet.</p>'
        }
      </div>
      ${
        quotesNotice
          ? `<div class="plan-panel__section"><div class="plan-panel__notice plan-panel__notice--warn">${escapeHtml(quotesNotice)}</div></div>`
          : ''
      }
      <div class="plan-panel__section auto-replan" id="auto_replan_section" hidden>
        <h3>Next Plan Ready</h3>
        <p id="auto_replan_message"></p>
        <div class="auto-replan__cta">
          <button type="button" class="plan-replay__button" id="auto_replan_apply">Reload Plan</button>
        </div>
      </div>
      <div class="plan-panel__section">
        <h3>Plan Log <span id="plan_log_indicator" style="display:inline-block;width:8px;height:8px;border-radius:50%;background:#22c55e;margin-left:6px;opacity:.6;"></span></h3>
        <ul class="plan-log" id="plan_log"></ul>
        <p class="plan-log__empty" id="plan_log_empty">No management updates yet.</p>
      </div>
      <div class="plan-panel__section" id="scenario_plans_section">
        <h3>Scenario Plans</h3>
        <div class="plan-replay__controls" id="scenario_controls">
          <div class="scenario-style-group" role="group" aria-label="Scenario style">
            <button type="button" class="plan-replay__button" data-scenario-style="scalp">Scalp</button>
            <button type="button" class="plan-replay__button plan-replay__button--active" data-scenario-style="intraday">Intraday</button>
            <button type="button" class="plan-replay__button" data-scenario-style="swing">Swing</button>
            <button type="button" class="plan-replay__button" data-scenario-style="reversal" disabled title="Reversal strategy coming soon">Reversal</button>
          </div>
        </div>
        <p id="scenario_status" class="plan-replay__status"></p>
      </div>
      <div class="plan-panel__section plan-replay">
        <h3>Market Replay</h3>
        <div class="plan-replay__controls">
          <input
            id="market_replay_minutes"
            class="plan-replay__input"
            type="number"
            min="1"
            max="${REPLAY_MAX_MINUTES}"
            step="1"
            value="${replayMinutesValue}"
            aria-label="Minutes to replay"
          />
          <button id="market_replay_start" type="button" class="plan-replay__button" ${startDisabledAttr}>Start Replay</button>
          <button id="market_replay_stop" type="button" class="plan-replay__button" ${stopDisabledAttr}>Stop</button>
        </div>
        <p id="market_replay_status" class="plan-replay__status">${replayStatusText}</p>
      </div>
      ${
        mergedPlanMeta.notes
          ? `<div class="plan-panel__section">
              <h3>Plan Notes</h3>
              <p>${mergedPlanMeta.notes}</p>
            </div>`
          : ''
      }
    `;
    if (planPanelEl) {
      if (window.innerWidth <= 1024) {
        planPanelEl.open = false;
      } else {
        planPanelEl.open = true;
      }
    }
    planLogListEl = document.getElementById('plan_log');
    planLogEmptyEl = document.getElementById('plan_log_empty');
    renderPlanLog();
    const autoReplanApplyBtn = document.getElementById('auto_replan_apply');
    if (autoReplanApplyBtn) {
      autoReplanApplyBtn.addEventListener('click', () => adoptPendingReplan({ viaAuto: false }));
    }
    renderAutoReplanBanner();
    if (!loggedInitialPlan) {
      const degraded = isFeedDegraded();
      const initialMessage = degraded
        ? 'Plan loaded — data feed degraded; use cached Polygon bars until live resumes.'
        : 'Plan loaded — follow the checklist before acting.';
      appendPlanLogEntry(initialMessage, Math.floor(Date.now() / 1000), degraded ? 'alert' : 'info');
      loggedInitialPlan = true;
    }
    attachReplayControls();
    attachScenarioControls();
    updatePlanPanelLastPrice(lastPrice);
  };

  const updatePlanPanelLastPrice = (value) => {
    const lastEl = document.getElementById('plan_last_price_value');
    if (!lastEl) return;
    lastEl.textContent = Number.isFinite(value) ? formatPrice(value) : '—';
  };

  const ensureScrollToRealTime = () => {
    if (isReplaying || !followLive) return;
    try {
      suppressFollowDetection = true;
      chart.timeScale().scrollToRealTime();
    } catch (error) {
      suppressFollowDetection = false;
      return;
    }
    window.requestAnimationFrame(() => {
      suppressFollowDetection = false;
    });
  };

  const updateRealtimeBar = (price, payload) => {
    if (!latestCandleData.length || !Number.isFinite(price)) return;
    const volumeFromPayload = Number.isFinite(payload?.volume) ? Number(payload.volume) : null;
    const secondsPerBar = Math.max(lastSecondsPerBar, 60);
    const eventTs = parseEventTimestamp(payload) ?? Math.floor(Date.now() / 1000);
    const bucketTime = Math.floor(eventTs / secondsPerBar) * secondsPerBar;

    const lastIdx = latestCandleData.length - 1;
    const lastBar = latestCandleData[lastIdx];
    if (!lastBar || !Number.isFinite(lastBar?.time)) {
      return;
    }

    if (bucketTime < lastBar.time) {
      // Out-of-order tick; ignore to avoid rewinding the chart.
      return;
    }

    if (bucketTime === lastBar.time) {
      const updated = {
        ...lastBar,
        close: price,
        high: Math.max(Number.isFinite(lastBar.high) ? lastBar.high : price, price),
        low: Math.min(Number.isFinite(lastBar.low) ? lastBar.low : price, price),
      };
      latestCandleData[lastIdx] = updated;
      candleSeries.update(updated);

      const volBar = latestVolumeData[latestVolumeData.length - 1];
      if (volBar) {
        const updatedVol = {
          ...volBar,
          ...(volumeFromPayload !== null ? { value: volumeFromPayload } : {}),
          color: updated.close >= updated.open ? '#22c55e88' : '#ef444488',
        };
        latestVolumeData[latestVolumeData.length - 1] = updatedVol;
        volumeSeries.update(updatedVol);
      }
      ensureScrollToRealTime();
      return;
    }

    // Advance to a new bar bucket.
    const newBar = {
      time: bucketTime,
      open: price,
      high: price,
      low: price,
      close: price,
    };
    latestCandleData.push(newBar);
    candleSeries.update(newBar);

    const newVolumeBar = {
      time: bucketTime,
      value: volumeFromPayload !== null ? volumeFromPayload : 0,
      color: price >= newBar.open ? '#22c55e88' : '#ef444488',
    };
    latestVolumeData.push(newVolumeBar);
    volumeSeries.update(newVolumeBar);
    updateSessionBands(latestCandleData, lastSecondsPerBar);
    ensureScrollToRealTime();
  };

  // Flashing indicator to show activity
  const pulsePlanLogIndicator = () => {
    const el = document.getElementById('plan_log_indicator');
    if (!el) return;
    el.style.opacity = el.style.opacity === '1' ? '.4' : '1';
  };
  setInterval(pulsePlanLogIndicator, 5000);

  // --- Scenario Plans (inline) ---
  const applyPlanResponse = (response, { logMessage } = {}) => {
    if (!response) return;
    pendingAutoReplan = null;
    renderAutoReplanBanner();
    const planBlock = response.plan || {};
    const structured = response.structured_plan || planBlock.structured_plan || null;
    const chartsParams = response.charts_params || (response.charts && response.charts.params) || {};

    const resolveEntry = () => {
      const candidate = planBlock.entry ?? structured?.entry?.level ?? chartsParams.entry;
      const value = toNumber(candidate);
      return Number.isFinite(value) ? value : mergedPlanMeta.entry ?? currentPlan.entry ?? null;
    };
    const resolveStop = () => {
      const candidate = planBlock.stop ?? structured?.stop ?? chartsParams.stop;
      const value = toNumber(candidate);
      return Number.isFinite(value) ? value : mergedPlanMeta.stop ?? currentPlan.stop ?? null;
    };
    const resolveTargets = () => {
      const fromPlan = numericArray(planBlock.targets);
      if (fromPlan.length) return fromPlan;
      const fromStructured = numericArray(structured?.targets);
      if (fromStructured.length) return fromStructured;
      if (typeof chartsParams.tp === 'string') {
        return chartsParams.tp
          .split(',')
          .map((token) => Number(token))
          .filter((value) => Number.isFinite(value));
      }
      return numericArray(currentPlan?.tps);
    };
    const resolveTargetMeta = () => {
      if (Array.isArray(planBlock.target_meta) && planBlock.target_meta.length) return planBlock.target_meta;
      if (Array.isArray(response.target_meta) && response.target_meta.length) return response.target_meta;
      if (structured && Array.isArray(structured.meta) && structured.meta.length) return structured.meta;
      return mergedPlanMeta.target_meta || [];
    };

    const entry = resolveEntry();
    const stop = resolveStop();
    const targets = resolveTargets();
    const targetMeta = resolveTargetMeta();
    const directionToken = (planBlock.direction || structured?.direction || response.bias || mergedPlanMeta.bias || 'long')
      .toString()
      .toLowerCase();
    const styleToken = (planBlock.style || response.style || mergedPlanMeta.style || 'intraday').toString().toLowerCase();
    const runner = planBlock.runner || structured?.runner || mergedPlanMeta.runner || null;
    const strategyId = planBlock.strategy || structured?.strategy || response.strategy_id || mergedPlanMeta.strategy || null;

    const notesCandidate = planBlock.notes || response.notes || structured?.rationale || mergedPlanMeta.notes || null;
    const confidenceValue = toNumber(response.confidence ?? planBlock.confidence ?? structured?.confidence ?? mergedPlanMeta.confidence);
    const rrValue = toNumber(response.rr_to_t1 ?? planBlock.rr_to_t1 ?? mergedPlanMeta.risk_reward);

    basePlan = {
      entry,
      stop,
      tps: targets,
      direction: directionToken,
      strategy: strategyId,
      atr: toNumber(planBlock.atr ?? response.plan?.atr ?? mergedPlanMeta.atr) ?? mergedPlanMeta.atr ?? null,
      notes: notesCandidate,
      tpMeta: targetMeta,
      runner,
    };
    currentPlan = JSON.parse(JSON.stringify(basePlan));

    Object.assign(mergedPlanMeta, {
      symbol: (planBlock.symbol || response.symbol || symbol).toUpperCase(),
      style: styleToken,
      style_display: planBlock.style_display || response.style_display || (styleToken ? styleToken.toUpperCase() : mergedPlanMeta.style_display),
      bias: directionToken,
      entry,
      stop,
      targets,
      target_meta: targetMeta,
      runner,
      confidence: confidenceValue,
      risk_reward: rrValue,
      notes: notesCandidate,
      atr: basePlan.atr,
      expected_move: toNumber(response.expected_move ?? mergedPlanMeta.expected_move),
      horizon_minutes: toNumber(response.horizon_minutes ?? mergedPlanMeta.horizon_minutes),
      warnings: Array.isArray(response.warnings) && response.warnings.length ? response.warnings : planBlock.warnings || [],
      strategy: strategyId,
      confluence: Array.isArray(planBlock.confluence_tags)
        ? planBlock.confluence_tags
        : Array.isArray(response.confluence_tags)
          ? response.confluence_tags
          : mergedPlanMeta.confluence,
      confidence_factors: Array.isArray(planBlock.confidence_factors)
        ? planBlock.confidence_factors
        : Array.isArray(response.confidence_factors)
          ? response.confidence_factors
          : mergedPlanMeta.confidence_factors,
    });

    const layersCandidate = planBlock.plan_layers || response.plan_layers || null;
    planLayers = layersCandidate;
    planLayersMeta = (layersCandidate && layersCandidate.meta) || {};
    planZones = Array.isArray(layersCandidate?.zones) ? layersCandidate.zones : [];
    mergedPlanMeta.quotes_notice = planLayersMeta.quotes_notice || null;
    allKeyLevels = collectInitialLevels();
    mergedPlanMeta.key_levels = planBlock.key_levels || response.key_levels || mergedPlanMeta.key_levels;
    showAllLevels = params.get('supportingLevels') !== '0';
    setLevelGroupsFromMeta(planLayersMeta, allKeyLevels);
    const refreshSupporting = dedupeLevels(parseNamedLevels(params.get('levels')));
    if (refreshSupporting.length) {
      levelGroups.supplemental = dedupeLevels([
        ...refreshSupporting,
        ...levelGroups.supplemental,
      ]);
    }
    updateLevelVisibility();
    setOrDeleteParam('supportingLevels', showAllLevels ? '1' : '0');

    setOrDeleteParam('style', styleToken);
    setOrDeleteParam('direction', directionToken);
    if (Number.isFinite(entry)) setOrDeleteParam('entry', entry.toFixed(2)); else params.delete('entry');
    if (Number.isFinite(stop)) setOrDeleteParam('stop', stop.toFixed(2)); else params.delete('stop');
    if (targets.length) {
      setOrDeleteParam('tp', targets.map((value) => (Number.isFinite(value) ? value.toFixed(2) : value)).join(','));
    } else {
      params.delete('tp');
    }

    planLogEntries.length = 0;
    const banner = planBlock.session_state?.banner;
    if (banner) {
      const bannerAsOf = planBlock.session_state?.as_of;
      const bannerTs = bannerAsOf ? Math.floor(new Date(bannerAsOf).getTime() / 1000) : Math.floor(Date.now() / 1000);
      appendPlanLogEntry(banner, bannerTs, 'info');
    }
    const rationale = structured?.rationale || notesCandidate;
    if (rationale) {
      appendPlanLogEntry(rationale, Math.floor(Date.now() / 1000), 'info');
    }
    if (logMessage) {
      appendPlanLogEntry(logMessage, Math.floor(Date.now() / 1000), 'info');
    }

    renderPlanPanel(lastKnownPrice);
    applyPlanStatus('intact', notesCandidate || latestPlanNote, planBlock.next_step || 'hold_plan', rrValue);

    const nextPlanId = planBlock.plan_id || response.plan_id || currentPlanId;
    const nextPlanVersion = planBlock.version || response.version || currentPlanVersion;
    return { planId: nextPlanId, planVersion: nextPlanVersion };
  };

  const scenarioConfig = { style: 'intraday', baseStyle: 'intraday', busy: false };
  scenarioConfig.baseStyle = (mergedPlanMeta.style || '').toLowerCase() || scenarioConfig.baseStyle;
  scenarioConfig.style = scenarioConfig.baseStyle;
  let scenarioOverlayData = null; // { entry, stop, tps[], label, plan_id, plan_version, levels, zones }
  const scenarioLineIds = new Set();

  const fetchPlanIdea = async (planId) => {
    const resp = await fetch(`${baseUrl}/idea/${encodeURIComponent(planId)}`, { cache: 'no-store' });
    if (!resp.ok) {
      throw new Error(`Snapshot unavailable (${resp.status})`);
    }
    return resp.json();
  };

  const applyPlanSnapshot = async (payload, { logMessage, fallbackPlanId, fallbackPlanVersion } = {}) => {
    const result = applyPlanResponse(payload, { logMessage }) || {};
    const appliedPlanId = result.planId || fallbackPlanId || currentPlanId;
    const appliedPlanVersion = result.planVersion || fallbackPlanVersion || currentPlanVersion;
    setCurrentPlanId(appliedPlanId, appliedPlanVersion);
    await loadTickerInsights(appliedPlanId);
    forceAutoFocus = true;
    await fetchBars();
    return { planId: appliedPlanId, planVersion: appliedPlanVersion };
  };

  const setScenarioStatus = (message, isError = false) => {
    const el = document.getElementById('scenario_status');
    if (el) {
      el.textContent = message || '';
      el.style.color = isError ? '#fca5a5' : '';
    }
  };

  const extractScenarioFromResponse = (data) => {
    try {
      const plan = data?.plan || data || {};
      const entry = Number.isFinite(plan.entry) ? plan.entry : Number(plan?.structured_plan?.entry?.level);
      const stop = Number.isFinite(plan.stop) ? plan.stop : Number(plan?.structured_plan?.stop);
      const tps = Array.isArray(plan.targets) ? plan.targets : (plan?.structured_plan?.targets || []);
      const plan_id = String(plan.plan_id || '').trim();
      const plan_version = Number.isFinite(Number(plan.version)) ? String(plan.version) : null;
      return { entry, stop, tps, plan_id, plan_version };
    } catch {
      return { entry: null, stop: null, tps: [], plan_id: null, plan_version: null };
    }
  };

  const removeScenarioLines = () => {
    scenarioLineIds.forEach((id) => {
      const line = priceLineMap.get(id);
      if (line) {
        candleSeries.removePriceLine(line);
        priceLineMap.delete(id);
      }
    });
    scenarioLineIds.clear();
  };

  const applyScenarioOverlayLines = (activeIds) => {
    removeScenarioLines();
    if (!scenarioOverlayData) return;
    const addLine = (id, options) => {
      scenarioLineIds.add(id);
      const line = setPriceLine(id, options);
      if (line && activeIds) activeIds.add(id);
    };
    if (Number.isFinite(scenarioOverlayData.entry)) {
      addLine('scenario:entry', {
        price: scenarioOverlayData.entry,
        color: '#60a5fa',
        title: (scenarioOverlayData.label || 'Scenario') + ' Entry',
        lineWidth: 1,
        lineStyle: LightweightCharts.LineStyle.Dotted,
      });
    }
    if (Number.isFinite(scenarioOverlayData.stop)) {
      addLine('scenario:stop', {
        price: scenarioOverlayData.stop,
        color: '#fb7185',
        title: (scenarioOverlayData.label || 'Scenario') + ' Stop',
        lineWidth: 1,
        lineStyle: LightweightCharts.LineStyle.Dashed,
      });
    }
    (scenarioOverlayData.tps || []).forEach((tp, idx) => {
      if (!Number.isFinite(tp)) return;
      addLine(`scenario:tp:${idx + 1}`, {
        price: tp,
        color: '#93c5fd',
        title: (scenarioOverlayData.label || 'Scenario') + ` TP${idx + 1}`,
        lineWidth: 1,
        lineStyle: LightweightCharts.LineStyle.Dashed,
      });
    });
    (scenarioOverlayData.levels || []).forEach((level, idx) => {
      const price = Number(level?.price);
      if (!Number.isFinite(price)) return;
      const label = level?.label ? `${level.label} (Scenario)` : `Scenario Level ${idx + 1}`;
      addLine(`scenario:level:${idx}:${price.toFixed(4)}`, {
        price,
        color: '#a5b4fc',
        title: label,
        lineWidth: 1,
        lineStyle: LightweightCharts.LineStyle.Dotted,
      });
    });
    (scenarioOverlayData.zones || []).forEach((zone, idx) => {
      const high = Number(zone?.high);
      const low = Number(zone?.low);
      if (Number.isFinite(high)) {
        addLine(`scenario:zone:${idx}:high`, {
          price: high,
          color: '#bfdbfe',
          title: (zone?.label || 'Scenario Zone') + ' High',
          lineWidth: 1,
          lineStyle: LightweightCharts.LineStyle.Dashed,
        });
      }
      if (Number.isFinite(low)) {
        addLine(`scenario:zone:${idx}:low`, {
          price: low,
          color: '#bfdbfe',
          title: (zone?.label || 'Scenario Zone') + ' Low',
          lineWidth: 1,
          lineStyle: LightweightCharts.LineStyle.Dashed,
        });
      }
    });
  };

  const scenarioGenerate = async () => {
    if (scenarioConfig.busy) return;
    scenarioConfig.busy = true;
    setScenarioStatus('Generating scenario…');
    try {
      const response = await fetch(`${baseUrl}/gpt/plan`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbol, style: scenarioConfig.style }),
      });
      if (!response.ok) throw new Error(`/gpt/plan failed (${response.status})`);
      const data = await response.json();
      const extracted = extractScenarioFromResponse(data);
      let scenarioLayers = null;
      if (extracted.plan_id) {
        try {
          const layersResp = await fetch(
            `${baseUrl}/api/v1/gpt/chart-layers?plan_id=${encodeURIComponent(extracted.plan_id)}`,
            { cache: 'no-store' },
          );
          if (layersResp.ok) {
            scenarioLayers = await layersResp.json();
          }
        } catch (layerErr) {
          console.warn('Scenario layer fetch failed', layerErr);
        }
      }
      const layerLevels = Array.isArray(scenarioLayers?.levels)
        ? scenarioLayers.levels
            .map((item) => ({ price: Number(item?.price), label: item?.label || null }))
            .filter((item) => Number.isFinite(item.price))
        : [];
      const layerZones = Array.isArray(scenarioLayers?.zones)
        ? scenarioLayers.zones
            .map((item) => ({
              high: Number(item?.high),
              low: Number(item?.low),
              label: item?.label || item?.kind || null,
            }))
            .filter((item) => Number.isFinite(item.high) || Number.isFinite(item.low))
        : [];
      const styleLabel = `${scenarioConfig.style.charAt(0).toUpperCase()}${scenarioConfig.style.slice(1)}`;
      scenarioOverlayData = {
        entry: Number.isFinite(extracted.entry) ? extracted.entry : null,
        stop: Number.isFinite(extracted.stop) ? extracted.stop : null,
        tps: (extracted.tps || []).filter((v) => Number.isFinite(v)),
        label: `Scenario · ${styleLabel}`,
        plan_id: extracted.plan_id,
        plan_version: extracted.plan_version,
        levels: layerLevels,
        zones: layerZones,
        snapshot: data,
      };
      setScenarioStatus('Scenario overlay active — use Adopt or Clear');
      applyScenarioOverlayLines();
      const adoptBtn = document.getElementById('scenario_adopt');
      const clearBtn = document.getElementById('scenario_clear');
      if (adoptBtn) adoptBtn.disabled = false;
      if (clearBtn) clearBtn.disabled = false;
      chart.priceScale('right').applyOptions({ autoScale: true });
    } catch (err) {
      setScenarioStatus(err?.message || String(err) || 'Scenario generation failed', true);
      const adoptBtn = document.getElementById('scenario_adopt');
      const clearBtn = document.getElementById('scenario_clear');
      if (adoptBtn) adoptBtn.disabled = true;
      if (clearBtn) clearBtn.disabled = true;
      scenarioOverlayData = null;
      removeScenarioLines();
    } finally {
      scenarioConfig.busy = false;
    }
  };

  const adoptScenario = async (adoptBtn, clearBtn, setActiveFn) => {
    if (!scenarioOverlayData) return;
    pauseFollowLive();
    setScenarioStatus('Adopting scenario…');
    try {
      let payload;
      const overlayPlanId = scenarioOverlayData.plan_id ? String(scenarioOverlayData.plan_id).trim() : null;
      if (overlayPlanId) {
        try {
          payload = await fetchPlanIdea(overlayPlanId);
        } catch (ideaError) {
          if (scenarioOverlayData.snapshot) {
            payload = scenarioOverlayData.snapshot;
          } else {
            throw ideaError;
          }
        }
      } else if (scenarioOverlayData.snapshot) {
        payload = scenarioOverlayData.snapshot;
      } else {
        throw new Error('Scenario snapshot unavailable');
      }
      const logMessage = scenarioOverlayData.label ? `${scenarioOverlayData.label} adopted.` : 'Scenario adopted.';
      const applied = await applyPlanSnapshot(payload, {
        logMessage,
        fallbackPlanId: overlayPlanId,
        fallbackPlanVersion: scenarioOverlayData.plan_version,
      });
      scenarioConfig.baseStyle = (mergedPlanMeta.style || '').toLowerCase();
      scenarioConfig.style = scenarioConfig.baseStyle;
      scenarioOverlayData = null;
      removeScenarioLines();
      if (adoptBtn) adoptBtn.disabled = true;
      if (clearBtn) clearBtn.disabled = true;
      if (typeof setActiveFn === 'function') setActiveFn(scenarioConfig.baseStyle);
      chart.priceScale('right').applyOptions({ autoScale: true });
      showToast('Scenario adopted — live updates active.');
      setScenarioStatus('Scenario adopted — live updates active.');
    } catch (error) {
      console.error('Scenario adoption failed', error);
      setScenarioStatus(error instanceof Error ? `Adoption failed: ${error.message}` : 'Adoption failed.', true);
      showToast('Scenario adoption failed', { tone: 'warn' });
    }
  };

  const attachScenarioControls = () => {
    const container = document.getElementById('scenario_controls');
    if (!container) return;
    // Add Adopt and Clear buttons inline
    const adoptBtn = document.createElement('button');
    adoptBtn.id = 'scenario_adopt';
    adoptBtn.type = 'button';
    adoptBtn.className = 'plan-replay__button';
    adoptBtn.textContent = 'Adopt This Scenario';
    adoptBtn.disabled = true;
    const clearBtn = document.createElement('button');
    clearBtn.id = 'scenario_clear';
    clearBtn.type = 'button';
    clearBtn.className = 'plan-replay__button';
    clearBtn.textContent = 'Clear Overlay';
    clearBtn.disabled = true;
    const buttons = Array.from(container.querySelectorAll('button[data-scenario-style]'));
    const setActive = (style) => {
      buttons.forEach((btn) => {
        if (btn.getAttribute('data-scenario-style') === style) {
          btn.classList.add('plan-replay__button--active');
        } else {
          btn.classList.remove('plan-replay__button--active');
        }
      });
    };
    const baseStyle = (mergedPlanMeta.style || '').toLowerCase();
    scenarioConfig.baseStyle = baseStyle || scenarioConfig.baseStyle;
    const overlayActive = !!scenarioOverlayData && scenarioOverlayData.plan_id;
    if (!overlayActive) {
      scenarioConfig.style = scenarioConfig.baseStyle;
    }
    let matchedInitial = false;
    buttons.forEach((btn) => {
      btn.addEventListener('click', () => {
        const next = btn.getAttribute('data-scenario-style');
        if (!next) return;
        scenarioConfig.style = next;
        setActive(next);
        if (next === scenarioConfig.baseStyle) {
          scenarioOverlayData = null;
          removeScenarioLines();
          setScenarioStatus('Using live plan (no scenario overlay).');
          adoptBtn.disabled = true;
          clearBtn.disabled = true;
          chart.priceScale('right').applyOptions({ autoScale: true });
        } else {
          scenarioGenerate();
        }
      });
      if (!matchedInitial && btn.getAttribute('data-scenario-style') === scenarioConfig.baseStyle) {
        matchedInitial = true;
        setActive(scenarioConfig.baseStyle);
      }
    });
    if (!matchedInitial && buttons.length) {
      const fallbackStyle = buttons[0].getAttribute('data-scenario-style');
      if (fallbackStyle) {
        scenarioConfig.baseStyle = fallbackStyle;
        if (!overlayActive) {
          scenarioConfig.style = fallbackStyle;
        }
        setActive(fallbackStyle);
      }
    }
    container.appendChild(adoptBtn);
    container.appendChild(clearBtn);
    const autoRearmBtn = document.createElement('button');
    autoRearmBtn.id = 'auto_rearm_toggle';
    autoRearmBtn.type = 'button';
    autoRearmBtn.className = 'plan-replay__button';
    autoRearmBtn.setAttribute('aria-pressed', autoRearmEnabled ? 'true' : 'false');
    container.appendChild(autoRearmBtn);
    adoptBtn.addEventListener('click', () => {
      adoptScenario(adoptBtn, clearBtn, setActive);
    });
    clearBtn.addEventListener('click', () => {
      scenarioOverlayData = null;
      setScenarioStatus('Scenario cleared.');
      removeScenarioLines();
      adoptBtn.disabled = true;
      clearBtn.disabled = true;
      chart.priceScale('right').applyOptions({ autoScale: true });
    });
    autoRearmBtn.addEventListener('click', () => {
      setAutoRearmEnabled(!autoRearmEnabled);
    });
    const styleToHighlight = overlayActive ? scenarioConfig.style : scenarioConfig.baseStyle;
    setActive(styleToHighlight);
    if (overlayActive) {
      adoptBtn.disabled = false;
      clearBtn.disabled = false;
      setScenarioStatus('Scenario overlay active — use Adopt or Clear');
    } else {
      adoptBtn.disabled = true;
      clearBtn.disabled = true;
      setScenarioStatus('Select a style to generate a scenario overlay.');
    }
    updateAutoRearmToggle();
  };

  const loadTickerInsights = async (planId = currentPlanId) => {
    if (!planId || !headerSymbolMetaEl) return;
    try {
      const resp = await fetch(`${baseUrl}/idea/${encodeURIComponent(planId)}`, { cache: 'no-store' });
      if (!resp.ok) return;
      const data = await resp.json();
      const parts = [];
      let highlight = false;
      const earnings = data?.earnings || null;
      if (earnings) {
        const nextAt = earnings.next_earnings_at ? new Date(earnings.next_earnings_at) : null;
        let dte = Number.isFinite(Number(earnings.dte_to_earnings)) ? Number(earnings.dte_to_earnings) : null;
        if (dte === null && nextAt) {
          dte = Math.round((nextAt.getTime() - Date.now()) / 86400000);
        }
        const slot = (earnings.pre_or_post || '').toUpperCase();
        const formatter = new Intl.DateTimeFormat('en-US', { month: 'short', day: 'numeric' });
        const dateLabel = nextAt ? formatter.format(nextAt) : '';
        let windowLabel = '';
        if (dte !== null) {
          if (dte === 0) windowLabel = 'today';
          else if (dte > 0) windowLabel = `in ${dte}d`;
          else windowLabel = `${Math.abs(dte)}d ago`;
        }
        const earningsPieces = ['Earnings'];
        if (windowLabel) earningsPieces.push(windowLabel);
        if (dateLabel) {
          const slotLabel = slot && slot !== 'NONE' ? ` ${slot}` : '';
          earningsPieces.push(`(${dateLabel}${slotLabel})`);
        }
        parts.push(earningsPieces.join(' '));
        if (earnings.earnings_flag === 'today' || earnings.earnings_flag === 'near') {
          highlight = true;
        }
      }
      const sentiment = data?.sentiment || null;
      if (sentiment && typeof sentiment.symbol_sentiment === 'number') {
        const score = Number(sentiment.symbol_sentiment);
        const scoreLabel = `${score >= 0 ? '+' : ''}${(score * 100).toFixed(0)}bps`;
        const riskLabel = sentiment.headline_risk && sentiment.headline_risk !== 'normal' ? sentiment.headline_risk : null;
        const sentimentPieces = [`Sentiment ${scoreLabel}`];
        if (riskLabel) sentimentPieces.push(riskLabel);
        parts.push(sentimentPieces.join(' · '));
      }
      if (parts.length) {
        setSymbolMeta(parts.join(' • '), highlight);
      } else {
        setSymbolMeta('');
      }
    } catch (error) {
      console.warn('ticker insight fetch failed', error);
    }
  };

  const setReplayStatusMessage = (message) => {
    replayStatusMessage = message || '';
    if (replayStatusEl) {
      replayStatusEl.textContent = replayStatusMessage;
    }
  };

  const syncReplayControls = () => {
    if (replayMinutesInput) {
      if (!isReplaying) {
        replayMinutesInput.value = clampReplayMinutes(replayConfig.minutes);
      }
      replayMinutesInput.disabled = isReplaying;
    }
    if (replayStartButton) replayStartButton.disabled = isReplaying;
    if (replayStopButton) replayStopButton.disabled = !isReplaying;
    if (replayStatusEl) {
      replayStatusEl.textContent = replayStatusMessage;
    }
  };

  const fetchReplayBars = async (minutes) => {
    const minutesClamped = clampReplayMinutes(minutes);
    const resolutionToken = currentResolution || '1';
    const secondsPerBar = Math.max(resolutionToSeconds(resolutionToken), 60);
    const nowSec = Math.floor(Date.now() / 1000);
    const minBarCount = Math.max(Math.ceil((minutesClamped * 60) / secondsPerBar), 1);
    const targetBarCount = Math.max(minBarCount, 2);
    const spanSeconds = Math.max(secondsPerBar * (targetBarCount + 1), secondsPerBar * 2);
    const from = nowSec - spanSeconds;
    const qs = new URLSearchParams({
      symbol,
      resolution: resolutionToken,
      from: String(from),
      to: String(nowSec),
    });
    const response = await fetch(`${baseUrl}/tv-api/bars?${qs.toString()}`, { cache: 'no-store' });
    if (!response.ok) {
      throw new Error(`Replay data request failed (${response.status})`);
    }
    const payload = await response.json();
    if (payload.s !== 'ok' || !Array.isArray(payload.t) || !payload.t.length) {
      return [];
    }
    const bars = payload.t.map((time, idx) => ({
      time: payload.t[idx],
      open: payload.o[idx],
      high: payload.h[idx],
      low: payload.l[idx],
      close: payload.c[idx],
      volume: Array.isArray(payload.v) ? payload.v[idx] || 0 : 0,
    }));
    bars.sort((a, b) => a.time - b.time);
    const cutoff = nowSec - minutesClamped * 60;
    const recent = bars.filter((bar) => bar.time >= cutoff);
    if (recent.length >= 2) {
      return recent.slice(-targetBarCount);
    }
    if (recent.length === 1) {
      const lastIndex = bars.findIndex((bar) => bar.time === recent[0].time);
      if (lastIndex > 0) {
        return bars.slice(Math.max(0, lastIndex - (targetBarCount - 1)), lastIndex + 1);
      }
      return recent;
    }
    return bars.slice(-targetBarCount);
  };

  const stopMarketReplay = ({
    restoreChart = true,
    resumeStream = true,
    message = '',
    skipFetch = false,
  } = {}) => {
    if (replayTimer) {
      window.clearTimeout(replayTimer);
      replayTimer = null;
    }
    replayFetchToken += 1;
    const wasReplaying = isReplaying;
    isReplaying = false;
    replayQueue = [];
    replayIndex = 0;

    if (restoreChart && replaySavedCandleData.length) {
      const restoredCandles = replaySavedCandleData.map((item) => ({ ...item }));
      const restoredVolume = replaySavedVolumeData.map((item) => ({ ...item }));
      candleSeries.setData(restoredCandles);
      volumeSeries.setData(restoredVolume);
      latestCandleData = restoredCandles.map((item) => ({ ...item }));
      latestVolumeData = restoredVolume.map((item) => ({ ...item }));
      if (replaySavedVisibleRange) {
        try {
          chart.timeScale().setVisibleRange(replaySavedVisibleRange);
        } catch {
          chart.timeScale().fitContent();
        }
      } else {
        chart.timeScale().fitContent();
        if (centerTimeToken) {
          applyTimeCentering({ force: true });
        }
      }
    } else if (restoreChart) {
      const clonedCandles = latestCandleData.map((item) => ({ ...item }));
      const clonedVolume = latestVolumeData.map((item) => ({ ...item }));
      candleSeries.setData(clonedCandles);
      volumeSeries.setData(clonedVolume);
      latestCandleData = clonedCandles.map((item) => ({ ...item }));
      latestVolumeData = clonedVolume.map((item) => ({ ...item }));
      chart.timeScale().fitContent();
      if (centerTimeToken) {
        applyTimeCentering({ force: true });
      }
    }
    if (focusToken === 'plan') {
      applyPlanPriceFocus(currentPlan, { force: true });
    } else {
      chart.priceScale('right').applyOptions({ autoScale: true });
    }
    const restoredLast = latestCandleData.length ? latestCandleData[latestCandleData.length - 1].close : null;
    if (Number.isFinite(restoredLast)) {
      lastKnownPrice = restoredLast;
      updateHeaderPricing(restoredLast);
      updatePlanPanelLastPrice(restoredLast);
    }
    replaySavedCandleData = [];
    replaySavedVolumeData = [];
    replaySavedVisibleRange = null;

    updateSessionBands(latestCandleData, lastSecondsPerBar);
    renderSessionBands();

    if (replayPrevPhase || replayPrevNote) {
      applyMarketStatus(replayPrevPhase || 'closed', replayPrevNote || undefined);
    }
    replayPrevPhase = null;
    replayPrevNote = null;

    if (resumeStream && replayHadStream) {
      connectStream();
    }
    replayHadStream = false;

    if (message) {
      setReplayStatusMessage(message);
    } else if (wasReplaying) {
      setReplayStatusMessage('');
    } else {
      setReplayStatusMessage(replayStatusMessage);
    }
    syncReplayControls();

    if (!skipFetch) {
      fetchBars();
    }
  };

  const playNextReplayBar = () => {
    if (!isReplaying) return;
    if (replayIndex >= replayQueue.length) {
      stopMarketReplay({
        restoreChart: true,
        resumeStream: true,
        message: 'Replay complete. Resuming live stream…',
      });
      return;
    }
    const frame = replayQueue[replayIndex];
    const candle = frame.candle;
    candleSeries.update(candle);
    latestCandleData.push({ ...candle });

    const volumeValue = Number.isFinite(frame.volume) ? frame.volume : 0;
    const volumeBar = {
      time: candle.time,
      value: volumeValue,
      color: candle.close >= candle.open ? '#22c55e88' : '#ef444488',
    };
    latestVolumeData.push({ ...volumeBar });
    volumeSeries.update(volumeBar);
    updateSessionBands(latestCandleData, lastSecondsPerBar);

    lastKnownPrice = candle.close;
    updateHeaderPricing(lastKnownPrice);
    updatePlanPanelLastPrice(lastKnownPrice);
    const total = replayQueue.length;
    replayIndex += 1;
    const progress = Math.min(replayIndex, total);
    setReplayStatusMessage(`Playing ${progress}/${total}`);
    syncReplayControls();
    replayTimer = window.setTimeout(playNextReplayBar, REPLAY_STEP_MS);
  };

  const startMarketReplay = async () => {
    if (isReplaying) return;
    const minutesRaw = replayMinutesInput ? replayMinutesInput.value : replayConfig.minutes;
    const minutes = clampReplayMinutes(minutesRaw);
    replayConfig.minutes = minutes;
    if (replayMinutesInput) replayMinutesInput.value = minutes;

    replayFetchToken += 1;
    const token = replayFetchToken;
    isReplaying = true;
    replayQueue = [];
    replayIndex = 0;
    replayPrevPhase = currentMarketPhase;
    replayPrevNote = latestMarketNote;
    replayHadStream = Boolean(streamSource);
    if (streamSource) {
      try {
        streamSource.close();
      } catch {
        // ignore
      }
      streamSource = null;
    }
    const tfLabel = (activeTimeframe && activeTimeframe.label) || currentResolution || '1m';
    applyMarketStatus('replay', `Replaying last ${minutes} minutes using ${tfLabel} bars.`);
    setReplayStatusMessage(`Preparing ${tfLabel} replay…`);
    syncReplayControls();

    try {
      const bars = await fetchReplayBars(minutes);
      if (!isReplaying || token !== replayFetchToken) {
        return;
      }
      if (!Array.isArray(bars) || bars.length === 0) {
        stopMarketReplay({
          restoreChart: true,
          resumeStream: true,
          message: 'No intraday data available for that window.',
          skipFetch: true,
        });
        return;
      }

      replaySavedCandleData = latestCandleData.map((item) => ({ ...item }));
      replaySavedVolumeData = latestVolumeData.map((item) => ({ ...item }));
      try {
        replaySavedVisibleRange = chart.timeScale().getVisibleRange();
      } catch {
        replaySavedVisibleRange = null;
      }

      const baselineCandles = replaySavedCandleData.map((item) => ({ ...item }));
      const baselineVolume = replaySavedVolumeData.map((item) => ({ ...item }));
      const replayFrames = bars.map((bar) => ({
        candle: {
          time: bar.time,
          open: bar.open,
          high: bar.high,
          low: bar.low,
          close: bar.close,
        },
        volume: Number.isFinite(bar.volume) ? bar.volume : 0,
      }));
      if (!replayFrames.length) {
        throw new Error('Replay preparation failed (no frames)');
      }

      const baseLength = Math.max(baselineCandles.length - replayFrames.length, 0);
      const initialCandles = baseLength ? baselineCandles.slice(0, baseLength) : [];
      const initialVolume = baseLength ? baselineVolume.slice(0, baseLength) : [];

      candleSeries.setData(initialCandles);
      volumeSeries.setData(initialVolume);
      latestCandleData = initialCandles.map((item) => ({ ...item }));
      latestVolumeData = initialVolume.map((item) => ({ ...item }));
      updateSessionBands(latestCandleData, lastSecondsPerBar);

      const priorClose = latestCandleData.length
        ? latestCandleData[latestCandleData.length - 1].close
        : replayFrames[0].candle.open;
      if (Number.isFinite(priorClose)) {
        lastKnownPrice = priorClose;
        updateHeaderPricing(lastKnownPrice);
        updatePlanPanelLastPrice(lastKnownPrice);
      }

      replayQueue = replayFrames;
      try {
        if (replaySavedVisibleRange) {
          chart.timeScale().setVisibleRange(replaySavedVisibleRange);
        } else if (initialCandles.length) {
          chart.timeScale().setVisibleRange({
            from: initialCandles[0].time,
            to: replayFrames[replayFrames.length - 1].candle.time,
          });
        } else {
          chart.timeScale().fitContent();
        }
      } catch {
        chart.timeScale().fitContent();
      }

      replayIndex = 0;
      setReplayStatusMessage(`Playing 0/${replayQueue.length}`);
      syncReplayControls();
      playNextReplayBar();
    } catch (err) {
      console.error('Market replay failed', err);
      stopMarketReplay({
        restoreChart: true,
        resumeStream: true,
        message: `Replay failed: ${err.message || err}`,
      });
    }
  };

  const attachReplayControls = () => {
    replayStatusEl = document.getElementById('market_replay_status');
    replayStartButton = document.getElementById('market_replay_start');
    replayStopButton = document.getElementById('market_replay_stop');
    replayMinutesInput = document.getElementById('market_replay_minutes');

    if (replayMinutesInput) {
      replayMinutesInput.value = clampReplayMinutes(replayConfig.minutes);
      replayMinutesInput.addEventListener('change', () => {
        const updated = clampReplayMinutes(replayMinutesInput.value);
        replayConfig.minutes = updated;
        replayMinutesInput.value = updated;
      });
      replayMinutesInput.addEventListener('input', () => {
        replayConfig.minutes = clampReplayMinutes(replayMinutesInput.value);
      });
    }

    if (replayStartButton) {
      replayStartButton.addEventListener('click', (event) => {
        event.preventDefault();
        startMarketReplay();
      });
    }
    if (replayStopButton) {
      replayStopButton.addEventListener('click', (event) => {
        event.preventDefault();
        if (isReplaying) {
          stopMarketReplay({
            restoreChart: true,
            resumeStream: true,
            message: 'Replay stopped. Resuming live stream…',
          });
        }
      });
    }
    syncReplayControls();
  };

  const updateHeaderPricing = (lastPrice) => {
    if (headerLastPriceEl) {
      headerLastPriceEl.textContent = Number.isFinite(lastPrice) ? `Last: ${formatPrice(lastPrice)}` : '';
    }
  };

  const rangeTokenRaw = (params.get('range') || '').toLowerCase();

  const resolveSpanSeconds = () => {
    const base = resolutionToSeconds(currentResolution) * 600;
    if (!rangeTokenRaw) return base;

    if (/^\d+$/.test(rangeTokenRaw)) {
      const barsCount = parseInt(rangeTokenRaw, 10);
      if (Number.isFinite(barsCount) && barsCount > 0) {
        return resolutionToSeconds(currentResolution) * barsCount;
      }
    }

    const match = rangeTokenRaw.match(/^(\d+)([dwmy])$/);
    if (match) {
      const value = parseInt(match[1], 10);
      const unit = match[2];
      const UNIT_SECONDS = { d: 86400, w: 604800, m: 2629800, y: 31557600 };
      const mult = UNIT_SECONDS[unit];
      if (mult && Number.isFinite(value) && value > 0) {
        return value * mult;
      }
    }

    if (rangeTokenRaw === 'fit') {
      return base;
    }

    return base;
  };

  const fetchBars = async () => {
    if (isReplaying) return;
    const token = ++fetchToken;
    const resetFocusFlags = forceAutoFocus || followLive;
    if (resetFocusFlags) {
      hasAppliedFocusRange = false;
      hasAppliedTimeCenter = false;
      hasAppliedTimeWindow = false;
    }
    try {
      const now = Math.floor(Date.now() / 1000);
      const minSpan = resolutionToSeconds(currentResolution) * 200;
      const maxSpan = 60 * 60 * 24 * 365 * 2;
      const span = Math.min(Math.max(resolveSpanSeconds(), minSpan), maxSpan);
      const from = now - span;
      const qs = new URLSearchParams({
        symbol,
        resolution: currentResolution,
        from: String(from),
        to: String(now),
      }).toString();
      const response = await fetch(`${baseUrl}/tv-api/bars?${qs}`);
      if (!response.ok) throw new Error(`bars request failed (${response.status})`);
      const payload = await response.json();
      if (payload.s !== 'ok' || !payload.t?.length) throw new Error('No data');
      if (token !== fetchToken || isReplaying) {
        return;
      }

      const bars = payload.t.map((time, idx) => ({
        time: payload.t[idx],
        open: payload.o[idx],
        high: payload.h[idx],
        low: payload.l[idx],
        close: payload.c[idx],
        volume: payload.v[idx] || 0,
      }));

      if (bars.length) {
        const lastBar = bars[bars.length - 1];
        if (Number.isFinite(lastBar?.time)) {
          setDataLastUpdate(lastBar.time);
        }
      }

      const planForFrame = clonePlan();

      const candleData = bars.map((bar) => ({
        time: bar.time,
        open: bar.open,
        high: bar.high,
        low: bar.low,
        close: bar.close,
      }));
      candleSeries.setData(candleData);
      latestCandleData = candleData.map((bar) => ({ ...bar }));

      const volumeData = bars.map((bar) => ({
        time: bar.time,
        value: bar.volume,
        color: bar.close >= bar.open ? '#22c55e88' : '#ef444488',
      }));
      volumeSeries.setData(volumeData);
      latestVolumeData = volumeData.map((bar) => ({ ...bar }));

      const secondsPerBar = Math.max(resolutionToSeconds(currentResolution), 60);
      lastSecondsPerBar = secondsPerBar;
      updateSessionBands(candleData, secondsPerBar);

      const lastPrice = bars[bars.length - 1]?.close ?? null;
      lastKnownPrice = lastPrice;
      updateHeaderPricing(lastPrice);
      renderPlanPanel(lastPrice);

      let scaleMultiplier = 1;
      if (scalePlanToken !== 'off') {
        if (scalePlanToken === 'auto') {
          if (Number.isFinite(planForFrame.entry) && Number.isFinite(lastPrice) && planForFrame.entry && lastPrice) {
            const ratio = lastPrice / planForFrame.entry;
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
        if (Number.isFinite(planForFrame.entry)) planForFrame.entry *= scaleMultiplier;
        if (Number.isFinite(planForFrame.stop)) planForFrame.stop *= scaleMultiplier;
        planForFrame.tps = planForFrame.tps.map((tp) => (Number.isFinite(tp) ? tp * scaleMultiplier : tp));
        if (planForFrame.runner && Number.isFinite(planForFrame.runner.anchor)) {
          planForFrame.runner.anchor *= scaleMultiplier;
        }
      }

      currentPlan = planForFrame;

      const overlayValues = [
        planForFrame.entry,
        planForFrame.stop,
        ...planForFrame.tps,
        ...keyLevels.map((lvl) => lvl.price),
      ].filter((value) => Number.isFinite(value));

      const vwapRequested = (params.get('vwap') || '').toLowerCase() !== 'false';
      if (vwapRequested && !vwapSeries) {
        vwapSeries = chart.addLineSeries({
          lineWidth: 2,
          color: '#f8fafc',
          title: 'VWAP',
        });
      }
      let lastVwap = null;
      if (vwapSeries) {
        const values = [];
        let cumulativePV = 0;
        let cumulativeVol = 0;
        let currentSession = null;
        for (let i = 0; i < candleData.length; i += 1) {
          const bar = candleData[i];
          const vol = volumeData[i]?.value || 0;
          const typical = (bar.high + bar.low + bar.close) / 3;
          const sessionKey = new Date(bar.time * 1000).toISOString().slice(0, 10);
          if (sessionKey !== currentSession) {
            currentSession = sessionKey;
            cumulativePV = 0;
            cumulativeVol = 0;
          }
          cumulativePV += typical * vol;
          cumulativeVol += vol;
          const vwapValue = cumulativeVol > 0 ? cumulativePV / cumulativeVol : typical;
          if (Number.isFinite(vwapValue)) {
            lastVwap = vwapValue;
          }
          values.push({ time: bar.time, value: vwapValue });
        }
        vwapSeries.setData(values);
      }
      if (Number.isFinite(lastVwap)) {
        overlayValues.push(lastVwap);
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

      const activeLineIds = new Set();
      const registerLine = (id, options) => {
        const line = setPriceLine(id, options);
        if (line) {
          activeLineIds.add(id);
        }
      };

      registerLine('entry', {
        price: planForFrame.entry,
        color: '#facc15',
        title: 'Entry',
        lineWidth: 2,
        lineStyle: LightweightCharts.LineStyle.Solid,
      });

      registerLine('stop', {
        price: planForFrame.stop,
        color: '#ef4444',
        title: 'Stop',
        lineWidth: 2,
        lineStyle: LightweightCharts.LineStyle.Solid,
      });

      planForFrame.tps.forEach((tp, idx) => {
        if (!Number.isFinite(tp)) return;
        const meta = Array.isArray(mergedPlanMeta.target_meta) ? mergedPlanMeta.target_meta[idx] || {} : {};
        const sequence = Number.isFinite(meta.sequence) ? meta.sequence : idx + 1;
        const isRunner = sequence >= 3;
        const label = isRunner ? 'Runner' : meta.label || `TP${sequence}`;
        const color = isRunner ? '#c084fc' : '#22c55e';
        const id = isRunner ? 'runner-tp' : `tp:${sequence}`;
        registerLine(id, {
          price: tp,
          color,
          title: label,
          lineWidth: 2,
          lineStyle: LightweightCharts.LineStyle.Dashed,
        });
      });

      if (Number.isFinite(lastVwap)) {
        registerLine('vwap', {
          price: lastVwap,
          color: '#f8fafc',
          title: 'VWAP',
          lineWidth: 2,
          lineStyle: LightweightCharts.LineStyle.Solid,
        });
      }

      if (scenarioOverlayData) {
        applyScenarioOverlayLines(activeLineIds);
      }

      [...keyLevels]
        .filter((level) => Number.isFinite(level.price))
        .sort((a, b) => b.price - a.price)
        .forEach((level, idx) => {
          const label = formatLevelLabel(level, idx);
          const id = buildLevelLineId(label, level.price);
          registerLine(id, {
            price: level.price,
            color: '#94a3b8',
            title: label,
            lineWidth: 1,
            lineStyle: LightweightCharts.LineStyle.Dotted,
          });
        });
      if (planForFrame.runner && Number.isFinite(planForFrame.runner.anchor)) {
        const runnerLabel = planForFrame.runner.label || 'Runner Trail';
        registerLine('runner-anchor', {
          price: planForFrame.runner.anchor,
          color: '#ff4fa3',
          title: runnerLabel,
          lineWidth: 2,
          lineStyle: LightweightCharts.LineStyle.Dashed,
        });
      }

      prunePriceLines(activeLineIds);

      candleSeries.setMarkers([]);

      const allowAutoFocus = forceAutoFocus || followLive;
      if (focusToken === 'plan') {
        if (allowAutoFocus) {
          applyPlanPriceFocus(planForFrame, { force: true });
          applyPlanTimeWindow({ force: true });
          if (centerTimeToken) {
            applyTimeCentering({ force: true });
          } else {
            chart.timeScale().scrollToPosition(0, true);
          }
        }
      } else if (allowAutoFocus) {
        chart.priceScale('right').applyOptions({ autoScale: true });
        chart.timeScale().fitContent();
        if (centerTimeToken) {
          applyTimeCentering({ force: true });
        }
      }

      debugEl.style.display = 'none';
      debugEl.textContent = '';
    } catch (err) {
      debugEl.style.display = 'block';
      debugEl.textContent = `Error loading data: ${err.message}`;
    }
    forceAutoFocus = false;
  };

  renderHeader();
  initializeTimeframes();
  setWatermark();
  fetchBars();
  connectStream();
  loadTickerInsights();
  window.addEventListener('resize', () => {
    chart.resize(container.clientWidth, container.clientHeight);
    if (planPanelEl && window.innerWidth > 1024) {
      planPanelEl.open = true;
    }
    renderPlanPanel(lastKnownPrice);
    if (focusToken === 'plan') {
      applyPlanTimeWindow({ force: true });
    }
    renderSessionBands();
  });
  window.setInterval(fetchBars, TIMEFRAME_REFRESH_MS);
  window.addEventListener('beforeunload', () => {
    if (streamSource) {
      try {
        streamSource.close();
      } catch (err) {
        // ignore
      }
    }
  });
})().catch((error) => {
  console.error('tv bootstrap error', error);
});
