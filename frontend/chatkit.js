// Initial hint for API base. We'll auto-resolve at runtime if wrong.
const BACKEND_URL = (typeof window !== 'undefined' && window.__BACKEND_URL__ !== undefined)
  ? window.__BACKEND_URL__
  : '';

const state = {
  widgets: [],
};

const selectors = {
  chatElement: document.getElementById("chatkit"),
  widgetList: document.getElementById("widgetList"),
  widgetPlaceholder: document.getElementById("widgetPlaceholder"),
  reconnectButton: document.getElementById("reconnect"),
  statusBanner: document.getElementById("statusBanner"),
};

const renderWidgets = () => {
  const { widgetList, widgetPlaceholder } = selectors;
  if (!widgetList) {
    return;
  }
  widgetList.innerHTML = "";
  if (!state.widgets.length) {
    widgetPlaceholder?.removeAttribute("hidden");
    widgetList.appendChild(widgetPlaceholder);
    return;
  }
  widgetPlaceholder?.setAttribute("hidden", "true");
  state.widgets.forEach((widget) => {
    const card = document.createElement("article");
    card.className = "widget-card";
    if (widget.type === "trade_proposal") {
      card.innerHTML = `
        <h3>${widget.symbol || "Trade Idea"}</h3>
        <p><strong>Strategy:</strong> ${widget.strategy || "—"}</p>
        <p><strong>Entry:</strong> ${widget.entry || "—"}</p>
        <p><strong>Stop:</strong> ${widget.stop || "—"}</p>
        <p><strong>Target:</strong> ${widget.target || "—"}</p>
        ${widget.rationale ? `<p style="margin-top:0.5rem;">${widget.rationale}</p>` : ""}
        <div class="widget-actions">
          <button type="button" data-action="follow">Follow Trade</button>
          <button type="button" data-action="alerts">Alerts</button>
          <button type="button" data-action="explain">Explain</button>
        </div>
      `;
    } else if (widget.type === "trading_plan") {
      const steps = (widget.steps || []).map((step) => `<li>${step}</li>`).join("");
      card.innerHTML = `
        <h3>${widget.title || "Trading Plan"}</h3>
        <ol>${steps}</ol>
      `;
    } else if (widget.type === "chart") {
      const symbol = widget.symbol || "SPY";
      card.innerHTML = `
        <h3>Live Chart – ${symbol}</h3>
        <iframe
          title="TradingView ${symbol}"
          src="https://s.tradingview.com/widgetembed/?symbol=${encodeURIComponent(symbol)}&interval=${encodeURIComponent(widget.interval || "5")}&hide_top_toolbar=1&symboledit=1&saveimage=0&theme=dark"
          style="border:0;width:100%;height:260px;border-radius:12px;"
          loading="lazy"
        ></iframe>
      `;
    } else {
      card.innerHTML = `<h3>Assistant Widget</h3><p>${JSON.stringify(widget)}</p>`;
    }
    widgetList.appendChild(card);
  });
};

const appendWidget = (payload) => {
  if (!payload || typeof payload !== "object") {
    return;
  }
  if (payload.reset) {
    state.widgets = [];
  }
  if (payload.type) {
    state.widgets.push(payload);
  }
  renderWidgets();
};

const parseWidgetBlocks = (message) => {
  if (!message?.content) {
    return;
  }
  message.content.forEach((block) => {
    if (block.type !== "output_text" || typeof block.text !== "string") {
      return;
    }
    const text = block.text.trim();
    if (!text.startsWith("{")) {
      return;
    }
    try {
      const parsed = JSON.parse(text);
      if (parsed.widget) {
        appendWidget(parsed.widget);
      }
    } catch (error) {
      console.warn("Widget JSON parse failed", error);
    }
  });
};

const setStatus = (text, options = {}) => {
  const { statusBanner } = selectors;
  if (!statusBanner) return;
  if (!text) {
    statusBanner.classList.remove("visible");
    statusBanner.textContent = "";
    return;
  }
  statusBanner.textContent = text;
  statusBanner.classList.add("visible");
  if (options.variant === "error") {
    statusBanner.style.background = "rgba(248, 113, 113, 0.2)";
    statusBanner.style.border = "1px solid rgba(248, 113, 113, 0.45)";
    statusBanner.style.color = "#fecaca";
  } else {
    statusBanner.style.background = "rgba(59, 130, 246, 0.15)";
    statusBanner.style.border = "1px solid rgba(59, 130, 246, 0.35)";
    statusBanner.style.color = "#bfdbfe";
  }
};

let apiBase = BACKEND_URL;

const postLog = async (level, message, context = {}) => {
  try {
    const base = apiBase || '';
    await fetch(`${base}/api/debug/log`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ level, message, context }),
    });
  } catch (e) {
    // ignore network errors here
    console.debug('[client-log-failed]', e);
  }
};

const loadChatKitScript = async () => {
  const urls = [
    // Try our same-origin proxy first (served by FastAPI)
    `${apiBase || ''}/assets/chatkit.js`,
    // Public CDNs
    'https://cdn.jsdelivr.net/npm/@openai/chatkit-widget@latest/dist/web.js',
    'https://cdn.platform.openai.com/deployments/chatkit/chatkit.js',
    'https://unpkg.com/@openai/chatkit-widget@latest/dist/web.js',
  ];
  for (const url of urls) {
    await postLog('info', 'loading chatkit script', { url });
    const p = new Promise((resolve, reject) => {
      const s = document.createElement('script');
      s.src = url;
      s.async = true;
      s.onload = () => {
        postLog('info', 'chatkit script load success', { url });
        resolve();
      };
      s.onerror = (e) => {
        postLog('error', 'chatkit script load error', { url });
        reject(new Error('Script load failed'));
      };
      document.head.appendChild(s);
    });
    try {
      await p;
      // verify component registration
      if (window.customElements?.get('openai-chatkit')) {
        return true;
      }
    } catch (e) {
      // try next url
    }
  }
  return false;
};

const trySession = async (base) => {
  try {
    const res = await fetch(`${base}/api/chatkit/session`, { method: 'POST' });
    if (!res.ok) {
      const text = await res.text().catch(() => '');
      throw new Error(`status=${res.status} ${text}`);
    }
    const json = await res.json();
    return json;
  } catch (e) {
    await postLog('warning', 'session attempt failed', { base, err: String(e) });
    return null;
  }
};

const bootstrapChatKit = async () => {
  const { chatElement, reconnectButton } = selectors;
  if (!chatElement) {
    console.error("Missing chat element");
    return;
  }
  setStatus("Connecting…");
  reconnectButton?.classList.remove("visible");
  reconnectButton?.setAttribute("disabled", "true");

  try {
    const ok = await loadChatKitScript();
    if (!ok || !window.customElements?.get('openai-chatkit')) {
      await postLog('error', 'openai-chatkit component not registered', {
        userAgent: navigator.userAgent,
        origin: location.origin,
      });
      setStatus('Assistant script unavailable', { variant: 'error' });
      reconnectButton?.classList.add('visible');
      reconnectButton?.removeAttribute('disabled');
      return;
    }
    // Resolve API base: prefer same-origin '', otherwise fall back to configured hint or localhost.
    const candidates = Array.from(new Set([
      '',
      BACKEND_URL || '',
      'http://localhost:8000',
    ]));
    let session = null;
    for (const base of candidates) {
      session = await trySession(base);
      if (session && session.client_secret) {
        apiBase = base; // remember working base
        break;
      }
    }
    if (!session || !session.client_secret) {
      throw new Error('All session endpoints failed');
    }
    const { client_secret: clientSecret, workflow_id: workflowId, user_id: userId } = session;
    if (!clientSecret || typeof clientSecret !== "string") {
      throw new Error("Backend did not return a client_secret.");
    }
    if (clientSecret.startsWith("dummy")) {
      setStatus("Unable to authenticate (dummy token)", { variant: "error" });
      await postLog('error', 'dummy client token received', {});
      reconnectButton?.classList.add("visible");
      reconnectButton?.removeAttribute("disabled");
      return;
    }
    // Set both attribute spellings to match differing builds
    chatElement.setAttribute("client-token", clientSecret);
    chatElement.setAttribute("client-secret", clientSecret);
    if (workflowId) {
      chatElement.setAttribute("workflow-id", workflowId);
    }
    if (userId) {
      chatElement.setAttribute("user-id", userId);
    }
    await postLog('info', 'applied client secret to element', {
      hasToken: Boolean(clientSecret),
      workflowId: workflowId || null,
    });
  } catch (error) {
    console.error("Failed to initialize ChatKit", error);
    setStatus("Connection failed", { variant: "error" });
    reconnectButton?.classList.add("visible");
    reconnectButton?.removeAttribute("disabled");
  }
};

document.addEventListener("DOMContentLoaded", () => {
  const { chatElement, reconnectButton } = selectors;
  if (!chatElement) {
    console.error("ChatKit element missing from DOM.");
    return;
  }

  chatElement.addEventListener("chatkit:ready", () => {
    setStatus("Online");
    postLog('info', 'chatkit:ready');
    setTimeout(() => setStatus(""), 1500);
  });

  chatElement.addEventListener("chatkit:error", (event) => {
    console.error("ChatKit reported an error", event?.detail);
    postLog('error', 'chatkit:error', { detail: event?.detail || null });
    setStatus("Assistant error", { variant: "error" });
    reconnectButton?.classList.add("visible");
    reconnectButton?.removeAttribute("disabled");
  });

  chatElement.addEventListener("chatkit:assistant-message", (event) => {
    const detail = event.detail || {};
    postLog('info', 'assistant-message', { keys: Object.keys(detail || {}) });
    parseWidgetBlocks(detail.message);
  });

  reconnectButton?.addEventListener("click", () => {
    state.widgets = [];
    renderWidgets();
    bootstrapChatKit();
  });

  renderWidgets();
  bootstrapChatKit();
});
