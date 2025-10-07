const BACKEND_URL = window.__BACKEND_URL__ || "http://localhost:8000";

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

const ensureWebComponent = async () => {
  if (window.customElements?.get("openai-chatkit")) {
    return;
  }
  if (window.customElements?.whenDefined) {
    try {
      await window.customElements.whenDefined("openai-chatkit");
      return;
    } catch {
      /* ignore */
    }
  }
  // Fallback polling
  await new Promise((resolve) => setTimeout(resolve, 300));
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
    await ensureWebComponent();
    const response = await fetch(`${BACKEND_URL}/api/chatkit/session`, {
      method: "POST",
    });
    if (!response.ok) {
      const detail = await response.text();
      throw new Error(detail || `Session request failed (${response.status})`);
    }
    const { client_secret: clientSecret, workflow_id: workflowId, user_id: userId } = await response.json();
    if (!clientSecret || typeof clientSecret !== "string") {
      throw new Error("Backend did not return a client_secret.");
    }
    if (clientSecret.startsWith("dummy")) {
      setStatus("Unable to authenticate (dummy token)", { variant: "error" });
      reconnectButton?.classList.add("visible");
      reconnectButton?.removeAttribute("disabled");
      return;
    }
    chatElement.setAttribute("client-token", clientSecret);
    if (workflowId) {
      chatElement.setAttribute("workflow-id", workflowId);
    }
    if (userId) {
      chatElement.setAttribute("user-id", userId);
    }
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
    setTimeout(() => setStatus(""), 1500);
  });

  chatElement.addEventListener("chatkit:error", (event) => {
    console.error("ChatKit reported an error", event?.detail);
    setStatus("Assistant error", { variant: "error" });
    reconnectButton?.classList.add("visible");
    reconnectButton?.removeAttribute("disabled");
  });

  chatElement.addEventListener("chatkit:assistant-message", (event) => {
    const detail = event.detail || {};
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
