// Helper to initialize the ChatKit Web Component
//
// This script waits for the DOM to load, then attaches a click handler to the
// “Start Chat” button.  When clicked it requests a ChatKit session from the
// backend and mounts the <openai-chatkit> element into the page.  The
// component accepts attributes such as `client-token` and `workflow-id` to
// configure the chat.  For more options see the official documentation.

document.addEventListener('DOMContentLoaded', () => {
  const API_BASE = 'http://localhost:8000';
  const tradeContext = {
    tradeId: null,
    symbol: 'NVDA',
    direction: 'long',
    entry: 465.5,
  };

  const button = document.getElementById('startChat');
  const container = document.getElementById('chatContainer');
  const followBtn = document.getElementById('btnFollowTrade');
  const updateBtn = document.getElementById('btnTradeUpdates');
  const explainBtn = document.getElementById('btnExplainTrade');
  const feedContainer = document.getElementById('tradeFeedMessages');

  const appendFeed = (message, variant = 'info') => {
    const entry = document.createElement('div');
    entry.className = `feed-entry feed-${variant}`;
    entry.innerHTML = `<strong>${new Date().toLocaleTimeString()}:</strong> ${message}`;
    if (feedContainer.querySelector('.placeholder')) {
      feedContainer.innerHTML = '';
    }
    feedContainer.appendChild(entry);
  };

  const initTradingView = (symbol) => {
    const loadChart = () => {
      if (window.TradingView && window.TradingView.widget) {
        // Clear any previous widget contents
        const target = document.getElementById('tradingviewContainer');
        if (!target) {
          return;
        }
        target.innerHTML = '';
        // eslint-disable-next-line no-new
        new window.TradingView.widget({
          autosize: true,
          symbol: `${symbol}`,
          interval: '5',
          timezone: 'Etc/UTC',
          theme: 'light',
          style: '1',
          locale: 'en',
          hide_side_toolbar: false,
          withdateranges: true,
          allow_symbol_change: true,
          container_id: 'tradingviewContainer',
        });
      } else {
        setTimeout(loadChart, 400);
      }
    };
    loadChart();
  };

  const ensureTradeId = () => {
    if (!tradeContext.tradeId) {
      tradeContext.tradeId = `${tradeContext.symbol.toLowerCase()}-${Date.now()}`;
    }
    return tradeContext.tradeId;
  };

  button.addEventListener('click', async () => {
    button.disabled = true;
    button.textContent = 'Loading...';
    try {
      const resp = await fetch(`${API_BASE}/api/chatkit/session`, { method: 'POST' });
      if (!resp.ok) {
        const text = await resp.text().catch(() => '');
        throw new Error(`Failed to fetch ChatKit session (${resp.status}): ${text || resp.statusText}`);
      }
      const { client_secret } = await resp.json();
      // Create the ChatKit component
      const chatEl = document.createElement('openai-chatkit');
      chatEl.setAttribute('client-token', client_secret);
      // Optional: specify your workflow ID here if needed
      // chatEl.setAttribute('workflow-id', 'YOUR_WORKFLOW_ID');
      chatEl.style.width = '100%';
      chatEl.style.height = '600px';
      // Clear any existing content and append the chat
      container.innerHTML = '';
      container.appendChild(chatEl);

      chatEl.addEventListener('chatkit:ready', () => {
        appendFeed('Chat assistant connected. Ask it to monitor the NVDA breakout.', 'success');
      });

      // If we only have a dummy token, warn the user and re-enable the button.
      if (!client_secret || String(client_secret).startsWith('dummy')) {
        console.warn('ChatKit is using a dummy client token. Set real keys in .env to initialize the widget.');
        const note = document.createElement('p');
        note.style.color = '#a00';
        note.textContent = 'Chat not connected (dummy token). Set OPENAI_API_KEY and WORKFLOW_ID in .env and restart the API.';
        container.prepend(note);
      }

      // Restore the button so it doesn’t look frozen
      button.disabled = false;
      button.textContent = 'Start Chat';
    } catch (err) {
      console.error(err);
      alert('Error initializing chat: ' + err.message);
      button.disabled = false;
      button.textContent = 'Start Chat';
    }
  });

  followBtn?.addEventListener('click', async () => {
    const tradeId = ensureTradeId();
    try {
      const resp = await fetch(`${API_BASE}/api/follow`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          trade_id: tradeId,
          symbol: tradeContext.symbol,
          direction: tradeContext.direction,
          entry_price: tradeContext.entry,
        }),
      });
      if (!resp.ok) {
        const text = await resp.text().catch(() => '');
        throw new Error(text || `Follow request failed with status ${resp.status}`);
      }
      const payload = await resp.json();
      appendFeed(`Follower started: ${payload.message}`, 'success');
    } catch (error) {
      console.error(error);
      appendFeed(`Failed to follow trade: ${error.message}`, 'error');
    }
  });

  updateBtn?.addEventListener('click', () => {
    const tradeId = ensureTradeId();
    appendFeed(`Subscribed to trade updates for ${tradeId}. Monitor ChatKit for live coaching.`, 'info');
  });

  explainBtn?.addEventListener('click', () => {
    appendFeed(
      'Trade rationale: NVDA is reclaiming the opening range with strong volume and stacked EMAs. ATR-based stop allows 1R risk, targeting VWAP extensions.',
      'info',
    );
  });

  initTradingView(tradeContext.symbol);
});
