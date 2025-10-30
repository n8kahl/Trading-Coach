import { useEffect, useRef, useState } from "react";
import { makeBackoff } from "./useBackoff";

type SocketStatus = "connecting" | "connected" | "disconnected";

export function usePlanSocket(url: string, planId: string, onDelta: (msg: unknown) => void): SocketStatus {
  const [status, setStatus] = useState<SocketStatus>("connecting");
  const backoff = useRef(makeBackoff());
  const planRef = useRef(planId);

  useEffect(() => {
    planRef.current = planId;
  }, [planId]);

  useEffect(() => {
    let closed = false;
    let ws: WebSocket | null = null;
    let retryTimer: number | null = null;
     let reconnectScheduled = false;

    function scheduleReconnect() {
      if (closed) return;
      if (reconnectScheduled) return;
      reconnectScheduled = true;
      setStatus("disconnected");
      if (retryTimer != null) window.clearTimeout(retryTimer);
      if (ws && ws.readyState === WebSocket.OPEN) {
        try {
          ws.close();
        } catch {
          /* ignore */
        }
      }
      ws = null;
      const delay = backoff.current();
      retryTimer = window.setTimeout(() => {
        reconnectScheduled = false;
        connect();
      }, delay);
    }

    function connect() {
      if (closed) return;
      if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
        return;
      }
      reconnectScheduled = false;
      setStatus("connecting");
      ws = new WebSocket(url);
      ws.onopen = () => {
        setStatus("connected");
        backoff.current = makeBackoff();
        try {
          ws?.send(JSON.stringify({ type: "subscribe", planId: planRef.current }));
        } catch {
          /* ignore */
        }
      };
      ws.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data as string);
          if (payload && typeof payload === "object" && payload.type === "ping") {
            ws?.send(JSON.stringify({ type: "pong", planId: planRef.current, ts: Date.now() }));
            return;
          }
          onDelta(payload);
        } catch {
          // ignore malformed payloads
        }
      };
      ws.onerror = scheduleReconnect;
      ws.onclose = scheduleReconnect;
    }

    connect();

    return () => {
      closed = true;
      if (retryTimer != null) window.clearTimeout(retryTimer);
      ws?.close();
    };
  }, [url, onDelta]);

  return status;
}
