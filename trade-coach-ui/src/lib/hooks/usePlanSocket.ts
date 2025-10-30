import { useEffect, useRef, useState } from "react";
import { makeBackoff } from "./useBackoff";

type SocketStatus = "connecting" | "connected" | "disconnected";

export function usePlanSocket(url: string, onDelta: (msg: unknown) => void): SocketStatus {
  const [status, setStatus] = useState<SocketStatus>("connecting");
  const backoff = useRef(makeBackoff());

  useEffect(() => {
    let closed = false;
    let ws: WebSocket | null = null;
    let retryTimer: number | null = null;

    function scheduleReconnect() {
      if (closed) return;
      setStatus("disconnected");
      if (retryTimer != null) window.clearTimeout(retryTimer);
      retryTimer = window.setTimeout(connect, backoff.current());
    }

    function connect() {
      if (closed) return;
      setStatus("connecting");
      ws = new WebSocket(url);
      ws.onopen = () => {
        setStatus("connected");
        backoff.current = makeBackoff();
      };
      ws.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data as string);
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

