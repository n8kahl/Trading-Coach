import { useEffect, useRef, useState } from "react";
import { makeBackoff } from "./useBackoff";

type StreamStatus = "connecting" | "connected" | "disconnected";

export function useSymbolStream(url: string, onTick: (tick: unknown) => void): StreamStatus {
  const [status, setStatus] = useState<StreamStatus>("connecting");
  const backoff = useRef(makeBackoff());

  useEffect(() => {
    let closed = false;
    let source: EventSource | null = null;
    let retryTimer: number | null = null;

    function scheduleReconnect() {
      if (closed) return;
      setStatus("disconnected");
      if (retryTimer != null) window.clearTimeout(retryTimer);
      source?.close();
      retryTimer = window.setTimeout(connect, backoff.current());
    }

    function connect() {
      if (closed) return;
      setStatus("connecting");
      source = new EventSource(url);
      source.onopen = () => {
        setStatus("connected");
        backoff.current = makeBackoff();
      };
      source.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data);
          onTick(payload);
        } catch {
          // ignore malformed payloads
        }
      };
      source.onerror = scheduleReconnect;
    }

    connect();

    return () => {
      closed = true;
      if (retryTimer != null) window.clearTimeout(retryTimer);
      source?.close();
    };
  }, [url, onTick]);

  return status;
}

