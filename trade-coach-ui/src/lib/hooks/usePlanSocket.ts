import { useEffect, useMemo, useRef, useState } from "react";
import { getPlanStream, type SocketStatus } from "@/lib/streams/planStream";

export function usePlanSocket(url: string, planId: string, onDelta: (msg: unknown) => void): SocketStatus {
  const stream = useMemo(() => getPlanStream(planId, url), [planId, url]);
  const [status, setStatus] = useState<SocketStatus>(stream.getStatus());
  const [heartbeat, setHeartbeat] = useState<number>(() => stream.getLastHeartbeat());
  const [derivedStatus, setDerivedStatus] = useState<SocketStatus>(stream.getStatus());
  const listenerRef = useRef(onDelta);

  useEffect(() => {
    listenerRef.current = onDelta;
  }, [onDelta]);

  useEffect(() => stream.onStatus(setStatus), [stream]);

  useEffect(
    () =>
      stream.onHeartbeat((ts) => {
        setHeartbeat(ts);
      }),
    [stream],
  );

  useEffect(() => {
    const unsubscribe = stream.subscribe((payload) => {
      listenerRef.current(payload);
    });
    return () => {
      unsubscribe();
    };
  }, [stream]);

  useEffect(() => {
    const update = () => {
      if (status === "disconnected") {
        setDerivedStatus("disconnected");
        return;
      }
      if (!heartbeat) {
        setDerivedStatus(status);
        return;
      }
      const age = Date.now() - heartbeat;
      if (age > 35_000 && status === "connected") {
        setDerivedStatus("connecting");
      } else {
        setDerivedStatus(status);
      }
    };
    update();
    if (typeof window === "undefined") return;
    const timer = window.setInterval(update, 5_000);
    return () => {
      window.clearInterval(timer);
    };
  }, [status, heartbeat]);

  return derivedStatus;
}
