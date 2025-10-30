'use client';

import * as React from 'react';
import { validateTvUrl } from '@/lib/utils/url';

type Props = {
  chartUrl: string;                               // MUST be server-provided /tv URL from /gpt/chart-url
  overlays?: Record<string, any>;                 // pass-through; we do not fabricate or transform
  onSupportToggled?: (visible: boolean) => void;  // announces state up to parent if needed
};

export default function ChartContainer({ chartUrl, overlays, onSupportToggled }: Props) {
  const iframeRef = React.useRef<HTMLIFrameElement>(null);
  const [supportVisible, setSupportVisible] = React.useState(true);

  const tvSrc = React.useMemo(() => validateTvUrl(chartUrl), [chartUrl]);

  // postMessage helper to talk to the server-hosted viewer
  const postToTv = React.useCallback((msg: unknown) => {
    const frame = iframeRef.current;
    if (!frame?.contentWindow) return;
    frame.contentWindow.postMessage(msg, '*');
  }, []);

  const toggleSupporting = React.useCallback(() => {
    const next = !supportVisible;
    setSupportVisible(next);
    onSupportToggled?.(next);

    // Preferred: message the viewer to toggle overlay group "supporting"
    postToTv({ type: 'SET_OVERLAY_VISIBILITY', group: 'supporting', visible: next });

    // SR announcement
    const live = document.getElementById('overlay-live');
    if (live) live.textContent = `Supporting levels ${next ? 'shown' : 'hidden'}.`;
  }, [supportVisible, onSupportToggled, postToTv]);

  return (
    <section className="relative rounded-xl border border-[var(--border)] bg-[var(--surface)] overflow-hidden">
      {/* Floating controls (left-top), avoiding price axis on the right */}
      <div className="pointer-events-none absolute left-2 top-2 z-10 flex flex-col gap-2">
        <div className="pointer-events-auto inline-flex items-center gap-1 rounded bg-[var(--chip)] px-2 py-1">
          <span className="text-xs text-[var(--muted)]">Overlays</span>
          <button
            type="button"
            aria-pressed={supportVisible}
            onClick={toggleSupporting}
            className="rounded px-2 py-1 text-xs focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[var(--ring)] hover:bg-[var(--border)]"
          >
            {supportVisible ? 'Hide Supporting Levels' : 'Show Supporting Levels'}
          </button>
        </div>
      </div>

      {/* Live region for overlay announcements */}
      <p id="overlay-live" className="sr-only" aria-live="polite" />

      <iframe
        ref={iframeRef}
        title="Chart"
        src={tvSrc}
        className="block h-[56vh] w-full md:h-[70vh]"
        loading="lazy"
      />
    </section>
  );
}
