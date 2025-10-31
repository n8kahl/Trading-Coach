"use client";

import clsx from "clsx";
import * as React from "react";
import styles from "./CoachNote.module.css";
import type { CoachNote } from "@/lib/plan/coach";

type CoachNoteProps = {
  note: CoachNote;
  subdued?: boolean;
  loading?: boolean;
  actions?: React.ReactNode;
  metrics?: Array<{ key: string; label: string; value: string; ariaLabel?: string }>;
  live?: boolean;
};

export default function CoachNote({
  note,
  subdued = false,
  loading = false,
  actions = null,
  metrics = [],
  live = true,
}: CoachNoteProps) {
  const contentRef = React.useRef<HTMLParagraphElement | null>(null);
  const [canExpand, setCanExpand] = React.useState(false);
  const [expanded, setExpanded] = React.useState(false);

  React.useEffect(() => {
    setExpanded(false);
  }, [note.text, note.updatedAt]);

  React.useEffect(() => {
    if (typeof window === "undefined") return;
    const element = contentRef.current;
    if (!element || typeof ResizeObserver === "undefined") {
      return;
    }
    const observer = new ResizeObserver(() => {
      const lineHeight = parseFloat(window.getComputedStyle(element).lineHeight || "0");
      if (!Number.isFinite(lineHeight) || lineHeight === 0) {
        setCanExpand(false);
        return;
      }
      const clampHeight = lineHeight * 6;
      setCanExpand(element.scrollHeight - 1 > clampHeight);
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, [note.text, note.updatedAt]);

  const handleToggle = React.useCallback(() => {
    setExpanded((prev) => !prev);
  }, []);

  const progress = Number.isFinite(note.progressPct) ? Math.max(0, Math.min(100, Math.round(note.progressPct))) : 0;
  const progressScale = Number.isFinite(note.progressPct) ? Math.max(0, Math.min(1, note.progressPct / 100)) : 0;
  const filteredMetrics = React.useMemo(
    () =>
      metrics.filter(
        (metric, index, list) =>
          metric &&
          typeof metric.label === "string" &&
          metric.label.trim().length > 0 &&
          typeof metric.value === "string" &&
          metric.value.trim().length > 0 &&
          list.findIndex((entry) => entry.key === metric.key) === index,
      ),
    [metrics],
  );

  return (
    <section
      className={clsx(styles.root, subdued && "border-neutral-700/40 bg-neutral-900/80", loading && styles.loading)}
      aria-live="polite"
      aria-atomic="true"
    >
      <div className={styles.header}>
        <div className={styles.titleRow}>
          <span
            className={clsx(styles.liveDot, !live && styles.liveDotPaused)}
            aria-hidden="true"
          />
          <span className={styles.label}>Coach Guidance</span>
          {filteredMetrics.length ? (
            <ul className={styles.metricList}>
              {filteredMetrics.map((metric) => (
                <li key={metric.key}>
                  <span
                    className={styles.metricPill}
                    aria-label={metric.ariaLabel ?? `${metric.label} ${metric.value}`}
                  >
                    <span className={styles.metricLabel}>{metric.label}</span>
                    <span className={styles.metricValue}>{metric.value}</span>
                  </span>
                </li>
              ))}
            </ul>
          ) : null}
        </div>
        <span
          className={styles.progressBadge}
          title="Progress to next objective reflects the server-reported progress toward the current coaching goal."
          aria-label={`Progress to next objective ${progress}%`}
        >
          <span className={styles.progressBadgeLabel}>Progress to next objective</span>
          <span className={styles.progressValue}>{progress}%</span>
        </span>
      </div>
      <div className={styles.contentWrapper}>
        <p
          key={note.updatedAt}
          ref={contentRef}
          className={clsx(styles.content, !expanded && canExpand && styles.contentClamped)}
        >
          {note.text}
        </p>
        {!expanded && canExpand ? <span className={styles.fade} /> : null}
      </div>
      {actions || canExpand ? (
        <div className={styles.controls}>
          {actions}
          {canExpand ? (
            <button type="button" className={styles.expandButton} onClick={handleToggle}>
              {expanded ? "Collapse" : "Expand"}
            </button>
          ) : null}
        </div>
      ) : null}
      <div className={styles.progressBar} role="presentation">
        <div className={styles.progressIndicator} style={{ transform: `scaleX(${progressScale})` }} />
      </div>
    </section>
  );
}
