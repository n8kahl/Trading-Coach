"use client";

import clsx from "clsx";
import * as React from "react";
import styles from "./CoachNote.module.css";
import type { CoachNote } from "@/lib/plan/coach";

type CoachNoteProps = {
  note: CoachNote;
  subdued?: boolean;
};

export default function CoachNote({ note, subdued = false }: CoachNoteProps) {
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

  return (
    <section
      className={clsx(styles.root, subdued && "border-neutral-700/40 bg-neutral-900/80")}
      aria-live="polite"
      aria-atomic="true"
    >
      <div className={styles.header}>
        <span className={styles.label}>Coach Guidance</span>
        <span className={styles.progressValue}>{progress}%</span>
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
      <div className={styles.controls}>
        {canExpand ? (
          <button type="button" className={styles.expandButton} onClick={handleToggle}>
            {expanded ? "Collapse" : "Expand"}
          </button>
        ) : null}
      </div>
      <div className={styles.progressBar} role="presentation">
        <div className={styles.progressIndicator} style={{ transform: `scaleX(${progressScale})` }} />
      </div>
    </section>
  );
}
