"""Prometheus metrics helpers for the Trading Coach backend."""

from __future__ import annotations

from typing import Iterable

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest


PLAN_DURATION_MS = Histogram(
    "plan_duration_ms",
    "Latency of plan hydration in milliseconds.",
    labelnames=("mode",),
    buckets=(25, 50, 75, 100, 150, 250, 400, 600, 1000, 2000),
)

CANDIDATE_COUNT = Histogram(
    "candidate_count",
    "Number of plan candidates evaluated per request.",
    labelnames=("mode",),
    buckets=(1, 2, 3, 5, 10, 20, 40),
)

TP_HIT = Counter(
    "tp_hit",
    "Count of TP1 hits recorded by the evaluator.",
    labelnames=("source",),
)

SL_HIT = Counter(
    "sl_hit",
    "Count of stop-loss hits recorded by the evaluator.",
    labelnames=("source",),
)

RR_BELOW_MIN = Counter(
    "rr_below_min",
    "Number of plans whose RR to TP1 fell below the configured floor.",
    labelnames=("source",),
)

EM_CAPPED_TP = Counter(
    "em_capped_tp",
    "Number of TP1 hits that required expected-move capping.",
    labelnames=("source",),
)

SELECTOR_REJECTED_TOTAL = Counter(
    "selector_rejected_total",
    "Total option selector rejections broken out by reason.",
    labelnames=("reason", "source"),
)


def record_plan_duration(mode: str, duration_ms: float) -> None:
    PLAN_DURATION_MS.labels(mode=mode or "unknown").observe(max(0.0, float(duration_ms)))


def record_candidate_count(mode: str, count: int) -> None:
    CANDIDATE_COUNT.labels(mode=mode or "unknown").observe(max(0, int(count)))


def record_tp_hit(source: str) -> None:
    TP_HIT.labels(source=source or "unknown").inc()


def record_sl_hit(source: str) -> None:
    SL_HIT.labels(source=source or "unknown").inc()


def record_rr_below_min(source: str) -> None:
    RR_BELOW_MIN.labels(source=source or "unknown").inc()


def record_em_capped_tp(source: str) -> None:
    EM_CAPPED_TP.labels(source=source or "unknown").inc()


def record_selector_rejections(rejections: Iterable[dict[str, str | None]], *, source: str) -> None:
    for entry in rejections:
        reason = str(entry.get("reason") or "").upper()
        if not reason:
            continue
        SELECTOR_REJECTED_TOTAL.labels(reason=reason, source=source or "unknown").inc()


def prometheus_response() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST


__all__ = [
    "CANDIDATE_COUNT",
    "EM_CAPPED_TP",
    "PLAN_DURATION_MS",
    "RR_BELOW_MIN",
    "SELECTOR_REJECTED_TOTAL",
    "SL_HIT",
    "TP_HIT",
    "prometheus_response",
    "record_candidate_count",
    "record_em_capped_tp",
    "record_plan_duration",
    "record_rr_below_min",
    "record_selector_rejections",
    "record_sl_hit",
    "record_tp_hit",
]

