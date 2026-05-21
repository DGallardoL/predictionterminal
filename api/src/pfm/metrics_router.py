"""``GET /metrics/audit`` — per-endpoint latency + 5xx audit.

This is the read side of :mod:`pfm.metrics`. It exposes percentile latency
(p50/p95/p99) and the 5xx error rate observed since process start, grouped
by templated path (so e.g. ``/terminal/jumps/btc-price-by-eoy`` and
``/terminal/jumps/eth-price-by-eoy`` collapse to the route template
``/terminal/jumps/{slug}``).

The endpoint is intentionally namespaced under ``/metrics/*`` so the
recording middleware in :mod:`pfm.main` can skip it (preventing recursion
and keeping the audit a true reflection of upstream traffic).
"""

from __future__ import annotations

from fastapi import APIRouter

from pfm.metrics import get_tracker

router = APIRouter()


@router.get(
    "/metrics/audit",
    summary="Per-endpoint latency p50/p95/p99 + 5xx rate audit",
    tags=["ops"],
)
def metrics_audit() -> dict:
    """Return the current snapshot of per-endpoint latency observations.

    Response shape::

        {
            "endpoints": {
                "/health": {
                    "count": 12,
                    "p50_ms": 1.234,
                    "p95_ms": 4.567,
                    "p99_ms": 7.890,
                    "err_rate": 0.0,
                    "errors_5xx": 0
                },
                ...
            },
            "total_requests": 12
        }
    """
    tracker = get_tracker()
    endpoints = tracker.snapshot()
    return {
        "endpoints": endpoints,
        "total_requests": tracker.total_requests(),
    }
