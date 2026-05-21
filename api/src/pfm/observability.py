"""Prometheus metrics for the FastAPI app.

Exposes a ``/metrics`` endpoint (text/plain) and a request-timing middleware
that populates the standard counters/histograms below. Other modules can
import the counters directly to record domain-specific events (cache hits,
upstream calls, alpha-lab runs, alerts, factor-history fetches, etc.).

For ad-hoc instrumentation use the :func:`track_metric` decorator, which
times any callable and increments a per-call counter. See module-level
metric definitions for the full set of series exposed at ``/metrics``.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from typing import Any, TypeVar

from fastapi import FastAPI
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.requests import Request
from starlette.responses import Response

# --- HTTP request metrics ---------------------------------------------------

requests_total = Counter(
    "requests_total",
    "Total HTTP requests served by the API.",
    labelnames=("path", "method", "status"),
)

request_duration_seconds = Histogram(
    "request_duration_seconds",
    "HTTP request latency in seconds.",
    labelnames=("path", "method"),
)

# --- Cache metrics ----------------------------------------------------------

cache_hits = Counter(
    "cache_hits",
    "Cache lookups that returned a stored value.",
    labelnames=("backend",),
)

cache_misses = Counter(
    "cache_misses",
    "Cache lookups that did not find a stored value.",
    labelnames=("backend",),
)

# --- Upstream-source metrics ------------------------------------------------

polymarket_requests_total = Counter(
    "polymarket_requests_total",
    "Outbound HTTP requests to Polymarket APIs.",
    labelnames=("endpoint", "status"),
)

factor_history_fetches_total = Counter(
    "pfm_factor_history_fetches_total",
    "Factor-history fetches by source and outcome.",
    labelnames=("source", "status"),
)

factor_history_fetch_duration_seconds = Histogram(
    "pfm_factor_history_fetch_duration_seconds",
    "Latency of a single factor-history fetch by source.",
    labelnames=("source",),
)

# --- Alpha-lab + alerts -----------------------------------------------------

alpha_lab_runs_total = Counter(
    "pfm_alpha_lab_runs_total",
    "Alpha-lab strategy runs by lifecycle status.",
    labelnames=("status",),
)

alerts_fired_total = Counter(
    "pfm_alerts_fired_total",
    "Alerts dispatched by rule kind, channel, and delivery status.",
    labelnames=("rule_kind", "channel", "status"),
)

# --- Realtime / SSE ---------------------------------------------------------

realtime_clients = Gauge(
    "pfm_realtime_clients",
    "Number of currently connected SSE clients.",
)

realtime_pollers = Gauge(
    "pfm_realtime_pollers",
    "Number of active upstream pollers in the realtime hub.",
)

# --- Live signals -----------------------------------------------------------

live_signals_last_run_age_seconds = Gauge(
    "pfm_live_signals_last_run_age_seconds",
    "Seconds since the last successful live_signals recompute.",
)

live_signals_recompute_duration_seconds = Histogram(
    "pfm_live_signals_recompute_duration_seconds",
    "Wall-time of a full live_signals recompute pass.",
)

# --- Factor model fits ------------------------------------------------------

factor_model_fits_total = Counter(
    "pfm_factor_model_fits_total",
    "Factor-model fits served by /fit, labelled by ticker and factor count.",
    labelnames=("ticker", "n_factors"),
)

factor_model_fit_duration_seconds = Histogram(
    "pfm_factor_model_fit_duration_seconds",
    "Wall-time of a single factor-model fit.",
)


# --- track_metric decorator -------------------------------------------------

F = TypeVar("F", bound=Callable[..., Any])


def track_metric(name: str, **labels: str) -> Callable[[F], F]:
    """Time ``func`` and bump ``<name>_total`` + ``<name>_duration_seconds``.

    The decorator lazily creates a Counter and a Histogram named after the
    first ``name`` it sees. Subsequent calls reuse the same metric
    instances, so calling ``track_metric("foo")`` from multiple modules is
    safe — Prometheus' default registry de-duplicates by name.

    Labels passed as kwargs are bound at decoration time. For per-call
    dynamic labels, instrument the callable manually with the module-level
    counters above.

    Example::

        @track_metric("pfm_factor_model_fits", ticker="aapl")
        def fit_aapl(...): ...
    """
    counter_name = f"{name}_total"
    hist_name = f"{name}_duration_seconds"

    label_keys = tuple(sorted(labels.keys()))
    label_values = tuple(labels[k] for k in label_keys)

    counter = _get_or_create_counter(counter_name, f"Calls to {name}", label_keys)
    hist = _get_or_create_hist(hist_name, f"Latency of {name} in seconds", label_keys)

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                if label_keys:
                    counter.labels(*label_values).inc()
                    hist.labels(*label_values).observe(elapsed)
                else:
                    counter.inc()
                    hist.observe(elapsed)
            return result

        return wrapper  # type: ignore[return-value]

    return decorator


def _get_or_create_counter(name: str, doc: str, labelnames: tuple[str, ...]) -> Counter:
    """Return an existing Counter by name, or create one. Idempotent."""
    from prometheus_client import REGISTRY

    existing = REGISTRY._names_to_collectors.get(name)
    if existing is not None:
        return existing  # type: ignore[return-value]
    if labelnames:
        return Counter(name, doc, labelnames=labelnames)
    return Counter(name, doc)


def _get_or_create_hist(name: str, doc: str, labelnames: tuple[str, ...]) -> Histogram:
    """Return an existing Histogram by name, or create one. Idempotent."""
    from prometheus_client import REGISTRY

    existing = REGISTRY._names_to_collectors.get(name)
    if existing is not None:
        return existing  # type: ignore[return-value]
    if labelnames:
        return Histogram(name, doc, labelnames=labelnames)
    return Histogram(name, doc)


# --- Wiring -----------------------------------------------------------------


def setup_metrics(app: FastAPI) -> None:
    """Register the metrics middleware and the ``/metrics`` endpoint on ``app``.

    The middleware uses ``request.url.path`` for the ``path`` label, which
    can explode cardinality if many distinct paths exist; this is acceptable
    here because the route table is bounded and known at startup.
    """

    @app.middleware("http")
    async def _metrics_middleware(request: Request, call_next):
        start = time.perf_counter()
        try:
            response = await call_next(request)
            status = response.status_code
        except Exception:
            requests_total.labels(path=request.url.path, method=request.method, status="500").inc()
            request_duration_seconds.labels(path=request.url.path, method=request.method).observe(
                time.perf_counter() - start
            )
            raise
        request_duration_seconds.labels(path=request.url.path, method=request.method).observe(
            time.perf_counter() - start
        )
        requests_total.labels(
            path=request.url.path, method=request.method, status=str(status)
        ).inc()
        return response

    @app.get("/metrics", include_in_schema=False)
    def _metrics_endpoint() -> Response:
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
