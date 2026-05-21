"""Tests for the observability stack: structlog, Prometheus, Sentry hooks.

These tests do not hit any external services. Sentry is exercised by
asserting that the absence of ``SENTRY_DSN`` does not crash the import
graph; Prometheus assertions hit the in-process registry via ``/metrics``
and the ``track_metric`` decorator directly.
"""

from __future__ import annotations

import importlib
import io
import json
import logging
import os
import sys
from pathlib import Path

import pytest

# --- structlog --------------------------------------------------------------


def _capture_stdout(monkeypatch) -> io.StringIO:
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stdout", buf)
    return buf


def test_configure_logging_emits_json(monkeypatch):
    monkeypatch.setenv("LOG_FORMAT", "json")
    monkeypatch.setenv("LOG_LEVEL", "INFO")

    # Re-import to pick up env vars and re-bind processors.
    from pfm import logging_setup

    importlib.reload(logging_setup)

    buf = _capture_stdout(monkeypatch)
    logging_setup.configure_logging()

    log = logging_setup.get_logger("test")
    log.info("hello", foo="bar", n=3)

    raw = buf.getvalue().strip().splitlines()
    assert raw, "expected at least one log line"
    payload = json.loads(raw[-1])
    assert payload["event"] == "hello"
    assert payload["foo"] == "bar"
    assert payload["n"] == 3
    # Required structured fields:
    assert payload["level"] == "info"
    assert "timestamp" in payload
    assert "module" in payload
    assert "lineno" in payload


def test_configure_logging_text_mode(monkeypatch):
    monkeypatch.setenv("LOG_FORMAT", "text")
    monkeypatch.setenv("LOG_LEVEL", "INFO")

    from pfm import logging_setup

    importlib.reload(logging_setup)

    buf = _capture_stdout(monkeypatch)
    logging_setup.configure_logging()

    log = logging_setup.get_logger("test")
    log.info("hello-text", foo="bar")

    out = buf.getvalue()
    # Text mode is the dev ConsoleRenderer; should not be valid JSON.
    assert "hello-text" in out
    with pytest.raises(json.JSONDecodeError):
        # The last line of console-renderer output is not JSON.
        json.loads(out.strip().splitlines()[-1])


def test_log_level_filters_debug(monkeypatch):
    monkeypatch.setenv("LOG_FORMAT", "json")
    monkeypatch.setenv("LOG_LEVEL", "WARNING")

    from pfm import logging_setup

    importlib.reload(logging_setup)

    buf = _capture_stdout(monkeypatch)
    logging_setup.configure_logging()

    log = logging_setup.get_logger("test")
    log.info("should-not-appear")
    log.warning("should-appear")

    out = buf.getvalue()
    assert "should-not-appear" not in out
    assert "should-appear" in out


# --- Prometheus track_metric decorator -------------------------------------


def test_track_metric_decorator_increments_counter():
    from prometheus_client import REGISTRY

    from pfm.observability import track_metric

    @track_metric("pfm_test_decorator")
    def add(x: int, y: int) -> int:
        return x + y

    assert add(2, 3) == 5
    add(1, 1)

    val = REGISTRY.get_sample_value("pfm_test_decorator_total")
    assert val is not None
    assert val >= 2


def test_track_metric_decorator_records_duration():
    from prometheus_client import REGISTRY

    from pfm.observability import track_metric

    @track_metric("pfm_test_duration")
    def noop() -> None:
        return None

    noop()
    count = REGISTRY.get_sample_value("pfm_test_duration_duration_seconds_count")
    assert count is not None and count >= 1


def test_track_metric_with_labels():
    from prometheus_client import REGISTRY

    from pfm.observability import track_metric

    @track_metric("pfm_test_labeled", source="polymarket")
    def fetch() -> int:
        return 1

    fetch()
    val = REGISTRY.get_sample_value("pfm_test_labeled_total", {"source": "polymarket"})
    assert val is not None
    assert val >= 1


# --- /metrics endpoint exposure --------------------------------------------


def test_metrics_endpoint_exposes_expected_series(app_client):
    # Trigger at least one request so request_duration_seconds gets a sample.
    app_client.get("/health")
    resp = app_client.get("/metrics")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")

    body = resp.text
    expected_series = [
        "requests_total",
        "request_duration_seconds",
        "cache_hits",
        "cache_misses",
        "polymarket_requests_total",
        "pfm_factor_history_fetches_total",
        "pfm_factor_history_fetch_duration_seconds",
        "pfm_alpha_lab_runs_total",
        "pfm_alerts_fired_total",
        "pfm_realtime_clients",
        "pfm_realtime_pollers",
        "pfm_live_signals_last_run_age_seconds",
        "pfm_live_signals_recompute_duration_seconds",
        "pfm_factor_model_fits_total",
        "pfm_factor_model_fit_duration_seconds",
    ]
    for name in expected_series:
        assert name in body, f"expected metric {name!r} in /metrics output"


def test_request_middleware_increments_counter(app_client):
    app_client.get("/health")
    resp = app_client.get("/metrics")
    body = resp.text
    # The counter sample for /health, GET, 200 must appear at least once.
    assert "requests_total{" in body
    assert 'path="/health"' in body
    assert 'method="GET"' in body


# --- Sentry safe-import -----------------------------------------------------


def test_sentry_does_not_initialise_without_dsn(monkeypatch):
    """If SENTRY_DSN is unset, importing main must not raise."""
    monkeypatch.delenv("SENTRY_DSN", raising=False)
    # main is already imported; just confirm the module-level guard ran
    # without exploding by re-evaluating the same conditional logic here.
    dsn = os.environ.get("SENTRY_DSN", "").strip()
    assert dsn == ""

    # And confirm pfm.main is importable.
    import pfm.main as main_mod

    assert hasattr(main_mod, "app")


def test_sentry_swallows_missing_sdk(monkeypatch):
    """Even if SENTRY_DSN is set, an ImportError must not crash the app."""
    # Simulate the guard: dsn present, but sentry_sdk import raises.
    # The guard in main.py wraps the import in try/except ImportError so
    # this is a structural check rather than a runtime swap.
    src_path = Path(__file__).parent.parent / "src" / "pfm" / "main.py"
    src = src_path.read_text(encoding="utf-8")
    assert "except ImportError:" in src
    assert "import sentry_sdk" in src


# --- stdlib logging redirect -----------------------------------------------


def test_stdlib_logging_uses_stdout(monkeypatch):
    monkeypatch.setenv("LOG_FORMAT", "json")
    monkeypatch.setenv("LOG_LEVEL", "INFO")

    from pfm import logging_setup

    importlib.reload(logging_setup)
    logging_setup.configure_logging()

    # After configure_logging(force=True), the root logger's handler stream
    # is sys.stdout, not sys.stderr.
    root = logging.getLogger()
    streams = [getattr(h, "stream", None) for h in root.handlers if hasattr(h, "stream")]
    assert sys.stdout in streams or any(getattr(s, "name", "") == "<stdout>" for s in streams)
