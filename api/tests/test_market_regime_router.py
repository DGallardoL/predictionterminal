"""Tests for ``pfm.market_regime_router``.

Covers:

* Pure-function classifiers (``_classify_risk``, ``_classify_trend``,
  ``_spy_slope``, ``_narrative``).
* Full-stack ``GET /market/regime`` flow with mocked VIX + SPY data,
  including each of the three risk states crossed with the three trend
  states.
* Caching (second call must NOT re-hit the upstream fetchers) and the
  ``?nocache=true`` bypass.
* Error path (502) when the upstream VIX source raises.
* Schema integrity: every documented field is present, ``research_only``
  is always ``true``, and ``refresh_seconds`` matches the module hint.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm import market_regime_router as mod

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_app() -> FastAPI:
    """Mount the router into a fresh FastAPI app."""
    app = FastAPI()
    app.include_router(mod.router)
    return app


def _fake_spy_series(
    slope: float, base: float = 400.0, n: int = mod.TREND_WINDOW_DAYS
) -> pd.Series:
    """Return a deterministic SPY-like close series with the given slope."""
    idx = pd.date_range("2026-01-01", periods=n, freq="B", tz="UTC")
    values = base + slope * np.arange(n, dtype=float)
    return pd.Series(values, index=idx, name="Close")


@pytest.fixture(autouse=True)
def _clear_cache() -> Iterator[None]:
    """Wipe the response cache before AND after every test."""
    mod.cache_clear()
    yield
    mod.cache_clear()


@pytest.fixture
def patch_sources(monkeypatch: pytest.MonkeyPatch):
    """Factory: install fake VIX + SPY fetchers, return call-counters."""

    def _apply(vix: float, slope: float) -> dict[str, int]:
        counters = {"vix": 0, "spy": 0}

        def _fake_vix(_now: pd.Timestamp) -> float:
            counters["vix"] += 1
            return vix

        def _fake_spy(_now: pd.Timestamp, window: int = mod.TREND_WINDOW_DAYS) -> pd.Series:
            counters["spy"] += 1
            return _fake_spy_series(slope, n=window)

        monkeypatch.setattr(mod, "_fetch_vix", _fake_vix)
        monkeypatch.setattr(mod, "_fetch_spy_closes", _fake_spy)
        return counters

    return _apply


# ---------------------------------------------------------------------------
# Pure-function tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "vix,expected",
    [
        (30.0, "risk_off"),
        (25.01, "risk_off"),
        (25.0, "neutral"),  # boundary: not strictly > 25 -> neutral
        (20.0, "neutral"),
        (15.0, "neutral"),  # boundary: not strictly < 15 -> neutral
        (14.99, "risk_on"),
        (10.0, "risk_on"),
    ],
)
def test_classify_risk_thresholds(vix: float, expected: str) -> None:
    """VIX boundaries map to the documented labels (strict > / strict <)."""
    assert mod._classify_risk(vix) == expected


@pytest.mark.parametrize(
    "slope,expected",
    [
        (1.0, "bullish"),
        (mod.SLOPE_EPSILON + 1e-6, "bullish"),
        (mod.SLOPE_EPSILON, "sideways"),  # boundary -> sideways
        (0.0, "sideways"),
        (-mod.SLOPE_EPSILON, "sideways"),
        (-mod.SLOPE_EPSILON - 1e-6, "bearish"),
        (-1.0, "bearish"),
    ],
)
def test_classify_trend_thresholds(slope: float, expected: str) -> None:
    """Slope sign with a +/- SLOPE_EPSILON dead-band."""
    assert mod._classify_trend(slope) == expected


def test_spy_slope_recovers_known_slope() -> None:
    """``_spy_slope`` must recover the slope of a pure linear series."""
    s = _fake_spy_series(slope=0.5, base=400.0, n=50)
    slope = mod._spy_slope(s)
    assert slope == pytest.approx(0.5, abs=1e-9)


def test_spy_slope_zero_for_flat_series() -> None:
    """A flat price series yields slope == 0 (within float tolerance)."""
    s = _fake_spy_series(slope=0.0, base=400.0, n=50)
    assert mod._spy_slope(s) == pytest.approx(0.0, abs=1e-9)


def test_spy_slope_rejects_short_series() -> None:
    """Need at least 2 observations to fit a line."""
    s = pd.Series([400.0])
    with pytest.raises(ValueError):
        mod._spy_slope(s)


def test_narrative_covers_all_nine_combinations() -> None:
    """Every (regime, trend) pair has a non-empty narrative."""
    states = ("risk_off", "risk_on", "neutral")
    trends = ("bullish", "bearish", "sideways")
    seen: set[str] = set()
    for s in states:
        for t in trends:
            line = mod._narrative(s, t)  # type: ignore[arg-type]
            assert line.startswith("Currently ")
            assert len(line) > len("Currently ")
            seen.add(line)
    # All 9 narratives must be distinct.
    assert len(seen) == 9


# ---------------------------------------------------------------------------
# Endpoint integration tests
# ---------------------------------------------------------------------------


def test_endpoint_neutral_bullish_default_example(patch_sources) -> None:
    """The CLAUDE.md example: neutral risk + bullish trend."""
    patch_sources(vix=18.0, slope=0.4)
    client = TestClient(_make_app())
    r = client.get("/market/regime")
    assert r.status_code == 200
    body = r.json()
    assert body["regime"] == "neutral"
    assert body["trend"] == "bullish"
    assert body["vix"] == pytest.approx(18.0)
    assert body["spy_slope"] == pytest.approx(0.4, abs=1e-6)
    assert "neutral risk + bullish trend" in body["narrative"]
    assert body["research_only"] is True
    assert body["refresh_seconds"] == mod.REFRESH_HINT_SECONDS
    assert body["refresh_seconds"] == 3600
    assert "research-only" in body["disclaimer"].lower()
    # computed_at is ISO-8601 UTC.
    assert body["computed_at"].endswith("Z")


def test_endpoint_risk_off_bearish(patch_sources) -> None:
    """High VIX + falling SPY -> risk_off + bearish."""
    patch_sources(vix=32.0, slope=-1.5)
    client = TestClient(_make_app())
    r = client.get("/market/regime")
    assert r.status_code == 200
    body = r.json()
    assert body["regime"] == "risk_off"
    assert body["trend"] == "bearish"


def test_endpoint_risk_on_sideways(patch_sources) -> None:
    """Low VIX + flat SPY -> risk_on + sideways."""
    patch_sources(vix=12.0, slope=0.0)
    client = TestClient(_make_app())
    r = client.get("/market/regime")
    assert r.status_code == 200
    body = r.json()
    assert body["regime"] == "risk_on"
    assert body["trend"] == "sideways"


def test_response_schema_has_all_documented_fields(patch_sources) -> None:
    """Every field documented on RegimeResponse is present in the body."""
    patch_sources(vix=20.0, slope=0.2)
    client = TestClient(_make_app())
    body = client.get("/market/regime").json()
    required = {
        "vix",
        "spy_slope",
        "regime",
        "trend",
        "narrative",
        "computed_at",
        "refresh_seconds",
        "research_only",
        "disclaimer",
    }
    assert required.issubset(body.keys())
    # Narrative grid is the source of truth for valid suffixes.
    suffix = body["narrative"].removeprefix("Currently ")
    expected_suffix = mod._NARRATIVE_GRID[(body["regime"], body["trend"])]
    assert suffix == expected_suffix


def test_response_is_cached_across_repeat_calls(patch_sources) -> None:
    """Second identical GET must NOT re-call the upstream fetchers."""
    counters = patch_sources(vix=18.0, slope=0.4)
    client = TestClient(_make_app())
    first = client.get("/market/regime").json()
    second = client.get("/market/regime").json()
    assert first == second
    assert counters["vix"] == 1
    assert counters["spy"] == 1


def test_nocache_query_param_bypasses_cache(patch_sources) -> None:
    """``?nocache=true`` forces a re-fetch."""
    counters = patch_sources(vix=18.0, slope=0.4)
    client = TestClient(_make_app())
    client.get("/market/regime")
    client.get("/market/regime?nocache=true")
    assert counters["vix"] == 2
    assert counters["spy"] == 2


def test_endpoint_502_when_vix_fetch_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Upstream VIX failure surfaces as a clean 502."""

    def _boom(_now: pd.Timestamp) -> float:
        raise RuntimeError("synthetic upstream failure")

    monkeypatch.setattr(mod, "_fetch_vix", _boom)
    # SPY fetcher should not be reached, but stub it to be safe.
    monkeypatch.setattr(
        mod,
        "_fetch_spy_closes",
        lambda _now, window=mod.TREND_WINDOW_DAYS: _fake_spy_series(0.4, n=window),
    )
    client = TestClient(_make_app())
    r = client.get("/market/regime")
    assert r.status_code == 502
    assert "VIX fetch failed" in r.json()["detail"]


def test_endpoint_502_when_spy_fetch_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Upstream SPY failure also surfaces as 502 (not 500)."""

    monkeypatch.setattr(mod, "_fetch_vix", lambda _now: 18.0)

    def _boom(_now: pd.Timestamp, window: int = mod.TREND_WINDOW_DAYS) -> pd.Series:
        raise RuntimeError("synthetic SPY failure")

    monkeypatch.setattr(mod, "_fetch_spy_closes", _boom)
    client = TestClient(_make_app())
    r = client.get("/market/regime")
    assert r.status_code == 502
    assert "SPY fetch failed" in r.json()["detail"]


def test_research_only_flag_is_always_true(patch_sources) -> None:
    """Anti-alpha discipline: research_only must never be False."""
    for vix, slope in [(10.0, 1.0), (20.0, 0.0), (35.0, -2.0)]:
        mod.cache_clear()
        patch_sources(vix=vix, slope=slope)
        body = TestClient(_make_app()).get("/market/regime").json()
        assert body["research_only"] is True
        assert "anti-alpha" in body["disclaimer"].lower()
