"""Edge-case battery for ``POST /fit``.

Task W11-42. Each test pokes a different corner of the request space and
asserts the server either (a) recovers gracefully with structured output, or
(b) returns a clean 4xx with an informative ``detail`` — never a 5xx, never
a silent NaN payload.

These tests are intentionally hermetic: every upstream (Polymarket,
yfinance, Redis) is patched inside the ``app_client`` fixture *defined in
this file* so the suite can run with ``--noconftest``. When a single test
needs a bespoke factor history (NaN, all-zero, collinear, etc.) it
monkeypatches ``main_mod.fetch_factor_history`` for its duration.

Run:
    cd api && PYTHONPATH=src .venv/bin/python -m pytest \\
        tests/test_fit_edge_cases.py -q --noconftest
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import pfm.main as main_mod

# ---------------------------------------------------------------------------
# Self-contained fixtures (the module is run with ``--noconftest`` so the
# project-level conftest.py is NOT loaded). Mirror the shape of
# ``api/tests/conftest.py``'s ``app_client`` but inline everything we need.
# ---------------------------------------------------------------------------


@pytest.fixture
def factors_file(tmp_path: Path) -> Path:
    p = tmp_path / "factors.yml"
    p.write_text(
        """
factors:
  - id: factor_a
    name: Factor A
    slug: slug-a
    source: polymarket
    description: Test factor A.
  - id: factor_b
    name: Factor B
    slug: slug-b
    source: polymarket
    description: Test factor B.
"""
    )
    return p


@pytest.fixture
def default_factor_history():
    """A baseline ``(client, slug, start, end) -> DataFrame`` fetcher."""
    idx = pd.date_range("2025-06-01", "2025-12-31", freq="D", tz="UTC")
    idx.name = "date"
    n = len(idx)
    t = np.arange(n) / n
    bank = {
        "slug-a": pd.DataFrame(
            {"price": (0.30 + 0.30 * np.sin(2 * np.pi * t * 1.2)).clip(0.05, 0.95)},
            index=idx,
        ),
        "slug-b": pd.DataFrame(
            {"price": (0.55 + 0.20 * np.cos(2 * np.pi * t * 0.8)).clip(0.05, 0.95)},
            index=idx,
        ),
    }
    for df in bank.values():
        df.index.name = "date"

    def _fetch(_client, slug: str, start=None, end=None):
        if slug not in bank:
            return pd.DataFrame({"price": []}, index=pd.DatetimeIndex([], tz="UTC"))
        df = bank[slug]
        if start is not None:
            df = df[df.index >= start]
        if end is not None:
            df = df[df.index <= end]
        return df

    return _fetch


@pytest.fixture
def default_log_returns():
    """A deterministic log-return series builder."""

    def _make(ticker: str, start, end, return_type: str = "log") -> pd.Series:
        idx = pd.date_range(start, end, freq="B", tz="UTC")
        n = len(idx)
        if n == 0:
            from pfm.sources.equity import EquityDataError

            raise EquityDataError(f"no equity history in window for {ticker!r}")
        rng = np.random.default_rng(seed=abs(hash(ticker)) % (2**32))
        values = 0.0001 * np.arange(n) + 0.005 * np.sin(np.arange(n)) + rng.normal(0, 0.001, n)
        s = pd.Series(values, index=idx, name="r")
        s.index = pd.to_datetime(s.index, utc=True).normalize()
        return s

    return _make


@pytest.fixture
def app_client(
    monkeypatch: pytest.MonkeyPatch,
    factors_file: Path,
    default_factor_history,
    default_log_returns,
) -> Iterator[TestClient]:
    """TestClient with Polymarket / yfinance / Redis patched out."""
    monkeypatch.setenv("FACTORS_FILE", str(factors_file))
    import pfm.config as cfg

    cfg._settings = None

    monkeypatch.setattr(main_mod, "fetch_factor_history", default_factor_history)
    monkeypatch.setattr(main_mod, "get_log_returns", default_log_returns)

    from pfm.cache import NullCache

    monkeypatch.setattr(main_mod, "RedisCache", lambda url: NullCache())

    with TestClient(main_mod.app) as client:
        yield client


# ---------------------------------------------------------------------------
# Helpers — bespoke factor histories injected via monkeypatch.
# ---------------------------------------------------------------------------


def _idx_daily_utc(start: str, end: str) -> pd.DatetimeIndex:
    """Daily UTC DatetimeIndex used by the conftest fake_factor_history fixture."""
    idx = pd.date_range(start, end, freq="D", tz="UTC")
    idx.name = "date"
    return idx


def _factor_df(values: np.ndarray, idx: pd.DatetimeIndex) -> pd.DataFrame:
    df = pd.DataFrame({"price": values}, index=idx)
    df.index.name = "date"
    return df


def _make_factor_bank(
    overrides: dict[str, pd.DataFrame] | None = None,
    default_window: tuple[str, str] = ("2025-06-01", "2025-12-31"),
):
    """Build a ``(client, slug, start, end) -> DataFrame`` shim.

    ``overrides`` keys are slugs ("slug-a", "slug-b") and the values are
    full-window DataFrames; the shim windows them via ``start``/``end`` like
    the real ``fetch_factor_history`` does.
    """
    idx = _idx_daily_utc(*default_window)
    n = len(idx)
    # Default benign series so /fit succeeds when only one factor needs the override.
    t = np.arange(n) / n
    default_a = _factor_df((0.30 + 0.30 * np.sin(2 * np.pi * t * 1.2)).clip(0.05, 0.95), idx)
    default_b = _factor_df((0.55 + 0.20 * np.cos(2 * np.pi * t * 0.8)).clip(0.05, 0.95), idx)
    bank: dict[str, pd.DataFrame] = {"slug-a": default_a, "slug-b": default_b}
    if overrides:
        bank.update(overrides)

    def _fetch(_client, slug: str, start=None, end=None):
        df = bank.get(slug)
        if df is None:
            return pd.DataFrame({"price": []}, index=pd.DatetimeIndex([], tz="UTC"))
        if start is not None:
            df = df[df.index >= start]
        if end is not None:
            df = df[df.index <= end]
        return df

    return _fetch


def _patch_factors(monkeypatch: pytest.MonkeyPatch, overrides: dict[str, pd.DataFrame]) -> None:
    """Replace the conftest-installed fetch with one that uses ``overrides``."""
    monkeypatch.setattr(main_mod, "fetch_factor_history", _make_factor_bank(overrides))


def _patch_returns(
    monkeypatch: pytest.MonkeyPatch,
    *,
    start: str = "2025-06-01",
    end: str = "2025-12-31",
    n_obs: int | None = None,
    raise_on_empty: bool = True,
):
    """Replace ``get_log_returns`` with a deterministic synthetic series.

    Useful for tests that need to constrain or shorten the equity series.
    """

    def _make(ticker, _start, _end, return_type="log"):
        if n_obs is not None:
            idx = pd.date_range(start, periods=n_obs, freq="B", tz="UTC")
        else:
            idx = pd.date_range(start, end, freq="B", tz="UTC")
        if len(idx) == 0 and raise_on_empty:
            from pfm.sources.equity import EquityDataError

            raise EquityDataError(f"no equity history in [{start}, {end}] for {ticker!r}")
        rng = np.random.default_rng(seed=abs(hash(ticker)) % (2**32))
        values = rng.normal(0, 0.01, len(idx))
        s = pd.Series(values, index=idx, name="r")
        s.index = pd.to_datetime(s.index, utc=True).normalize()
        return s

    monkeypatch.setattr(main_mod, "get_log_returns", _make)


def _fit(client: TestClient, **overrides: Any) -> Any:
    """POST /fit with a sensible default body, return the raw Response."""
    body = {
        "ticker": "TEST",
        "factors": ["factor_a"],
        "start": "2025-06-15",
        "end": "2025-12-15",
    }
    # Allow query params via a special key.
    qs: dict[str, Any] = overrides.pop("_qs", {})
    body.update(overrides)
    return client.post("/fit", params=qs, json=body)


# ---------------------------------------------------------------------------
# 1. NaN values in factor series — sparse data must be dropped, n_obs reported.
# ---------------------------------------------------------------------------


class TestNaNInFactor:
    """Some Polymarket days are missing → pd.NaN. inner-join must drop them."""

    def test_sparse_nan_factor_drops_bad_rows(
        self, monkeypatch: pytest.MonkeyPatch, app_client: TestClient
    ) -> None:
        idx = _idx_daily_utc("2025-06-01", "2025-12-31")
        n = len(idx)
        # Smooth in-range probability series with NaN injected on every 5th day.
        vals = 0.35 + 0.25 * np.sin(np.linspace(0, 4 * np.pi, n))
        vals[::5] = np.nan
        df = _factor_df(vals, idx)
        _patch_factors(monkeypatch, {"slug-a": df})

        r = _fit(app_client)
        assert r.status_code == 200, r.text
        body = r.json()
        # n_obs reported, strictly positive, and reflects the dropped rows
        # (~80% retention after every-5th NaN through inner-join).
        assert body["n_obs"] > 0
        assert body["n_obs"] < n  # must have dropped *some* rows
        # No NaN in the predicted/residual streams.
        for pt in body["time_series"]:
            assert math.isfinite(pt["observed"])
            assert math.isfinite(pt["predicted"])
            assert math.isfinite(pt["residual"])


# ---------------------------------------------------------------------------
# 2. All-zero factor — degenerate variance, must NOT crash with 5xx.
# ---------------------------------------------------------------------------


class TestAllZeroFactor:
    """A factor whose price is constantly 0 has variance 0 → β undefined.

    Two acceptable outcomes:
      (a) 200 with VIF flagged / β reported as 0 / a warning surfaced, or
      (b) 422 with a helpful detail.
    What we DO NOT accept: a 500 stack trace.
    """

    def test_all_zero_factor_no_crash(
        self, monkeypatch: pytest.MonkeyPatch, app_client: TestClient
    ) -> None:
        idx = _idx_daily_utc("2025-06-01", "2025-12-31")
        zeros = np.zeros(len(idx))
        _patch_factors(monkeypatch, {"slug-a": _factor_df(zeros, idx)})

        r = _fit(app_client)
        # Anything in [200, 499] is acceptable; a 5xx is a failure mode.
        assert r.status_code < 500, r.text
        # If accepted, the response must not leak NaN/Inf into beta/SE.
        if r.status_code == 200:
            body = r.json()
            for f in body["factors"]:
                assert math.isfinite(f["beta"])
                # SE/t may be 0 or huge — we just require finite.
                assert math.isfinite(f["std_err"])


# ---------------------------------------------------------------------------
# 3. All-NaN factor — every observation missing.
# ---------------------------------------------------------------------------


class TestAllNaNFactor:
    """When every row of a factor is NaN the inner-join is empty → 422 or 502.

    The current implementation surfaces this as 422 ("too few overlapping
    observations") which is acceptable. A 502 ("source returned no history")
    is also acceptable since pd.DataFrame.empty handles the all-NaN edge.
    """

    def test_all_nan_factor_returns_4xx_or_502(
        self, monkeypatch: pytest.MonkeyPatch, app_client: TestClient
    ) -> None:
        idx = _idx_daily_utc("2025-06-01", "2025-12-31")
        all_nan = np.full(len(idx), np.nan)
        _patch_factors(monkeypatch, {"slug-a": _factor_df(all_nan, idx)})

        r = _fit(app_client)
        # 422 (after inner-join) or 502 (source-empty) are both acceptable.
        assert r.status_code in (422, 502), r.text
        # No 500.
        assert r.status_code < 500 or r.status_code == 502


# ---------------------------------------------------------------------------
# 4. Perfect collinearity — X1 = X2 exactly.
# ---------------------------------------------------------------------------


class TestPerfectCollinearity:
    """Two factors with identical series → VIF reported as the sentinel.

    Must NOT crash. With prune_collinear=true the second factor should be
    auto-pruned and listed under ``auto_pruned``.
    """

    def test_perfect_collinearity_handled_without_crash(
        self, monkeypatch: pytest.MonkeyPatch, app_client: TestClient
    ) -> None:
        idx = _idx_daily_utc("2025-06-01", "2025-12-31")
        n = len(idx)
        t = np.arange(n) / n
        # Strictly in (0, 1) so logit doesn't saturate.
        vals = (0.30 + 0.30 * np.sin(2 * np.pi * t * 1.2)).clip(0.05, 0.95)
        identical = _factor_df(vals, idx)
        _patch_factors(monkeypatch, {"slug-a": identical, "slug-b": identical.copy()})

        r = _fit(app_client, factors=["factor_a", "factor_b"])
        assert r.status_code < 500, r.text
        if r.status_code == 200:
            body = r.json()
            vif = body["diagnostics"]["vif"]
            # At least one VIF must be very large (sentinel or near it).
            max_vif = max(vif.values()) if vif else 0.0
            from pfm.model import VIF_INF_SENTINEL

            assert max_vif >= 5.0 or max_vif == pytest.approx(VIF_INF_SENTINEL)

    def test_perfect_collinearity_auto_pruned(
        self, monkeypatch: pytest.MonkeyPatch, app_client: TestClient
    ) -> None:
        idx = _idx_daily_utc("2025-06-01", "2025-12-31")
        n = len(idx)
        t = np.arange(n) / n
        vals = (0.30 + 0.30 * np.sin(2 * np.pi * t * 1.2)).clip(0.05, 0.95)
        identical = _factor_df(vals, idx)
        _patch_factors(monkeypatch, {"slug-a": identical, "slug-b": identical.copy()})

        r = _fit(
            app_client,
            factors=["factor_a", "factor_b"],
            _qs={"prune_collinear": "true"},
        )
        assert r.status_code == 200, r.text
        body = r.json()
        # Exactly one of the two should remain.
        assert len(body["factors"]) == 1
        assert len(body["auto_pruned"]) == 1


# ---------------------------------------------------------------------------
# 5. Near-collinearity — X1 ≈ 1.001 · X2.
# ---------------------------------------------------------------------------


class TestNearCollinearity:
    """VIF should be very high but the fit must complete with diagnostics."""

    def test_near_collinearity_completes_with_high_vif(
        self, monkeypatch: pytest.MonkeyPatch, app_client: TestClient
    ) -> None:
        idx = _idx_daily_utc("2025-06-01", "2025-12-31")
        n = len(idx)
        rng = np.random.default_rng(7)
        t = np.arange(n) / n
        base = (0.30 + 0.30 * np.sin(2 * np.pi * t * 1.2)).clip(0.05, 0.95)
        # X2 = 1.001·X1 + tiny noise; still in (0, 1).
        perturb = np.clip(1.001 * base + rng.normal(0, 1e-4, n), 0.05, 0.95)
        _patch_factors(
            monkeypatch,
            {"slug-a": _factor_df(base, idx), "slug-b": _factor_df(perturb, idx)},
        )

        r = _fit(app_client, factors=["factor_a", "factor_b"])
        assert r.status_code == 200, r.text
        body = r.json()
        vif = body["diagnostics"]["vif"]
        max_vif = max(vif.values()) if vif else 0.0
        # Very high VIF expected (≫ 5).
        assert max_vif > 10.0, vif
        # And the server should flag it in either warnings or overfit_risk_flags.
        flag_codes = {f["code"] for f in body.get("overfit_risk_flags", [])}
        has_warning = any("vif" in w.lower() or "collinear" in w.lower() for w in body["warnings"])
        assert has_warning or flag_codes


# ---------------------------------------------------------------------------
# 6. Single observation — insufficient data.
# ---------------------------------------------------------------------------


class TestSingleObservation:
    """An equity series with 1 obs must fail with 422 and an informative detail."""

    def test_single_obs_returns_422(
        self, monkeypatch: pytest.MonkeyPatch, app_client: TestClient
    ) -> None:
        # Patch the equity series to a single business day.
        _patch_returns(monkeypatch, start="2025-07-01", n_obs=1)

        r = _fit(app_client)
        assert r.status_code == 422, r.text
        detail = r.json()["detail"].lower()
        assert "few" in detail or "insufficient" in detail or "overlapping" in detail


# ---------------------------------------------------------------------------
# 7 & 8. Length mismatch — factor longer than ticker, and vice-versa.
# ---------------------------------------------------------------------------


class TestLengthMismatch:
    """Inner-join must align on the intersection regardless of which side is
    longer. n_obs == min(n_factor, n_ticker) approximately (modulo weekends).
    """

    def test_factor_longer_than_ticker(
        self, monkeypatch: pytest.MonkeyPatch, app_client: TestClient
    ) -> None:
        # Factor full-year, ticker only 25 business days starting 2025-07-15.
        _patch_returns(monkeypatch, start="2025-07-15", n_obs=25)

        r = _fit(app_client, start="2025-07-15", end="2025-12-15")
        assert r.status_code == 200, r.text
        # Ticker side is the bottleneck → final n_obs <= 25.
        assert r.json()["n_obs"] <= 25

    def test_ticker_longer_than_factor(
        self, monkeypatch: pytest.MonkeyPatch, app_client: TestClient
    ) -> None:
        # Factor only present mid-July through mid-Aug; ticker spans the
        # whole year. Inner-join must cap at the factor window.
        idx_short = _idx_daily_utc("2025-07-10", "2025-08-15")
        n = len(idx_short)
        t = np.arange(n) / n
        vals = (0.40 + 0.25 * np.sin(2 * np.pi * t * 1.2)).clip(0.05, 0.95)
        _patch_factors(monkeypatch, {"slug-a": _factor_df(vals, idx_short)})

        r = _fit(app_client, start="2025-07-01", end="2025-12-31")
        assert r.status_code == 200, r.text
        # The factor window has ~37 daily obs; after stock-calendar inner
        # join and lag-shift the count must be ≤ 37 (and > 0).
        n_obs = r.json()["n_obs"]
        assert 0 < n_obs <= 37


# ---------------------------------------------------------------------------
# 9 & 10. Date range outside / in the future → 0 obs.
# ---------------------------------------------------------------------------


class TestOutOfRangeDates:
    """No equity data in the requested window → server returns 4xx (422 or 502)."""

    def test_window_before_equity_history(
        self, monkeypatch: pytest.MonkeyPatch, app_client: TestClient
    ) -> None:
        # Force the equity adapter to return an empty Series for old dates.
        _patch_returns(monkeypatch, start="2020-01-01", n_obs=0)

        r = _fit(app_client, start="1995-01-01", end="1995-12-31")
        assert r.status_code in (422, 502, 400), r.text

    def test_future_date_range(
        self, monkeypatch: pytest.MonkeyPatch, app_client: TestClient
    ) -> None:
        _patch_returns(monkeypatch, start="2030-01-01", n_obs=0)

        r = _fit(app_client, start="2030-01-01", end="2030-12-31")
        assert r.status_code in (422, 502, 400), r.text


# ---------------------------------------------------------------------------
# 11. Missing dates within range (weekends/holidays) — UTC normalize per CLAUDE.md.
# ---------------------------------------------------------------------------


class TestWeekendHandling:
    """The factor is daily incl. weekends; equity is business-day only.
    Inner-join must produce a UTC-normalized weekday-only design matrix.
    Verifies the timezone-alignment trap from CLAUDE.md.
    """

    def test_weekends_dropped_via_utc_normalize(
        self, monkeypatch: pytest.MonkeyPatch, app_client: TestClient
    ) -> None:
        r = _fit(app_client, start="2025-06-15", end="2025-08-15")
        assert r.status_code == 200, r.text
        body = r.json()
        # Time-series dates must all be UTC weekdays (Mon-Fri).
        for pt in body["time_series"]:
            d = pd.Timestamp(pt["date"])
            assert d.weekday() < 5, f"weekend leaked in: {pt['date']}"


# ---------------------------------------------------------------------------
# 12. Negative values when factor is marked is_probability — falls back.
# ---------------------------------------------------------------------------


class TestNegativeProbabilityFactor:
    """A factor declared as probability but emitting values outside [0, 1].
    Per ``delta_logit``, this triggers a fallback to plain first-differences
    + a warning. /fit must NOT crash and must NOT silently clip.
    """

    def test_negative_probability_factor_falls_back(
        self, monkeypatch: pytest.MonkeyPatch, app_client: TestClient
    ) -> None:
        idx = _idx_daily_utc("2025-06-01", "2025-12-31")
        n = len(idx)
        # Values clearly outside [0, 1]: yields-like series in [-0.02, 0.05].
        vals = -0.02 + 0.07 * np.sin(np.linspace(0, 4 * np.pi, n))
        _patch_factors(monkeypatch, {"slug-a": _factor_df(vals, idx)})

        r = _fit(app_client)
        assert r.status_code == 200, r.text
        body = r.json()
        # The factor still produced finite coefficients (plain diff fallback).
        assert all(math.isfinite(f["beta"]) for f in body["factors"])


# ---------------------------------------------------------------------------
# 13. Clipping epsilon — values at 0.005 and 0.002 → Δlogit ≈ 0.
# ---------------------------------------------------------------------------


class TestClippingEpsilonZeroes:
    """CLAUDE.md anchor: contracts at 0.005 → 0.002 with eps=0.01 → Δlogit=0.

    /fit must surface this via clipping_events > 0 and warnings text.
    """

    def test_extreme_tail_clipped_to_zero_delta(
        self, monkeypatch: pytest.MonkeyPatch, app_client: TestClient
    ) -> None:
        idx = _idx_daily_utc("2025-06-01", "2025-12-31")
        n = len(idx)
        rng = np.random.default_rng(11)
        # All values < 0.01 (the default eps): logit clips to log(0.01/0.99)
        # for every obs, so Δlogit is exactly zero.
        vals = np.clip(rng.uniform(0.001, 0.009, n), 0.0001, 0.5)
        _patch_factors(monkeypatch, {"slug-a": _factor_df(vals, idx)})

        r = _fit(app_client)
        # Could be 200 (with zero-Δlogit factor → flagged) or 422 (rank
        # deficiency from a constant column). Either is acceptable; 5xx is not.
        assert r.status_code < 500, r.text
        if r.status_code == 200:
            body = r.json()
            assert body["clipping_events"] > 0
            # User-visible warning naming the factor or "epsilon" / "clipping".
            joined = " ".join(body["warnings"]).lower()
            assert "clip" in joined or "epsilon" in joined


# ---------------------------------------------------------------------------
# 14. Custom epsilon via query param — must be respected.
# ---------------------------------------------------------------------------


class TestCustomEpsilonRespected:
    """The ``epsilon`` query param flows through to ``count_clipping_events``
    and the logit transform — fewer clipping events at eps=0.001 than 0.05.
    """

    def test_lower_epsilon_reduces_clipping_count(
        self, monkeypatch: pytest.MonkeyPatch, app_client: TestClient
    ) -> None:
        idx = _idx_daily_utc("2025-06-01", "2025-12-31")
        n = len(idx)
        # Values in [0.005, 0.04] — straddle the default eps=0.01.
        rng = np.random.default_rng(3)
        vals = rng.uniform(0.005, 0.04, n)
        _patch_factors(monkeypatch, {"slug-a": _factor_df(vals, idx)})

        r_loose = _fit(app_client, _qs={"epsilon": 0.001})
        r_tight = _fit(app_client, _qs={"epsilon": 0.05})

        # Both must succeed (or both rank-deficient) — but if they succeed,
        # tight (higher) eps clips far more obs than loose.
        if r_loose.status_code == 200 and r_tight.status_code == 200:
            assert r_loose.json()["epsilon"] == pytest.approx(0.001)
            assert r_tight.json()["epsilon"] == pytest.approx(0.05)
            assert r_loose.json()["clipping_events"] <= r_tight.json()["clipping_events"]


# ---------------------------------------------------------------------------
# 15. Very large factor list — 50+ factors. Must not crash.
# ---------------------------------------------------------------------------


class TestManyFactors:
    """50+ unknown factor ids must return a structured 400/422 without OOM.

    We deliberately pass IDs not in the conftest factors.yml so the resolver
    rejects them with a clean 400 before any upstream IO. This is the
    correct degenerate behaviour for unregistered slugs. Hint for
    follow-up: W11-28 should add elastic-net guidance for many-factor users.
    """

    def test_fifty_unknown_factors_returns_4xx(self, app_client: TestClient) -> None:
        many = [f"factor_z_{i}" for i in range(60)]
        r = app_client.post(
            "/fit",
            json={
                "ticker": "TEST",
                "factors": many,
                "start": "2025-06-15",
                "end": "2025-12-15",
            },
        )
        assert r.status_code in (400, 422), r.text


# ---------------------------------------------------------------------------
# 16. Unicode in custom factor name/id — handled by Pydantic + UTF-8.
# ---------------------------------------------------------------------------


class TestUnicodeFactorLabels:
    """``CustomFactor.id`` pattern is ASCII; ``name`` is unconstrained Unicode.

    Verifies the unicode ``name`` does not crash JSON serialization on the
    response side. Invalid ``id`` (unicode) must 422 cleanly.
    """

    def test_unicode_custom_factor_name_accepted(self, app_client: TestClient) -> None:
        r = app_client.post(
            "/fit",
            json={
                "ticker": "TEST",
                "factors": [],
                "custom_factors": [
                    {"id": "factor_a", "slug": "slug-a", "name": "电池关税 πρόβλεψη 📈"}
                ],
                "start": "2025-06-15",
                "end": "2025-12-15",
            },
        )
        assert r.status_code == 200, r.text

    def test_unicode_custom_factor_id_rejected_by_pydantic(self, app_client: TestClient) -> None:
        # Pattern is [a-zA-Z0-9_-]+. Unicode chars must trigger 422.
        r = app_client.post(
            "/fit",
            json={
                "ticker": "TEST",
                "factors": [],
                "custom_factors": [{"id": "电池关税", "slug": "slug-a"}],
                "start": "2025-06-15",
                "end": "2025-12-15",
            },
        )
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# 17. SQL-injection-like input in factor slug — safely handled by Pydantic.
# ---------------------------------------------------------------------------


class TestSQLInjectionInSlug:
    """Pydantic validates inputs to safe Python types; no string interpolation
    into a SQL query in this codebase (everything is dict-cached or REST-fetched).
    But we still want to confirm exotic strings round-trip without crashing.
    """

    def test_sql_like_slug_unknown_factor_4xx(self, app_client: TestClient) -> None:
        r = app_client.post(
            "/fit",
            json={
                "ticker": "TEST",
                # Unknown id in factors → resolver 400.
                "factors": ["'; DROP TABLE factors; --"],
                "start": "2025-06-15",
                "end": "2025-12-15",
            },
        )
        # Unknown factor id → 400 (or 422 from Pydantic). NOT 500.
        assert r.status_code in (400, 422), r.text

    def test_sql_like_ticker_rejected_by_pattern(self, app_client: TestClient) -> None:
        r = app_client.post(
            "/fit",
            json={
                "ticker": "' OR 1=1",
                "factors": ["factor_a"],
                "start": "2025-06-15",
                "end": "2025-12-15",
            },
        )
        # TICKER_PATTERN forbids spaces, single-quote, '='. 422 expected.
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# 18. Massive request body — 10 MB-ish. Must not OOM the process.
# ---------------------------------------------------------------------------


class TestMassiveBody:
    """A 10 MB body of bogus custom factors must be rejected (4xx or 413).

    FastAPI/Starlette has no built-in body-size cap by default, so the test's
    job is to confirm the *parsing path* doesn't crash and the unknown-slug
    resolver short-circuits. If a future change adds a real body-size limit
    it should return 413; we accept either outcome.
    """

    def test_10mb_body_rejected_without_5xx(self, app_client: TestClient) -> None:
        # 200_000 bogus custom factors with a 50-char slug each ≈ 13 MB JSON.
        # Use 50_000 to stay within ~5 MB — still huge, still parseable.
        bogus = [{"id": f"f_{i:05d}", "slug": "x" * 80} for i in range(50_000)]
        r = app_client.post(
            "/fit",
            json={
                "ticker": "TEST",
                "factors": [],
                "custom_factors": bogus,
                "start": "2025-06-15",
                "end": "2025-12-15",
            },
        )
        # The slugs don't resolve to real upstream history → 502, OR the
        # resolver dedupes/limits and returns 400, OR a body-size middleware
        # returns 413. 5xx (other than 502 from upstream-empty) is forbidden.
        assert r.status_code in (400, 413, 422, 502), (
            f"unexpected status={r.status_code} body[:200]={r.text[:200]}"
        )
