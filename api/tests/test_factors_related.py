"""Tests for ``GET /factors/{slug}/related`` (T29).

The router under test ships at ``pfm.factors_related_router`` (NOT
``pfm.factors.related_router`` — see the module docstring for the package-vs-
module reason). It is not wired into the running app yet (main.py:routes is
held by another active session), so each test mounts the router into a fresh
``FastAPI`` instance and stages factor history via the same monkeypatch hook
the rest of the regression suite uses (``pfm.regression_core.
_cached_factor_history``).
"""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm import factors_related_router as related_mod
from pfm.cache import NullCache
from pfm.config import Settings
from pfm.dependencies import (
    get_cache,
    get_factors_dep,
    get_polymarket_client,
)
from pfm.factors import FactorConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_factor(fid: str, slug: str) -> FactorConfig:
    return FactorConfig(
        id=fid,
        name=fid.replace("_", " ").title(),
        slug=slug,
        source="polymarket",
        description=f"Test factor {fid}",
        theme="test",
        is_probability=True,
    )


@pytest.fixture
def factor_catalog() -> dict[str, FactorConfig]:
    """A small synthetic catalog with one anchor and five candidates."""
    return {
        "anchor": _make_factor("anchor", "anchor-slug"),
        "c_pos": _make_factor("c_pos", "slug-pos"),
        "c_neg": _make_factor("c_neg", "slug-neg"),
        "c_mid": _make_factor("c_mid", "slug-mid"),
        "c_short": _make_factor("c_short", "slug-short"),
        "c_zero": _make_factor("c_zero", "slug-zero"),
    }


@pytest.fixture
def history_bank() -> dict[str, pd.DataFrame]:
    """Synthetic price history per slug, indexed by UTC midnight."""
    idx = pd.date_range("2026-03-01", periods=60, freq="D", tz="UTC")
    rng = np.random.default_rng(seed=42)
    anchor = pd.Series(np.linspace(0.30, 0.70, len(idx)), index=idx)
    # Perfectly anti-correlated (ρ ≈ -1) — but capped at finite p-value:
    neg = 1.0 - anchor
    # Strong positive ρ ≈ 0.95
    pos = anchor + rng.normal(0.0, 0.005, len(idx))
    # Mid ρ ≈ 0.5
    mid = 0.5 * anchor + rng.normal(0.0, 0.05, len(idx))
    # Short — only 10 obs, should be dropped (below MIN_OVERLAP_OBS=20)
    short = pd.Series(np.linspace(0.40, 0.50, 10), index=idx[:10])
    # Zero variance — constant, should also be dropped (ρ = NaN → 0)
    zero = pd.Series(np.full(len(idx), 0.5), index=idx)

    def _wrap(s: pd.Series) -> pd.DataFrame:
        df = pd.DataFrame({"price": s.values}, index=s.index)
        df.index.name = "date"
        return df

    return {
        "anchor-slug": _wrap(anchor),
        "slug-pos": _wrap(pos),
        "slug-neg": _wrap(neg),
        "slug-mid": _wrap(mid),
        "slug-short": _wrap(short),
        "slug-zero": _wrap(zero),
    }


@pytest.fixture
def mock_factor_history(
    monkeypatch: pytest.MonkeyPatch, history_bank: dict[str, pd.DataFrame]
) -> dict[str, int]:
    """Patch ``_cached_factor_history`` to return our synthetic bank.

    Returns a dict that counts calls per slug — handy for the cache-hit test.
    """
    call_counter: dict[str, int] = {}

    def _fake_cached(fc, start, end, poly, cache, settings):
        call_counter[fc.slug] = call_counter.get(fc.slug, 0) + 1
        df = history_bank.get(fc.slug)
        if df is None:
            return pd.DataFrame()
        return df

    # Patch the import the router uses (it does a deferred import inside
    # _fetch_series so we monkeypatch the source module).
    import pfm.regression_core as rc

    monkeypatch.setattr(rc, "_cached_factor_history", _fake_cached)
    return call_counter


@pytest.fixture
def client(factor_catalog: dict[str, FactorConfig]) -> Iterator[TestClient]:
    """Mount the router on a fresh FastAPI app with DI overridden."""
    # Always start each test from a cold cache so order doesn't matter.
    related_mod._cache_clear()

    app = FastAPI()
    app.include_router(related_mod.router)

    fake_settings = Settings(
        polymarket_gamma_url="http://gamma.test",
        polymarket_clob_url="http://clob.test",
    )
    fake_poly = MagicMock()

    app.dependency_overrides[get_factors_dep] = lambda: factor_catalog
    app.dependency_overrides[get_cache] = lambda: NullCache()
    app.dependency_overrides[get_polymarket_client] = lambda: fake_poly

    # Settings is also a Depends in the router signature; override via the
    # real ``get_settings`` callable from ``pfm.config``.
    from pfm.config import get_settings

    app.dependency_overrides[get_settings] = lambda: fake_settings

    with TestClient(app) as c:
        yield c

    related_mod._cache_clear()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_returns_top_n_sorted_by_abs_rho(
    client: TestClient, mock_factor_history: dict[str, int]
) -> None:
    """Highest |ρ| candidate first; short/zero dropped."""
    r = client.get("/factors/anchor-slug/related")
    assert r.status_code == 200, r.text
    payload = r.json()
    assert payload["anchor"] == "anchor-slug"
    assert payload["window_days"] == 30
    related = payload["related"]
    # c_short (n_obs<20) and c_zero (zero variance) must be dropped.
    slugs = [row["slug"] for row in related]
    assert "slug-short" not in slugs
    assert "slug-zero" not in slugs
    # c_neg and c_pos both have |ρ| ≈ 1; c_mid is lower.
    assert len(related) >= 2
    assert abs(related[0]["rho"]) >= abs(related[-1]["rho"])
    # Anchor itself must NEVER appear in the results.
    assert "anchor-slug" not in slugs


def test_unknown_anchor_returns_404(
    client: TestClient, mock_factor_history: dict[str, int]
) -> None:
    """Anchor slug not in factor catalog → 404."""
    r = client.get("/factors/does-not-exist/related")
    assert r.status_code == 404
    assert "not found" in r.json()["detail"].lower()


def test_anchor_with_no_other_factors(
    factor_catalog: dict[str, FactorConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When every non-anchor factor has empty history, return empty list (not 500)."""
    related_mod._cache_clear()
    app = FastAPI()
    app.include_router(related_mod.router)

    idx = pd.date_range("2026-03-01", periods=60, freq="D", tz="UTC")
    anchor_only = {
        "anchor-slug": pd.DataFrame({"price": np.linspace(0.3, 0.7, len(idx))}, index=idx)
    }

    def _fake_cached(fc, start, end, poly, cache, settings):
        df = anchor_only.get(fc.slug)
        return df if df is not None else pd.DataFrame()

    import pfm.regression_core as rc

    monkeypatch.setattr(rc, "_cached_factor_history", _fake_cached)

    fake_settings = Settings(
        polymarket_gamma_url="http://gamma.test",
        polymarket_clob_url="http://clob.test",
    )
    app.dependency_overrides[get_factors_dep] = lambda: factor_catalog
    app.dependency_overrides[get_cache] = lambda: NullCache()
    app.dependency_overrides[get_polymarket_client] = lambda: MagicMock()
    from pfm.config import get_settings

    app.dependency_overrides[get_settings] = lambda: fake_settings

    with TestClient(app) as c:
        r = c.get("/factors/anchor-slug/related")
    assert r.status_code == 200
    assert r.json()["related"] == []
    related_mod._cache_clear()


def test_cache_hit_does_not_refetch(
    client: TestClient, mock_factor_history: dict[str, int]
) -> None:
    """Second call within TTL must NOT re-invoke ``_cached_factor_history``."""
    r1 = client.get("/factors/anchor-slug/related")
    assert r1.status_code == 200
    calls_after_first = sum(mock_factor_history.values())
    assert calls_after_first > 0

    r2 = client.get("/factors/anchor-slug/related")
    assert r2.status_code == 200
    calls_after_second = sum(mock_factor_history.values())
    assert calls_after_second == calls_after_first  # no new fetches
    # Responses must match byte-for-byte.
    assert r1.json() == r2.json()


def test_cache_expiry_via_patched_clock(
    client: TestClient,
    mock_factor_history: dict[str, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once the patched clock advances past TTL, a refetch happens."""
    # Anchor the clock at t=0.
    clock = [1000.0]
    monkeypatch.setattr(related_mod, "_PERF_COUNTER", lambda: clock[0])

    r1 = client.get("/factors/anchor-slug/related")
    assert r1.status_code == 200
    calls_after_first = sum(mock_factor_history.values())

    # Within TTL — no refetch.
    clock[0] = 1000.0 + related_mod._CACHE_TTL_S - 1.0
    client.get("/factors/anchor-slug/related")
    assert sum(mock_factor_history.values()) == calls_after_first

    # Past TTL — refetch fires.
    clock[0] = 1000.0 + related_mod._CACHE_TTL_S + 1.0
    client.get("/factors/anchor-slug/related")
    assert sum(mock_factor_history.values()) > calls_after_first


def test_window_param_validation_rejects_too_small(
    client: TestClient, mock_factor_history: dict[str, int]
) -> None:
    """``window`` < 7 must 422."""
    r = client.get("/factors/anchor-slug/related?window=6")
    assert r.status_code == 422


def test_window_param_validation_rejects_too_large(
    client: TestClient, mock_factor_history: dict[str, int]
) -> None:
    """``window`` > 365 must 422."""
    r = client.get("/factors/anchor-slug/related?window=366")
    assert r.status_code == 422


def test_window_param_accepts_boundary_values(
    client: TestClient, mock_factor_history: dict[str, int]
) -> None:
    """Boundaries 7 and 365 must both be accepted (200)."""
    r_min = client.get("/factors/anchor-slug/related?window=7")
    assert r_min.status_code == 200
    assert r_min.json()["window_days"] == 7

    related_mod._cache_clear()

    r_max = client.get("/factors/anchor-slug/related?window=365")
    assert r_max.status_code == 200
    assert r_max.json()["window_days"] == 365


def test_pydantic_response_model_shape(
    client: TestClient, mock_factor_history: dict[str, int]
) -> None:
    """Every row must conform to the RelatedFactor schema."""
    r = client.get("/factors/anchor-slug/related")
    assert r.status_code == 200
    payload = r.json()
    assert set(payload.keys()) == {"anchor", "window_days", "related"}
    for row in payload["related"]:
        assert set(row.keys()) == {"slug", "rho", "p_value", "n_obs"}
        assert isinstance(row["slug"], str)
        assert -1.0 <= row["rho"] <= 1.0
        assert 0.0 <= row["p_value"] <= 1.0
        assert isinstance(row["n_obs"], int)
        assert row["n_obs"] >= 20  # MIN_OVERLAP_OBS


def test_results_capped_at_top_n(
    factor_catalog: dict[str, FactorConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With 15 correlated candidates, only the top 10 are returned."""
    related_mod._cache_clear()
    idx = pd.date_range("2026-03-01", periods=60, freq="D", tz="UTC")
    anchor = pd.Series(np.linspace(0.30, 0.70, len(idx)), index=idx)
    rng = np.random.default_rng(seed=7)

    catalog: dict[str, FactorConfig] = {"anchor": _make_factor("anchor", "anchor-slug")}
    bank = {
        "anchor-slug": pd.DataFrame({"price": anchor.values}, index=idx),
    }
    for i in range(15):
        slug = f"slug-{i:02d}"
        catalog[f"c{i}"] = _make_factor(f"c{i}", slug)
        # Each candidate has decreasing noise → decreasing |ρ|.
        bank[slug] = pd.DataFrame(
            {"price": (anchor + rng.normal(0, 0.01 * (i + 1), len(idx))).values},
            index=idx,
        )

    def _fake_cached(fc, start, end, poly, cache, settings):
        df = bank.get(fc.slug)
        return df if df is not None else pd.DataFrame()

    import pfm.regression_core as rc

    monkeypatch.setattr(rc, "_cached_factor_history", _fake_cached)

    app = FastAPI()
    app.include_router(related_mod.router)
    fake_settings = Settings(
        polymarket_gamma_url="http://gamma.test",
        polymarket_clob_url="http://clob.test",
    )
    app.dependency_overrides[get_factors_dep] = lambda: catalog
    app.dependency_overrides[get_cache] = lambda: NullCache()
    app.dependency_overrides[get_polymarket_client] = lambda: MagicMock()
    from pfm.config import get_settings

    app.dependency_overrides[get_settings] = lambda: fake_settings

    with TestClient(app) as c:
        r = c.get("/factors/anchor-slug/related")
    assert r.status_code == 200
    related = r.json()["related"]
    assert len(related) == 10  # TOP_N
    # Verify desc-by-|ρ| ordering.
    abs_rhos = [abs(row["rho"]) for row in related]
    assert abs_rhos == sorted(abs_rhos, reverse=True)
    related_mod._cache_clear()


def test_cache_key_separates_distinct_windows(
    client: TestClient, mock_factor_history: dict[str, int]
) -> None:
    """``?window=30`` and ``?window=60`` are cached independently."""
    r1 = client.get("/factors/anchor-slug/related?window=30")
    assert r1.status_code == 200
    calls_after_30 = sum(mock_factor_history.values())

    r2 = client.get("/factors/anchor-slug/related?window=60")
    assert r2.status_code == 200
    calls_after_60 = sum(mock_factor_history.values())
    # New window key → new fetches must have happened.
    assert calls_after_60 > calls_after_30
    assert r1.json()["window_days"] == 30
    assert r2.json()["window_days"] == 60


def test_min_overlap_filter_drops_short_candidates(
    client: TestClient, mock_factor_history: dict[str, int]
) -> None:
    """Candidates with <20 overlapping obs (e.g. slug-short with 10 obs) are dropped."""
    r = client.get("/factors/anchor-slug/related")
    related = r.json()["related"]
    slugs = [row["slug"] for row in related]
    assert "slug-short" not in slugs
    # All returned rows must satisfy n_obs >= MIN_OVERLAP_OBS.
    assert all(row["n_obs"] >= related_mod.MIN_OVERLAP_OBS for row in related)


def test_perfect_anti_correlation_p_value_near_zero(
    client: TestClient, mock_factor_history: dict[str, int]
) -> None:
    """slug-neg is constructed as 1 - anchor → ρ ≈ -1, p_value ≈ 0."""
    r = client.get("/factors/anchor-slug/related")
    related = r.json()["related"]
    neg = next((row for row in related if row["slug"] == "slug-neg"), None)
    assert neg is not None
    assert neg["rho"] < -0.99
    assert neg["p_value"] < 0.001
