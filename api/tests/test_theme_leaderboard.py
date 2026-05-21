"""Tests for ``GET /factors/themes/{theme}/leaderboard`` (W12-16).

The router under test ships at ``pfm.factors_theme_leaderboard_router``
(NOT ``pfm.factors.theme_leaderboard_router`` — see the module docstring
for the package-vs-module reason). It is not wired into the running app
yet (main.py:routes is held by another active session), so each test
mounts the router into a fresh ``FastAPI`` instance and stages factor
history via the same monkeypatch hook the related-factors test uses
(``pfm.regression_core._cached_factor_history``).
"""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm import factors_theme_leaderboard_router as lb_mod
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


def _make_factor(fid: str, slug: str, theme: str = "macro") -> FactorConfig:
    return FactorConfig(
        id=fid,
        name=fid.replace("_", " ").title(),
        slug=slug,
        source="polymarket",
        description=f"Test factor {fid}",
        theme=theme,
        is_probability=True,
    )


def _wrap(s: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"price": s.values}, index=s.index)
    df.index.name = "date"
    return df


@pytest.fixture
def factor_catalog() -> dict[str, FactorConfig]:
    """Catalog with three macro factors (one dead), two election factors."""
    return {
        "m_high": _make_factor("m_high", "macro-high", theme="macro"),
        "m_mid": _make_factor("m_mid", "macro-mid", theme="macro"),
        "m_low": _make_factor("m_low", "macro-low", theme="macro"),
        "m_dead": _make_factor("m_dead", "macro-dead", theme="macro"),
        "e_only": _make_factor("e_only", "election-only", theme="elections"),
        "e_steady": _make_factor("e_steady", "election-steady", theme="elections"),
    }


@pytest.fixture
def history_bank() -> dict[str, pd.DataFrame]:
    """Synthetic price history per slug, indexed by UTC midnight.

    Built so the desc-by-vol_7d ranking is unambiguous:
      ``macro-high``   → high-volatility recent week  (largest σ)
      ``macro-mid``    → moderate
      ``macro-low``    → very stable
      ``macro-dead``   → only 8 obs (below MIN_OBS_LIVE=30; should drop)
      ``election-only`` → 60 obs, well-defined vol
      ``election-steady`` → 60 obs, essentially flat
    """
    idx = pd.date_range("2026-03-01", periods=60, freq="D", tz="UTC")
    rng = np.random.default_rng(seed=42)

    # Stable baseline at 0.50 and inject a vol shock in the trailing 7 days
    # only for ``macro-high``. The vol calc uses the trailing-7 log-returns,
    # so this guarantees ``macro-high`` ranks first.
    base = np.full(len(idx), 0.50)

    high = base.copy()
    high[-7:] += rng.normal(0.0, 0.08, 7)  # large recent shocks

    mid = base.copy()
    mid[-7:] += rng.normal(0.0, 0.02, 7)  # moderate shocks

    low = base.copy()
    low[-7:] += rng.normal(0.0, 0.0005, 7)  # near-zero shocks

    dead = pd.Series(np.linspace(0.40, 0.50, 8), index=idx[:8])

    e_only = base.copy()
    e_only[-7:] += rng.normal(0.0, 0.04, 7)

    e_steady = base.copy()  # flat-ish; vol ≈ 0

    def _ser(arr: np.ndarray) -> pd.Series:
        # Clip into the open probability interval so the router doesn't
        # have to fall back to its lower-only clip.
        return pd.Series(np.clip(arr, 0.01, 0.99), index=idx)

    return {
        "macro-high": _wrap(_ser(high)),
        "macro-mid": _wrap(_ser(mid)),
        "macro-low": _wrap(_ser(low)),
        "macro-dead": _wrap(dead),
        "election-only": _wrap(_ser(e_only)),
        "election-steady": _wrap(_ser(e_steady)),
    }


@pytest.fixture
def mock_factor_history(
    monkeypatch: pytest.MonkeyPatch, history_bank: dict[str, pd.DataFrame]
) -> dict[str, int]:
    """Patch ``_cached_factor_history`` to return our synthetic bank.

    Returns a dict that counts calls per slug — used by the cache-hit test.
    """
    call_counter: dict[str, int] = {}

    def _fake_cached(fc, start, end, poly, cache, settings):
        call_counter[fc.slug] = call_counter.get(fc.slug, 0) + 1
        df = history_bank.get(fc.slug)
        if df is None:
            return pd.DataFrame()
        return df

    import pfm.regression_core as rc

    monkeypatch.setattr(rc, "_cached_factor_history", _fake_cached)
    return call_counter


def _mount(catalog: dict[str, FactorConfig]) -> TestClient:
    """Mount the router into a bare FastAPI app with DI overrides."""
    lb_mod._cache_clear()
    app = FastAPI()
    app.include_router(lb_mod.router)

    fake_settings = Settings(
        polymarket_gamma_url="http://gamma.test",
        polymarket_clob_url="http://clob.test",
    )
    app.dependency_overrides[get_factors_dep] = lambda: catalog
    app.dependency_overrides[get_cache] = lambda: NullCache()
    app.dependency_overrides[get_polymarket_client] = lambda: MagicMock()
    from pfm.config import get_settings

    app.dependency_overrides[get_settings] = lambda: fake_settings
    return TestClient(app)


@pytest.fixture
def client(factor_catalog: dict[str, FactorConfig]) -> Iterator[TestClient]:
    with _mount(factor_catalog) as c:
        yield c
    lb_mod._cache_clear()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_default_returns_macro_factors_sorted_desc_by_vol(
    client: TestClient, mock_factor_history: dict[str, int]
) -> None:
    """Highest vol_7d first; dead factor (n_obs<30) dropped by default."""
    r = client.get("/factors/themes/macro/leaderboard")
    assert r.status_code == 200, r.text
    payload = r.json()
    assert payload["theme"] == "macro"
    assert payload["n"] == 20  # N_DEFAULT
    slugs = [row["slug"] for row in payload["factors"]]
    # Dead candidate must be dropped at the default include_dead=false.
    assert "macro-dead" not in slugs
    # macro-high must be first by construction (largest σ in trailing 7d).
    assert slugs[0] == "macro-high"
    # Descending-by-vol ordering.
    vols = [row["vol_7d"] for row in payload["factors"]]
    assert vols == sorted(vols, reverse=True)


def test_unknown_theme_returns_404(client: TestClient, mock_factor_history: dict[str, int]) -> None:
    """A theme that doesn't appear in the catalog must 404."""
    r = client.get("/factors/themes/does-not-exist/leaderboard")
    assert r.status_code == 404
    assert "not found" in r.json()["detail"].lower()


def test_n_caps_result_count(client: TestClient, mock_factor_history: dict[str, int]) -> None:
    """``?n=2`` must trim the result to 2 rows."""
    r = client.get("/factors/themes/macro/leaderboard?n=2")
    assert r.status_code == 200
    assert len(r.json()["factors"]) == 2


def test_include_dead_true_keeps_short_history_factors(
    client: TestClient, mock_factor_history: dict[str, int]
) -> None:
    """``include_dead=true`` must retain the n_obs<30 factor."""
    r = client.get("/factors/themes/macro/leaderboard?include_dead=true")
    assert r.status_code == 200
    slugs = [row["slug"] for row in r.json()["factors"]]
    assert "macro-dead" in slugs


def test_include_dead_false_is_default(
    client: TestClient, mock_factor_history: dict[str, int]
) -> None:
    """Explicit ``include_dead=false`` matches the default behaviour."""
    r_default = client.get("/factors/themes/macro/leaderboard")
    r_explicit = client.get("/factors/themes/macro/leaderboard?include_dead=false")
    assert r_default.status_code == 200
    assert r_explicit.status_code == 200
    # Same slug set (order should match too).
    assert [row["slug"] for row in r_default.json()["factors"]] == [
        row["slug"] for row in r_explicit.json()["factors"]
    ]


def test_response_schema_shape(client: TestClient, mock_factor_history: dict[str, int]) -> None:
    """Every row must conform to the LeaderboardRow schema."""
    r = client.get("/factors/themes/macro/leaderboard")
    payload = r.json()
    assert set(payload.keys()) == {"theme", "n", "factors"}
    expected = {"slug", "label", "vol_7d", "mean_7d", "n_obs", "last_value"}
    for row in payload["factors"]:
        assert set(row.keys()) == expected
        assert isinstance(row["slug"], str)
        assert isinstance(row["label"], str)
        assert isinstance(row["vol_7d"], (int, float))
        assert isinstance(row["mean_7d"], (int, float))
        assert isinstance(row["n_obs"], int)
        assert row["vol_7d"] >= 0.0
        assert row["n_obs"] >= lb_mod.MIN_OBS_LIVE  # default include_dead=false


def test_results_scoped_to_theme(client: TestClient, mock_factor_history: dict[str, int]) -> None:
    """Only factors with matching theme appear in the response."""
    r = client.get("/factors/themes/macro/leaderboard")
    slugs = [row["slug"] for row in r.json()["factors"]]
    # No elections factor should leak in.
    assert "election-only" not in slugs
    assert "election-steady" not in slugs


def test_elections_theme_isolated(client: TestClient, mock_factor_history: dict[str, int]) -> None:
    """The elections theme returns only its own factors."""
    r = client.get("/factors/themes/elections/leaderboard")
    assert r.status_code == 200
    payload = r.json()
    assert payload["theme"] == "elections"
    slugs = [row["slug"] for row in payload["factors"]]
    assert set(slugs) <= {"election-only", "election-steady"}
    # The flat ("steady") factor should rank below the noisy one.
    if "election-only" in slugs and "election-steady" in slugs:
        idx_only = slugs.index("election-only")
        idx_steady = slugs.index("election-steady")
        assert idx_only < idx_steady


def test_n_validation_rejects_zero(client: TestClient, mock_factor_history: dict[str, int]) -> None:
    """``n=0`` must 422 (below N_MIN=1)."""
    r = client.get("/factors/themes/macro/leaderboard?n=0")
    assert r.status_code == 422


def test_n_validation_rejects_above_max(
    client: TestClient, mock_factor_history: dict[str, int]
) -> None:
    """``n>200`` must 422 (above N_MAX)."""
    r = client.get("/factors/themes/macro/leaderboard?n=201")
    assert r.status_code == 422


def test_cache_hit_does_not_refetch(
    client: TestClient, mock_factor_history: dict[str, int]
) -> None:
    """Second call within TTL must NOT re-invoke ``_cached_factor_history``."""
    r1 = client.get("/factors/themes/macro/leaderboard")
    assert r1.status_code == 200
    calls_after_first = sum(mock_factor_history.values())
    assert calls_after_first > 0

    r2 = client.get("/factors/themes/macro/leaderboard")
    assert r2.status_code == 200
    calls_after_second = sum(mock_factor_history.values())
    assert calls_after_second == calls_after_first
    assert r1.json() == r2.json()


def test_cache_expiry_via_patched_clock(
    client: TestClient,
    mock_factor_history: dict[str, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once the patched clock advances past TTL, a refetch must happen."""
    clock = [1000.0]
    monkeypatch.setattr(lb_mod, "_PERF_COUNTER", lambda: clock[0])

    client.get("/factors/themes/macro/leaderboard")
    calls_after_first = sum(mock_factor_history.values())

    # Within TTL — no refetch.
    clock[0] = 1000.0 + lb_mod._CACHE_TTL_S - 1.0
    client.get("/factors/themes/macro/leaderboard")
    assert sum(mock_factor_history.values()) == calls_after_first

    # Past TTL — refetch fires.
    clock[0] = 1000.0 + lb_mod._CACHE_TTL_S + 1.0
    client.get("/factors/themes/macro/leaderboard")
    assert sum(mock_factor_history.values()) > calls_after_first


def test_cache_key_separates_include_dead_variants(
    client: TestClient, mock_factor_history: dict[str, int]
) -> None:
    """``?include_dead=true`` and the default form must cache independently."""
    r1 = client.get("/factors/themes/macro/leaderboard")
    assert r1.status_code == 200
    calls_after_default = sum(mock_factor_history.values())

    r2 = client.get("/factors/themes/macro/leaderboard?include_dead=true")
    assert r2.status_code == 200
    calls_after_include_dead = sum(mock_factor_history.values())
    # New cache key → new fetches must have happened.
    assert calls_after_include_dead > calls_after_default
    # And the payloads must differ (dead factor present in one only).
    slugs_default = {row["slug"] for row in r1.json()["factors"]}
    slugs_dead = {row["slug"] for row in r2.json()["factors"]}
    assert "macro-dead" not in slugs_default
    assert "macro-dead" in slugs_dead


def test_upstream_error_skipped_does_not_5xx(
    factor_catalog: dict[str, FactorConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A factor whose history fetch raises must be skipped, not 5xx the panel."""
    from fastapi import HTTPException as FHTTPException

    idx = pd.date_range("2026-03-01", periods=60, freq="D", tz="UTC")
    good = pd.Series(
        np.clip(0.5 + np.random.default_rng(7).normal(0, 0.03, len(idx)), 0.01, 0.99),
        index=idx,
    )

    def _fake_cached(fc, start, end, poly, cache, settings):
        if fc.slug == "macro-high":
            raise FHTTPException(status_code=502, detail="upstream boom")
        if fc.slug in ("macro-mid", "macro-low"):
            return _wrap(good)
        return pd.DataFrame()

    import pfm.regression_core as rc

    monkeypatch.setattr(rc, "_cached_factor_history", _fake_cached)
    with _mount(factor_catalog) as c:
        r = c.get("/factors/themes/macro/leaderboard")
    assert r.status_code == 200
    slugs = [row["slug"] for row in r.json()["factors"]]
    # macro-high blew up upstream and must be silently dropped.
    assert "macro-high" not in slugs
    # The healthy candidates still surface.
    assert "macro-mid" in slugs or "macro-low" in slugs
    lb_mod._cache_clear()


def test_empty_history_factor_dropped(
    factor_catalog: dict[str, FactorConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A factor with completely empty history must not appear in results."""

    def _fake_cached(fc, start, end, poly, cache, settings):
        return pd.DataFrame()

    import pfm.regression_core as rc

    monkeypatch.setattr(rc, "_cached_factor_history", _fake_cached)
    with _mount(factor_catalog) as c:
        # include_dead=true to make sure the drop reason is "no last value",
        # not the "n_obs<30" filter.
        r = c.get("/factors/themes/macro/leaderboard?include_dead=true")
    assert r.status_code == 200
    assert r.json()["factors"] == []
    lb_mod._cache_clear()


def test_vol_calc_matches_numpy_reference(
    client: TestClient, mock_factor_history: dict[str, int]
) -> None:
    """vol_7d must equal np.std(log-diff) over the trailing 7 days (ddof=0)."""
    # Use the helper directly to verify the math (the route applies the same
    # logic but adds DI/cache layers).
    idx = pd.date_range("2026-03-01", periods=30, freq="D", tz="UTC")
    rng = np.random.default_rng(99)
    arr = np.clip(0.5 + rng.normal(0, 0.04, len(idx)), 0.01, 0.99)
    s = pd.Series(arr, index=idx)
    vol, mean, n_obs, last_value = lb_mod._vol_and_mean(s)

    expected_log_ret = np.diff(np.log(arr))[-lb_mod.WINDOW_DAYS :]
    expected_vol = float(np.std(expected_log_ret, ddof=0))
    expected_mean = float(np.mean(expected_log_ret))
    assert vol == pytest.approx(expected_vol, rel=1e-9, abs=1e-12)
    assert mean == pytest.approx(expected_mean, rel=1e-9, abs=1e-12)
    assert n_obs == 30
    assert last_value == pytest.approx(float(arr[-1]))
