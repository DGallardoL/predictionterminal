"""Tests for ``GET /factors/{slug}/correlation-matrix`` (W13-14).

The router under test is :mod:`pfm.factors_correlation_matrix_router`. It
isn't wired into the running app yet (``main.py:routes`` is held by another
active session), so each test mounts the router into a fresh
``FastAPI`` instance and stages factor history via the same monkeypatch
hook the rest of the regression suite uses
(``pfm.regression_core._cached_factor_history``).

The W12-23 ``correlations_cache.default_cache()`` is cleared between
tests so cache-hit assertions can be made without test-order coupling.
"""

from __future__ import annotations

from collections.abc import Iterator
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm import factors_correlation_matrix_router as corr_mod
from pfm.cache import NullCache
from pfm.config import Settings, get_settings
from pfm.dependencies import (
    get_cache,
    get_factors_dep,
    get_polymarket_client,
)
from pfm.factors import FactorConfig
from pfm.terminal.correlations_cache import default_cache

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


@pytest.fixture(autouse=True)
def _clear_caches() -> Iterator[None]:
    """Reset both the response TTL cache and the W12-23 matrix cache."""
    corr_mod._response_cache_clear()
    default_cache().clear()
    yield
    corr_mod._response_cache_clear()
    default_cache().clear()


@pytest.fixture
def factor_catalog() -> dict[str, FactorConfig]:
    """A small synthetic catalog with one anchor and six candidates."""
    return {
        "anchor": _make_factor("anchor", "anchor-slug"),
        "c_pos": _make_factor("c_pos", "slug-pos"),
        "c_neg": _make_factor("c_neg", "slug-neg"),
        "c_mid": _make_factor("c_mid", "slug-mid"),
        "c_low": _make_factor("c_low", "slug-low"),
        "c_short": _make_factor("c_short", "slug-short"),
        "c_zero": _make_factor("c_zero", "slug-zero"),
    }


@pytest.fixture
def history_bank() -> dict[str, pd.DataFrame]:
    """Synthetic price history per slug, indexed by UTC midnight."""
    idx = pd.date_range("2026-03-01", periods=60, freq="D", tz="UTC")
    rng = np.random.default_rng(seed=42)
    anchor = pd.Series(np.linspace(0.30, 0.70, len(idx)), index=idx)
    # Near-perfect anti-correlation
    neg = 1.0 - anchor
    # Strong positive ρ ≈ 0.99+
    pos = anchor + rng.normal(0.0, 0.003, len(idx))
    # Mid ρ ≈ 0.5
    mid = 0.5 * anchor + rng.normal(0.0, 0.06, len(idx))
    # Low ρ (much noisier)
    low = 0.2 * anchor + rng.normal(0.0, 0.20, len(idx))
    # Short — only 10 obs, must be dropped
    short = pd.Series(np.linspace(0.40, 0.50, 10), index=idx[:10])
    # Zero variance — constant, must be dropped
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
        "slug-low": _wrap(low),
        "slug-short": _wrap(short),
        "slug-zero": _wrap(zero),
    }


@pytest.fixture
def mock_factor_history(
    monkeypatch: pytest.MonkeyPatch, history_bank: dict[str, pd.DataFrame]
) -> dict[str, int]:
    """Patch ``_cached_factor_history`` to serve the synthetic bank.

    Returns a per-slug call counter — handy for the cache-hit test.
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


def _make_client(catalog: dict[str, FactorConfig]) -> TestClient:
    """Stand up a fresh app with DI overrides — small helper to avoid repetition."""
    app = FastAPI()
    app.include_router(corr_mod.router)

    fake_settings = Settings(
        polymarket_gamma_url="http://gamma.test",
        polymarket_clob_url="http://clob.test",
    )
    app.dependency_overrides[get_factors_dep] = lambda: catalog
    app.dependency_overrides[get_cache] = lambda: NullCache()
    app.dependency_overrides[get_polymarket_client] = lambda: MagicMock()
    app.dependency_overrides[get_settings] = lambda: fake_settings

    return TestClient(app)


@pytest.fixture
def client(factor_catalog: dict[str, FactorConfig]) -> Iterator[TestClient]:
    with _make_client(factor_catalog) as c:
        yield c


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_happy_path_returns_square_matrix(
    client: TestClient, mock_factor_history: dict[str, int]
) -> None:
    """Default ``top_n=20`` collapses to N+1 peers and a (N+1)×(N+1) matrix."""
    r = client.get("/factors/anchor-slug/correlation-matrix")
    assert r.status_code == 200, r.text
    payload = r.json()
    assert payload["anchor"] == "anchor-slug"
    assert payload["window_days"] == 30

    peers = payload["peers"]
    matrix = payload["matrix"]
    # Anchor + N peers ⇒ square matrix of dimension N+1
    assert len(matrix) == len(peers) + 1
    for row in matrix:
        assert len(row) == len(peers) + 1
    # Anchor must NEVER appear in the peer list.
    assert "anchor-slug" not in peers


def test_diagonal_is_ones_and_matrix_symmetric(
    client: TestClient, mock_factor_history: dict[str, int]
) -> None:
    """Mathematical invariants: diag=1 and ρ(i,j) == ρ(j,i)."""
    r = client.get("/factors/anchor-slug/correlation-matrix?top_n=5")
    assert r.status_code == 200
    matrix = r.json()["matrix"]
    n = len(matrix)
    for i in range(n):
        assert matrix[i][i] == pytest.approx(1.0, abs=1e-9)
        for j in range(i + 1, n):
            assert matrix[i][j] == pytest.approx(matrix[j][i], abs=1e-9)


def test_peers_sorted_by_abs_rho_desc(
    client: TestClient, mock_factor_history: dict[str, int]
) -> None:
    """Peer slugs come back in descending |ρ| against the anchor."""
    r = client.get("/factors/anchor-slug/correlation-matrix?top_n=10")
    matrix = r.json()["matrix"]
    peers = r.json()["peers"]
    # Row 0 is the anchor; its cells against peers should be |non-increasing|.
    abs_anchor_row = [abs(matrix[0][k + 1]) for k in range(len(peers))]
    assert abs_anchor_row == sorted(abs_anchor_row, reverse=True)


def test_unknown_anchor_returns_404(
    client: TestClient, mock_factor_history: dict[str, int]
) -> None:
    r = client.get("/factors/does-not-exist/correlation-matrix")
    assert r.status_code == 404
    assert "not found" in r.json()["detail"].lower()


def test_short_and_zero_variance_candidates_excluded(
    client: TestClient, mock_factor_history: dict[str, int]
) -> None:
    """slug-short (10 obs) and slug-zero (constant) must not appear."""
    r = client.get("/factors/anchor-slug/correlation-matrix?top_n=20")
    peers = r.json()["peers"]
    assert "slug-short" not in peers
    assert "slug-zero" not in peers


def test_top_n_caps_peer_count(client: TestClient, mock_factor_history: dict[str, int]) -> None:
    """``top_n=2`` returns at most 2 peers and a 3×3 matrix."""
    r = client.get("/factors/anchor-slug/correlation-matrix?top_n=2")
    payload = r.json()
    assert len(payload["peers"]) <= 2
    assert len(payload["matrix"]) == len(payload["peers"]) + 1


def test_window_param_validation(client: TestClient, mock_factor_history: dict[str, int]) -> None:
    """Out-of-range ``window_days`` rejected by FastAPI validation."""
    assert client.get("/factors/anchor-slug/correlation-matrix?window_days=6").status_code == 422
    assert client.get("/factors/anchor-slug/correlation-matrix?window_days=366").status_code == 422
    # Boundary values accepted.
    assert client.get("/factors/anchor-slug/correlation-matrix?window_days=7").status_code == 200
    corr_mod._response_cache_clear()
    default_cache().clear()
    assert client.get("/factors/anchor-slug/correlation-matrix?window_days=365").status_code == 200


def test_top_n_param_validation(client: TestClient, mock_factor_history: dict[str, int]) -> None:
    """Out-of-range ``top_n`` rejected."""
    assert client.get("/factors/anchor-slug/correlation-matrix?top_n=0").status_code == 422
    assert client.get("/factors/anchor-slug/correlation-matrix?top_n=51").status_code == 422


def test_response_cache_hit_skips_refetch(
    client: TestClient, mock_factor_history: dict[str, int]
) -> None:
    """Second identical call within TTL must not re-invoke history fetcher."""
    r1 = client.get("/factors/anchor-slug/correlation-matrix?top_n=5")
    assert r1.status_code == 200
    calls_after_first = sum(mock_factor_history.values())
    assert calls_after_first > 0

    r2 = client.get("/factors/anchor-slug/correlation-matrix?top_n=5")
    assert r2.status_code == 200
    calls_after_second = sum(mock_factor_history.values())
    assert calls_after_second == calls_after_first
    # Responses identical byte-for-byte.
    assert r1.json() == r2.json()


def test_response_cache_expiry_via_patched_clock(
    client: TestClient,
    mock_factor_history: dict[str, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Once the patched clock advances past TTL a refetch occurs."""
    clock = [1000.0]
    monkeypatch.setattr(corr_mod, "_PERF_COUNTER", lambda: clock[0])

    client.get("/factors/anchor-slug/correlation-matrix?top_n=5")
    calls_after_first = sum(mock_factor_history.values())

    # Within TTL — no refetch.
    clock[0] = 1000.0 + corr_mod._CACHE_TTL_S - 1.0
    client.get("/factors/anchor-slug/correlation-matrix?top_n=5")
    assert sum(mock_factor_history.values()) == calls_after_first

    # Past TTL — refetch fires.
    clock[0] = 1000.0 + corr_mod._CACHE_TTL_S + 1.0
    client.get("/factors/anchor-slug/correlation-matrix?top_n=5")
    assert sum(mock_factor_history.values()) > calls_after_first


def test_distinct_top_n_cache_independently(
    client: TestClient, mock_factor_history: dict[str, int]
) -> None:
    """``top_n=5`` and ``top_n=10`` produce separate cache entries."""
    r5 = client.get("/factors/anchor-slug/correlation-matrix?top_n=5")
    r10 = client.get("/factors/anchor-slug/correlation-matrix?top_n=10")
    assert r5.status_code == 200
    assert r10.status_code == 200
    assert len(r5.json()["peers"]) <= 5
    assert len(r10.json()["peers"]) <= 10
    # Larger top_n is a superset prefix of smaller.
    assert r10.json()["peers"][: len(r5.json()["peers"])] == r5.json()["peers"]


def test_perfect_anti_correlation_appears_with_negative_rho(
    client: TestClient, mock_factor_history: dict[str, int]
) -> None:
    """slug-neg = 1 - anchor ⇒ ρ ≈ -1 with the anchor."""
    r = client.get("/factors/anchor-slug/correlation-matrix?top_n=20")
    payload = r.json()
    peers = payload["peers"]
    assert "slug-neg" in peers
    neg_idx = peers.index("slug-neg") + 1  # +1 for anchor row
    rho = payload["matrix"][0][neg_idx]
    assert rho < -0.99


def test_anchor_with_empty_history_returns_one_by_one_matrix(
    factor_catalog: dict[str, FactorConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the anchor has no history we return ``[[1.0]]`` instead of 5xx."""

    def _empty(fc, start, end, poly, cache, settings):
        return pd.DataFrame()

    import pfm.regression_core as rc

    monkeypatch.setattr(rc, "_cached_factor_history", _empty)

    with _make_client(factor_catalog) as c:
        r = c.get("/factors/anchor-slug/correlation-matrix")
    assert r.status_code == 200
    payload = r.json()
    assert payload["peers"] == []
    assert payload["matrix"] == [[1.0]]


def test_response_pydantic_shape(client: TestClient, mock_factor_history: dict[str, int]) -> None:
    """Top-level keys and types match the documented schema."""
    r = client.get("/factors/anchor-slug/correlation-matrix")
    payload = r.json()
    assert set(payload.keys()) == {"anchor", "window_days", "peers", "matrix"}
    assert isinstance(payload["anchor"], str)
    assert isinstance(payload["window_days"], int)
    assert isinstance(payload["peers"], list)
    assert isinstance(payload["matrix"], list)
    for row in payload["matrix"]:
        assert isinstance(row, list)
        for cell in row:
            assert isinstance(cell, (int, float))
            # Strict-JSON requires finite numbers.
            assert -1.0 <= float(cell) <= 1.0


def test_matrix_values_are_clipped_to_unit_interval(
    client: TestClient, mock_factor_history: dict[str, int]
) -> None:
    """No cell may overshoot the [-1, 1] interval (no float drift past boundary)."""
    r = client.get("/factors/anchor-slug/correlation-matrix?top_n=10")
    matrix = r.json()["matrix"]
    flat = [c for row in matrix for c in row]
    assert all(-1.0 <= c <= 1.0 for c in flat)


def test_w12_23_correlations_cache_is_populated(
    client: TestClient, mock_factor_history: dict[str, int]
) -> None:
    """Calling the route must populate the W12-23 default cache."""
    cache_before = len(default_cache())
    r = client.get("/factors/anchor-slug/correlation-matrix?top_n=5")
    assert r.status_code == 200
    cache_after = len(default_cache())
    # At least one new ``CorrMatrix`` entry stored.
    assert cache_after >= cache_before + 1
