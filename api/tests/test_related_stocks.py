"""Tests for ``pfm.terminal.related_stocks_router`` — /terminal/related-stocks/{ticker}.

The /fit machinery is fully mocked via the module-level
``_default_exposure_fn`` hook so the tests are hermetic — no Polymarket,
no yfinance, no statsmodels regression. The router is mounted on a bare
:class:`FastAPI` app to avoid pulling the full ``pfm.main`` lifespan.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm.terminal import related_stocks_router as rs

# --- fixtures ---------------------------------------------------------------


@pytest.fixture
def synthetic_universe() -> tuple[str, ...]:
    """Compact 6-ticker synthetic universe to make assertions easy."""
    return ("NVDA", "AMD", "MSFT", "AAPL", "JPM", "XOM")


@pytest.fixture
def synthetic_factors() -> tuple[str, ...]:
    """3-factor basis: 'ai-boom', 'fed-cut', 'oil-100'."""
    return ("ai-boom", "fed-cut", "oil-100")


@pytest.fixture
def synthetic_exposures() -> dict[str, dict[str, float]]:
    """Hand-crafted exposures: NVDA & AMD are nearly co-linear (both heavy AI),
    MSFT is moderately ai-loaded, AAPL is unrelated, JPM is rate-driven, XOM is oil.
    """
    return {
        # NVDA: strongly ai-boom, mild fed-cut, near-zero oil
        "NVDA": {"ai-boom": 0.95, "fed-cut": 0.30, "oil-100": 0.02},
        # AMD: very close to NVDA — should be top peer.
        "AMD": {"ai-boom": 0.90, "fed-cut": 0.32, "oil-100": 0.05},
        # MSFT: ai-boom but less, similar fed exposure
        "MSFT": {"ai-boom": 0.55, "fed-cut": 0.30, "oil-100": 0.01},
        # AAPL: weak signals, mostly orthogonal
        "AAPL": {"ai-boom": 0.10, "fed-cut": 0.05, "oil-100": -0.02},
        # JPM: rate trade, fed-cut dominant, oppose AI
        "JPM": {"ai-boom": -0.30, "fed-cut": 0.80, "oil-100": 0.10},
        # XOM: oil-driven, anti-correlated to AI
        "XOM": {"ai-boom": -0.40, "fed-cut": -0.20, "oil-100": 0.95},
    }


@pytest.fixture
def patched_exposure_fn(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_exposures: dict[str, dict[str, float]],
) -> None:
    """Replace ``_default_exposure_fn`` with a lookup against the fixture."""

    def _fn(ticker: str, factors: tuple[str, ...]) -> dict[str, float]:
        exp = synthetic_exposures.get(ticker.upper(), {})
        # Only return entries for the requested factors.
        return {f: float(exp.get(f, 0.0)) for f in factors}

    monkeypatch.setattr(rs, "_default_exposure_fn", _fn)


@pytest.fixture
def patched_universe(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_universe: tuple[str, ...],
    synthetic_factors: tuple[str, ...],
) -> None:
    """Shrink the module-level universe + factor basis to the synthetic set."""
    monkeypatch.setattr(rs, "UNIVERSE_TICKERS", synthetic_universe)
    monkeypatch.setattr(rs, "DEFAULT_FACTOR_SLUGS", synthetic_factors)


@pytest.fixture(autouse=True)
def _clear_cache_between_tests() -> Iterator[None]:
    """Each test starts with an empty cache."""
    rs.clear_cache()
    yield
    rs.clear_cache()


@pytest.fixture
def client(patched_universe: None, patched_exposure_fn: None) -> TestClient:
    """FastAPI TestClient bound to a bare app mounting just the router."""
    app = FastAPI()
    app.include_router(rs.router)
    return TestClient(app)


# --- cosine_similarity unit tests -------------------------------------------


def test_cosine_similarity_identical_vectors_returns_1() -> None:
    v = {"a": 1.0, "b": 2.0, "c": 3.0}
    assert math.isclose(rs.cosine_similarity(v, v), 1.0, abs_tol=1e-9)


def test_cosine_similarity_orthogonal_returns_0() -> None:
    a = {"a": 1.0, "b": 0.0}
    b = {"a": 0.0, "b": 1.0}
    assert math.isclose(rs.cosine_similarity(a, b), 0.0, abs_tol=1e-9)


def test_cosine_similarity_anti_correlated_returns_minus_1() -> None:
    a = {"a": 1.0, "b": 2.0}
    b = {"a": -1.0, "b": -2.0}
    assert math.isclose(rs.cosine_similarity(a, b), -1.0, abs_tol=1e-9)


def test_cosine_similarity_zero_norm_returns_0() -> None:
    """Divide-by-zero guard: empty/zero vector returns 0, not NaN/Inf."""
    a = {"a": 0.0, "b": 0.0}
    b = {"a": 1.0, "b": 1.0}
    assert rs.cosine_similarity(a, b) == 0.0


def test_cosine_similarity_missing_keys_treated_as_zero() -> None:
    a = {"a": 1.0, "b": 1.0}
    b = {"a": 1.0}  # no 'b' key
    # cos = (1*1 + 1*0) / (sqrt(2) * sqrt(1)) = 1/sqrt(2)
    assert math.isclose(rs.cosine_similarity(a, b), 1.0 / math.sqrt(2.0), abs_tol=1e-9)


# --- shared_factors unit tests ----------------------------------------------


def test_shared_factors_intersection_correct() -> None:
    a = {"x": 0.5, "y": 0.05, "z": 0.30}
    b = {"x": 0.4, "y": 0.95, "z": 0.20}
    # x: both >= 0.15 → in
    # y: a is 0.05 < 0.15 → out
    # z: both >= 0.15 → in
    out = rs.shared_factors(a, b, threshold=0.15)
    assert set(out) == {"x", "z"}


def test_shared_factors_sorted_by_combined_magnitude() -> None:
    a = {"x": 0.2, "y": 0.9}
    b = {"x": 0.2, "y": 0.9}
    # y combined = 1.8, x combined = 0.4 → y first
    out = rs.shared_factors(a, b, threshold=0.15)
    assert out == ["y", "x"]


def test_shared_factors_respects_threshold() -> None:
    a = {"x": 0.1, "y": 0.2}
    b = {"x": 0.1, "y": 0.2}
    assert rs.shared_factors(a, b, threshold=0.15) == ["y"]
    assert rs.shared_factors(a, b, threshold=0.05) == ["y", "x"]


def test_shared_factors_handles_signed_loadings() -> None:
    """Magnitude-based — negative loadings still count if |β| >= threshold."""
    a = {"x": -0.5, "y": 0.5}
    b = {"x": -0.4, "y": -0.05}  # y is too small for b
    out = rs.shared_factors(a, b, threshold=0.15)
    assert out == ["x"]


# --- compute_related_stocks --------------------------------------------------


def test_known_ticker_returns_peers(
    patched_universe: None,
    patched_exposure_fn: None,
    synthetic_universe: tuple[str, ...],
) -> None:
    out = rs.compute_related_stocks("NVDA")
    assert out["anchor"] == "NVDA"
    # 5 peers in a 6-ticker universe (anchor excluded).
    assert len(out["peers"]) == 5


def test_top_n_caps_peer_list() -> None:
    """When ``top_n=2``, only the best two peers come back."""
    out = rs.compute_related_stocks("NVDA", top_n=2)
    assert len(out["peers"]) == 2


def test_self_similarity_excluded(
    patched_universe: None,
    patched_exposure_fn: None,
) -> None:
    """NVDA vs NVDA must NOT appear in the peer list — anchor self-excluded."""
    out = rs.compute_related_stocks("NVDA")
    assert all(p["ticker"] != "NVDA" for p in out["peers"])


def test_amd_is_top_peer_of_nvda(
    patched_universe: None,
    patched_exposure_fn: None,
) -> None:
    """With the hand-crafted exposures AMD should beat MSFT/AAPL/JPM/XOM."""
    out = rs.compute_related_stocks("NVDA")
    assert out["peers"][0]["ticker"] == "AMD"
    assert out["peers"][0]["similarity"] > 0.99  # nearly co-linear


def test_xom_is_low_or_negative_for_nvda(
    patched_universe: None,
    patched_exposure_fn: None,
) -> None:
    """XOM has opposite AI sign + heavy oil — should sit at the bottom."""
    out = rs.compute_related_stocks("NVDA")
    xom_entry = next(p for p in out["peers"] if p["ticker"] == "XOM")
    # last position by sim desc
    assert out["peers"][-1]["ticker"] == "XOM"
    assert xom_entry["similarity"] < 0.0


def test_peers_sorted_descending_by_similarity() -> None:
    out = rs.compute_related_stocks("NVDA")
    sims = [p["similarity"] for p in out["peers"]]
    assert sims == sorted(sims, reverse=True)


def test_unknown_ticker_raises_value_error() -> None:
    with pytest.raises(ValueError, match="unknown ticker"):
        rs.compute_related_stocks("NOTREAL")


def test_lowercase_ticker_normalised() -> None:
    """Ticker is case-insensitive — 'nvda' resolves to NVDA."""
    out = rs.compute_related_stocks("nvda")
    assert out["anchor"] == "NVDA"


def test_shared_factors_present_for_nvda_amd(
    patched_universe: None,
    patched_exposure_fn: None,
) -> None:
    """AI-boom is heavily loaded for both NVDA and AMD → must appear."""
    out = rs.compute_related_stocks("NVDA")
    amd_entry = next(p for p in out["peers"] if p["ticker"] == "AMD")
    assert "ai-boom" in amd_entry["shared_factors"]
    # AMD also has fed-cut loading >= threshold → shared.
    assert "fed-cut" in amd_entry["shared_factors"]


def test_shared_factors_excludes_low_exposure(
    patched_universe: None,
    patched_exposure_fn: None,
) -> None:
    """AAPL has tiny exposures across the board — its shared_factors must be empty."""
    out = rs.compute_related_stocks("NVDA")
    aapl_entry = next(p for p in out["peers"] if p["ticker"] == "AAPL")
    # AAPL all loadings < 0.15 → nothing shared.
    assert aapl_entry["shared_factors"] == []


def test_explicit_exposure_fn_overrides_default() -> None:
    """Caller can inject a custom exposure_fn without monkeypatching."""

    def _all_equal(ticker: str, factors: tuple[str, ...]) -> dict[str, float]:
        return dict.fromkeys(factors, 1.0)

    out = rs.compute_related_stocks(
        "NVDA",
        universe=("NVDA", "AMD", "MSFT"),
        factors=("a", "b"),
        exposure_fn=_all_equal,
    )
    # All vectors identical → similarity = 1.0 for every peer.
    assert all(math.isclose(p["similarity"], 1.0, abs_tol=1e-9) for p in out["peers"])


# --- endpoint / HTTP tests --------------------------------------------------


def test_endpoint_known_ticker_returns_payload(client: TestClient) -> None:
    r = client.get("/terminal/related-stocks/NVDA")
    assert r.status_code == 200
    body = r.json()
    assert body["anchor"] == "NVDA"
    assert isinstance(body["peers"], list)
    assert len(body["peers"]) >= 1
    # Shape: {ticker, similarity, shared_factors}
    first = body["peers"][0]
    assert set(first.keys()) == {"ticker", "similarity", "shared_factors"}
    assert isinstance(first["shared_factors"], list)


def test_endpoint_unknown_ticker_returns_404(client: TestClient) -> None:
    r = client.get("/terminal/related-stocks/NOTREAL")
    assert r.status_code == 404
    assert "unknown ticker" in r.json()["detail"].lower()


def test_endpoint_cache_hit(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """Second call must hit the cache and skip the compute path."""
    call_count = {"n": 0}
    real_compute = rs.compute_related_stocks

    def _spy(*args: Any, **kwargs: Any) -> dict[str, Any]:
        call_count["n"] += 1
        return real_compute(*args, **kwargs)

    monkeypatch.setattr(rs, "compute_related_stocks", _spy)

    r1 = client.get("/terminal/related-stocks/NVDA")
    r2 = client.get("/terminal/related-stocks/NVDA")
    assert r1.status_code == r2.status_code == 200
    assert r1.json() == r2.json()
    # compute called exactly once — the second request is served by the cache.
    assert call_count["n"] == 1


def test_endpoint_self_similarity_excluded_in_response(client: TestClient) -> None:
    """The anchor's own ticker must NOT appear in the peer list."""
    r = client.get("/terminal/related-stocks/NVDA")
    assert r.status_code == 200
    tickers = [p["ticker"] for p in r.json()["peers"]]
    assert "NVDA" not in tickers


def test_endpoint_lowercase_path_normalises(client: TestClient) -> None:
    r = client.get("/terminal/related-stocks/nvda")
    assert r.status_code == 200
    assert r.json()["anchor"] == "NVDA"


def test_endpoint_response_top_n_default_10(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the real ~100-ticker universe, default response caps at 10 peers."""
    # Use the real universe (no patched_universe fixture).
    rs.clear_cache()
    app = FastAPI()
    app.include_router(rs.router)
    c = TestClient(app)
    r = c.get("/terminal/related-stocks/NVDA")
    assert r.status_code == 200
    body = r.json()
    assert body["anchor"] == "NVDA"
    # DEFAULT_TOP_PEERS = 10
    assert len(body["peers"]) == rs.DEFAULT_TOP_PEERS
    # Distinct tickers, none equal to NVDA.
    tickers = [p["ticker"] for p in body["peers"]]
    assert "NVDA" not in tickers
    assert len(set(tickers)) == len(tickers)


def test_universe_size_is_around_100() -> None:
    """Sanity-check: the hardcoded universe must be ~100 tickers per spec."""
    # Spec says "~100"; allow a generous band.
    assert 80 <= len(rs.UNIVERSE_TICKERS) <= 150
    # No duplicates, all upper-case.
    assert len(set(rs.UNIVERSE_TICKERS)) == len(rs.UNIVERSE_TICKERS)
    assert all(t == t.upper() for t in rs.UNIVERSE_TICKERS)


def test_default_exposure_fn_is_deterministic() -> None:
    """The hash-based default must return the same vector twice in a row."""
    factors = ("ai-boom", "fed-cut", "oil-100")
    a = rs._default_exposure_fn("NVDA", factors)
    b = rs._default_exposure_fn("NVDA", factors)
    assert a == b
    # Different tickers → different exposures.
    c = rs._default_exposure_fn("XOM", factors)
    assert a != c


def test_default_exposure_fn_values_in_minus1_to_1() -> None:
    factors = ("ai-boom", "fed-cut", "oil-100", "btc-100k-2026")
    vals = rs._default_exposure_fn("NVDA", factors).values()
    for v in vals:
        assert -1.0 <= v <= 1.0
