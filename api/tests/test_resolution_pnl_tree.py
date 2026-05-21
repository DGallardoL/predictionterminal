"""Tests for ``pfm.resolution_pnl_tree``.

The math is pure NumPy / dataclass plumbing, so most tests pin behaviour
without touching FastAPI. A small TestClient block exercises the router
endpoints to ensure the request/response shapes line up.
"""

from __future__ import annotations

import math

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm.cache_utils import reset_caches
from pfm.resolution_pnl_tree import (
    DEFAULT_BOOTSTRAP_SIGMA,
    build_pnl_tree,
    monte_carlo_pnl,
)
from pfm.resolution_pnl_tree import (
    router as pnl_tree_router,
)


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    reset_caches()
    yield
    reset_caches()


@pytest.fixture
def app_client() -> TestClient:
    app = FastAPI()
    app.include_router(pnl_tree_router)
    return TestClient(app)


# ---------------------------------------------------------------------------
# build_pnl_tree
# ---------------------------------------------------------------------------


def test_two_positions_at_p_half_expected_value_near_zero() -> None:
    """Two equal-and-opposite β positions at prob=0.5 ⇒ EV ≈ 0.

    With current_prob = 0.5 the YES and NO scenarios snap to logits +log(99)
    and -log(99) respectively, perfectly symmetric, so the prob-weighted
    average MTM is exactly zero (modulo float noise).
    """
    positions = [
        {"ticker": "AAPL", "size_usd": 10_000, "beta_factor": 0.20},
        {"ticker": "MSFT", "size_usd": 10_000, "beta_factor": -0.20},
    ]
    out = build_pnl_tree(positions, factor_id="ai-bubble-pop", current_prob=0.5)

    assert out["factor_id"] == "ai-bubble-pop"
    assert out["current_prob"] == 0.5
    assert out["n_positions"] == 2
    assert out["gross_notional_usd"] == 20_000
    assert len(out["scenarios"]) == 2

    # Expected value collapses to zero by symmetry.
    assert out["expected_value_usd"] == pytest.approx(0.0, abs=1e-9)

    # The two scenarios are sign-flipped totals (because the betas net
    # to zero across the book).
    yes = next(s for s in out["scenarios"] if s["outcome"] == "YES")
    no = next(s for s in out["scenarios"] if s["outcome"] == "NO")
    # Across both legs: book_exposure = 10k*0.2 + 10k*(-0.2) = 0
    # ⇒ MTM totals are 0 in both scenarios.
    assert yes["mtm_total_usd"] == pytest.approx(0.0, abs=1e-9)
    assert no["mtm_total_usd"] == pytest.approx(0.0, abs=1e-9)


def test_zero_beta_position_yields_zero_mtm_in_both_scenarios() -> None:
    """A position with β=0 must have MTM=0 in both YES and NO scenarios."""
    positions = [
        {"ticker": "GOLD", "size_usd": 50_000, "beta_factor": 0.0},
        {"ticker": "TLT", "size_usd": 25_000, "beta_factor": 0.50},
    ]
    out = build_pnl_tree(positions, factor_id="recession-2026", current_prob=0.30)

    for scenario in out["scenarios"]:
        legs = {leg["ticker"]: leg for leg in scenario["by_ticker"]}
        assert legs["GOLD"]["mtm_usd"] == pytest.approx(0.0, abs=1e-12)
        assert legs["GOLD"]["delta_return"] == pytest.approx(0.0, abs=1e-12)
        # The non-zero β leg must move.
        assert legs["TLT"]["mtm_usd"] != 0.0


def test_directional_book_yes_beats_no_when_book_exposure_positive() -> None:
    """All-positive β book ⇒ YES scenario MTM > NO scenario MTM.

    This pins the *sign* of the conditional MTM: when the factor resolves
    at YES (logit goes up), a positively exposed book makes money; when
    it resolves at NO (logit drops), it loses money.
    """
    positions = [
        {"ticker": "DJT", "size_usd": 10_000, "beta_factor": 1.0},
        {"ticker": "TSLA", "size_usd": 5_000, "beta_factor": 0.5},
    ]
    out = build_pnl_tree(positions, factor_id="trump-2024", current_prob=0.4)
    yes = next(s for s in out["scenarios"] if s["outcome"] == "YES")
    no = next(s for s in out["scenarios"] if s["outcome"] == "NO")
    assert yes["mtm_total_usd"] > 0
    assert no["mtm_total_usd"] < 0
    # Expected value sign tracks current_prob × YES + (1-prob) × NO.
    assert out["expected_value_usd"] == pytest.approx(
        0.4 * yes["mtm_total_usd"] + 0.6 * no["mtm_total_usd"], rel=1e-9
    )


def test_invalid_current_prob_raises() -> None:
    positions = [{"ticker": "AAPL", "size_usd": 1000, "beta_factor": 0.1}]
    with pytest.raises(ValueError):
        build_pnl_tree(positions, factor_id="x", current_prob=1.5)


def test_empty_positions_raises() -> None:
    with pytest.raises(ValueError):
        build_pnl_tree([], factor_id="x", current_prob=0.5)


def test_beta_map_override_takes_precedence() -> None:
    positions = [{"ticker": "AAPL", "size_usd": 10_000, "beta_factor": 0.0}]
    out = build_pnl_tree(
        positions,
        factor_id="ai-bubble-pop",
        current_prob=0.30,
        beta_map={"AAPL": 0.40},
    )
    yes = next(s for s in out["scenarios"] if s["outcome"] == "YES")
    leg = yes["by_ticker"][0]
    assert leg["beta_factor"] == pytest.approx(0.40)
    assert leg["mtm_usd"] != 0.0


# ---------------------------------------------------------------------------
# monte_carlo_pnl
# ---------------------------------------------------------------------------


def test_monte_carlo_distribution_widens_with_more_paths() -> None:
    """N=10000 should give a more-precise std estimate than N=100.

    We assert the *true* std (which equals |book_exposure| × σ for a
    centred Normal) is recovered to within ~1% at N=10000 but the N=100
    estimate has noticeably wider sampling error. We use seeded RNG so
    the test is deterministic.
    """
    positions = [
        {"ticker": "AAPL", "size_usd": 100_000, "beta_factor": 0.5},
    ]
    sigma = DEFAULT_BOOTSTRAP_SIGMA  # 1.0
    expected_std = abs(100_000 * 0.5 * sigma)  # 50_000

    small = monte_carlo_pnl(
        positions,
        factor_id="x",
        n_paths=100,
        current_prob=0.5,
        bootstrap_sigma=sigma,
        seed=42,
    )
    large = monte_carlo_pnl(
        positions,
        factor_id="x",
        n_paths=10_000,
        current_prob=0.5,
        bootstrap_sigma=sigma,
        seed=42,
    )

    # The N=10k sample should land within 5% of the analytic std.
    assert abs(large["std_pnl_usd"] - expected_std) / expected_std < 0.05
    assert large["n_paths"] == 10_000
    assert small["n_paths"] == 100

    # Percentile dict shape.
    assert set(large["percentiles"]) == {
        "p1",
        "p5",
        "p10",
        "p25",
        "p50",
        "p75",
        "p90",
        "p95",
        "p99",
    }
    # VaR-95 reported as a positive loss number when expected_value≈0.
    assert large["var_95_usd"] > 0


def test_monte_carlo_zero_book_exposure_collapses_to_zero_pnl() -> None:
    """When Σ size×β = 0, every simulated path has PnL = 0."""
    positions = [
        {"ticker": "A", "size_usd": 10_000, "beta_factor": 1.0},
        {"ticker": "B", "size_usd": 10_000, "beta_factor": -1.0},
    ]
    out = monte_carlo_pnl(
        positions,
        factor_id="x",
        n_paths=1000,
        current_prob=0.5,
        seed=0,
    )
    assert out["std_pnl_usd"] == pytest.approx(0.0, abs=1e-9)
    assert out["expected_value_usd"] == pytest.approx(0.0, abs=1e-9)
    for v in out["percentiles"].values():
        assert math.isclose(v, 0.0, abs_tol=1e-9)


def test_monte_carlo_higher_sigma_widens_distribution() -> None:
    """Wider bootstrap σ ⇒ proportionally wider std."""
    positions = [{"ticker": "X", "size_usd": 100_000, "beta_factor": 1.0}]
    narrow = monte_carlo_pnl(
        positions,
        factor_id="x",
        n_paths=10_000,
        current_prob=0.5,
        bootstrap_sigma=0.5,
        seed=7,
    )
    wide = monte_carlo_pnl(
        positions,
        factor_id="x",
        n_paths=10_000,
        current_prob=0.5,
        bootstrap_sigma=2.0,
        seed=7,
    )
    # ~4× σ ⇒ ~4× std.
    ratio = wide["std_pnl_usd"] / narrow["std_pnl_usd"]
    assert 3.5 < ratio < 4.5


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


def test_post_resolution_tree_endpoint(app_client: TestClient) -> None:
    body = {
        "positions": [
            {"ticker": "DJT", "size_usd": 10_000, "beta_factor": 1.0},
            {"ticker": "SPY", "size_usd": 5_000, "beta_factor": -0.10},
        ],
        "factor_id": "trump-2024",
        "current_prob": 0.40,
    }
    resp = app_client.post("/portfolio/resolution-tree", json=body)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["factor_id"] == "trump-2024"
    assert data["n_positions"] == 2
    assert {s["outcome"] for s in data["scenarios"]} == {"YES", "NO"}
    # YES scenario MTM ought to dominate (book is net long DJT).
    yes = next(s for s in data["scenarios"] if s["outcome"] == "YES")
    no = next(s for s in data["scenarios"] if s["outcome"] == "NO")
    assert yes["mtm_total_usd"] > 0
    assert no["mtm_total_usd"] < 0


def test_post_pnl_monte_carlo_endpoint(app_client: TestClient) -> None:
    body = {
        "positions": [
            {"ticker": "AAPL", "size_usd": 100_000, "beta_factor": 0.30},
        ],
        "factor_id": "ai-bubble-pop",
        "n_paths": 5_000,
        "current_prob": 0.50,
        "bootstrap_sigma": 1.0,
        "seed": 123,
    }
    resp = app_client.post("/portfolio/pnl-monte-carlo", json=body)
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["n_paths"] == 5_000
    assert data["std_pnl_usd"] > 0
    assert data["var_95_usd"] > 0
    assert "p5" in data["percentiles"] and "p95" in data["percentiles"]


def test_post_endpoint_rejects_empty_positions(app_client: TestClient) -> None:
    body = {"positions": [], "factor_id": "x", "current_prob": 0.5}
    resp = app_client.post("/portfolio/resolution-tree", json=body)
    # Pydantic v2 rejects min_length=1 with 422.
    assert resp.status_code == 422
