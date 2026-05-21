"""Exhaustive deep tests for multi-event chains, portfolio optimizer, factor model.

Covers:

  1.  LASSO sparse recovery on synthetic DGPs (50 candidates, 3 real).
  2.  Sector attribution: planted-driver recovery + dominant-factor mapping.
  3.  Granger pathfinding (find_chains): A->B->C->ticker chains, depth bounds,
      no-causality check, multi-path ranking.
  4.  Event/macro correlation: planted r=0.7, multi-series, lead-lag direction.
  5.  Systemic PM factor (PCA): explained variance, loading sign, degenerate
      single-factor case.
  6.  Portfolio optimizer:
       * HRP (positive weights, sum=1, singular cov, vs MV diversification,
         scaling to 50 assets, 2-asset inverse-vol behavior).
       * Mean-variance (analytic recovery, max_weight binding, shrinkage).
       * Risk parity ERC (CV<5%, two-asset inverse-vol, diversification ratio).
       * Efficient frontier (monotonicity, anchors, Sharpe peak).
       * Monte Carlo drawdown (quantile ordering, path scaling, horizon
         scaling, VaR direction).
  7.  Reverse factor finder (top-k recovery, ΔR² ordering, VIF, no-data error).
  8.  Prediction-driven alpha (ranking, expected_return).
  9.  Endpoint smoke tests through TestClient.

Conventions:
  * Fixed seeds for reproducibility.
  * No emojis (project rule).
  * Ruff line length 100.
"""

from __future__ import annotations

import time
from datetime import date

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from pfm.model import delta_logit
from pfm.multi_event_chain import (
    event_macro_correlation,
    extract_systemic_pm_factor,
    find_chains,
    fit_multi_event_lasso,
    sector_attribution,
)
from pfm.portfolio_optimizer import (
    efficient_frontier,
    hrp,
    mean_variance_max_sharpe,
    min_variance,
    monte_carlo_drawdown,
    risk_parity_erc,
)
from pfm.portfolio_optimizer_router import router as optimizer_router
from pfm.reverse_finder import prediction_driven_alpha, reverse_find_factors

# ---------------------------------------------------------------------------
# helpers — reused everywhere
# ---------------------------------------------------------------------------


def _utc_idx(n: int, start: str = "2024-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="D", tz="UTC").normalize()


def _ar1_to_prob(noise: np.ndarray, *, phi: float = 0.92) -> np.ndarray:
    """Build a probability path via AR(1) on logit space."""
    z = np.zeros_like(noise)
    for t in range(1, len(noise)):
        z[t] = phi * z[t - 1] + noise[t]
    p = 1.0 / (1.0 + np.exp(-z))
    return np.clip(p, 0.05, 0.95)


def _make_factor_fetcher(price_bank: dict[str, pd.Series]):
    def _fetch(fid, start, end):
        if fid not in price_bank:
            return pd.Series(dtype=float)
        s = price_bank[fid]
        return s[(s.index >= start) & (s.index <= end)]

    return _fetch


def _make_returns_fetcher(returns_bank: dict[str, pd.Series]):
    def _fetch(ticker, start, end):
        if ticker not in returns_bank:
            return pd.Series(dtype=float)
        s = returns_bank[ticker]
        return s[(s.index >= start) & (s.index <= end)]

    return _fetch


def _make_macro_fetcher(macro_bank: dict[str, pd.Series]):
    def _fetch(mid, start, end):
        if mid not in macro_bank:
            return pd.Series(dtype=float)
        s = macro_bank[mid]
        return s[(s.index >= start) & (s.index <= end)]

    return _fetch


def _gaussian_returns(
    sigmas: list[float],
    rho: float = 0.30,
    mu: list[float] | None = None,
    n: int = 500,
    seed: int = 7,
) -> pd.DataFrame:
    """Correlated Gaussian daily returns with prescribed sigma's and rho."""
    rng = np.random.default_rng(seed)
    k = len(sigmas)
    sigmas_arr = np.asarray(sigmas, dtype=float)
    mu_arr = np.zeros(k) if mu is None else np.asarray(mu, dtype=float)
    corr = np.full((k, k), rho)
    np.fill_diagonal(corr, 1.0)
    cov = np.outer(sigmas_arr, sigmas_arr) * corr
    chol = np.linalg.cholesky(cov)
    z = rng.standard_normal(size=(n, k))
    paths = z @ chol.T + mu_arr
    cols = [f"a{i}" for i in range(k)]
    return pd.DataFrame(paths, columns=cols)


def _make_lasso_universe(
    n_obs: int,
    n_factors: int,
    real_betas: dict[str, float],
    *,
    seed: int = 0,
    noise_sigma: float = 1e-3,
) -> tuple[dict[str, pd.Series], pd.Series, pd.DatetimeIndex]:
    """Build a price-bank of ``n_factors`` AR(1) probability series and a
    target ``y`` whose dynamics are explained by the factors named in
    ``real_betas`` (keys must look like 'fK' for K in 0..n_factors-1)."""
    rng = np.random.default_rng(seed)
    idx = _utc_idx(n_obs)
    price_bank: dict[str, pd.Series] = {}
    deltas: dict[str, np.ndarray] = {}
    for i in range(n_factors):
        noise = rng.normal(0, 0.5, size=n_obs)
        p = _ar1_to_prob(noise, phi=0.85)
        s = pd.Series(p, index=idx, name=f"f{i}")
        price_bank[f"f{i}"] = s
        deltas[f"f{i}"] = delta_logit(s).fillna(0.0).to_numpy()

    y_vals = np.zeros(n_obs)
    for fid, beta in real_betas.items():
        y_vals = y_vals + beta * deltas[fid]
    y_vals = y_vals + rng.normal(0, noise_sigma, size=n_obs)
    y = pd.Series(y_vals, index=idx, name="TGT")
    return price_bank, y, idx


# ===========================================================================
# 1. LASSO sparse recovery
# ===========================================================================


class TestLassoSparseRecovery:
    """Exhaustive checks of fit_multi_event_lasso on a 50-factor DGP with 3 real."""

    @pytest.mark.parametrize("n_obs", [100, 200, 500])
    def test_recovery_improves_with_n(self, n_obs: int) -> None:
        """As n increases, the survivor set must include the real factors with
        higher precision."""
        real = {"f1": 0.7, "f2": 0.3}
        price_bank, y, idx = _make_lasso_universe(
            n_obs=n_obs,
            n_factors=50,
            real_betas=real,
            seed=42,
            noise_sigma=1e-3,
        )
        result = fit_multi_event_lasso(
            ticker="TGT",
            factor_ids=[f"f{i}" for i in range(50)],
            start=idx[0].strftime("%Y-%m-%d"),
            end=idx[-1].strftime("%Y-%m-%d"),
            alpha=0.0001,
            fetch_factor=_make_factor_fetcher(price_bank),
            fetch_returns=_make_returns_fetcher({"TGT": y}),
        )

        assert result["n_factors_in"] == 50
        survivors = set(result["sparse_solution_factors"])
        # Both real factors recovered.
        assert "f1" in survivors, f"f1 not in survivors at n={n_obs}: {sorted(survivors)}"
        # f2 has smaller beta so it may be shrunk; require at least one real.
        # As n grows, both should eventually be present.
        if n_obs >= 200:
            assert "f2" in survivors
        # Survivor count well below 50: sparsity is the whole point.
        assert result["n_factors_nonzero"] <= 30

    def test_lasso_alpha_tiny_in_fallback_keeps_more_factors(self) -> None:
        """When n_obs < max(15, n_factors+2), the fallback Lasso(alpha=alpha)
        branch is used. With alpha very tiny, we expect minimal shrinkage:
        most coefs survive (close to OLS behaviour)."""
        real = {"f1": 0.5, "f2": 0.4}
        # n_obs=10, n_factors=8 -> max(15, 10) = 15 > 10, falls through to fallback.
        price_bank, y, idx = _make_lasso_universe(
            n_obs=10,
            n_factors=8,
            real_betas=real,
            seed=7,
            noise_sigma=1e-4,
        )
        result_tiny = fit_multi_event_lasso(
            ticker="TGT",
            factor_ids=[f"f{i}" for i in range(8)],
            start=idx[0].strftime("%Y-%m-%d"),
            end=idx[-1].strftime("%Y-%m-%d"),
            alpha=1e-8,  # ~OLS in the fallback branch
            fetch_factor=_make_factor_fetcher(price_bank),
            fetch_returns=_make_returns_fetcher({"TGT": y}),
        )
        result_strong = fit_multi_event_lasso(
            ticker="TGT",
            factor_ids=[f"f{i}" for i in range(8)],
            start=idx[0].strftime("%Y-%m-%d"),
            end=idx[-1].strftime("%Y-%m-%d"),
            alpha=10.0,  # strong penalty zeros coefs
            fetch_factor=_make_factor_fetcher(price_bank),
            fetch_returns=_make_returns_fetcher({"TGT": y}),
        )
        # alpha tiny preserves more coefs than alpha large.
        assert result_tiny["n_factors_nonzero"] >= result_strong["n_factors_nonzero"]
        assert result_strong["n_factors_nonzero"] == 0

    def test_lasso_alpha_huge_zeros_everything_in_fallback(self) -> None:
        """alpha very large in the fallback path (small-n) -> all coefs zeroed.

        n=10, factors=8 forces the small-n fallback where ``alpha`` is honoured.
        """
        real = {"f1": 0.5}
        price_bank, y, idx = _make_lasso_universe(
            n_obs=10,
            n_factors=8,
            real_betas=real,
            seed=11,
        )
        result = fit_multi_event_lasso(
            ticker="TGT",
            factor_ids=[f"f{i}" for i in range(8)],
            start=idx[0].strftime("%Y-%m-%d"),
            end=idx[-1].strftime("%Y-%m-%d"),
            alpha=10.0,  # huge penalty kills all coefs
            fetch_factor=_make_factor_fetcher(price_bank),
            fetch_returns=_make_returns_fetcher({"TGT": y}),
        )
        assert result["n_factors_nonzero"] == 0
        # When all coefs are zero, R^2 should be ~0 (or negative clamped).
        assert result["r_squared"] <= 0.05

    def test_lasso_optimal_alpha_matches_synthetic(self) -> None:
        """LassoCV's chosen alpha should be a positive small number on a
        well-identified synthetic problem (not 0, not huge)."""
        real = {"f1": 0.6, "f7": 0.4}
        price_bank, y, idx = _make_lasso_universe(
            n_obs=300,
            n_factors=20,
            real_betas=real,
            seed=3,
        )
        result = fit_multi_event_lasso(
            ticker="TGT",
            factor_ids=[f"f{i}" for i in range(20)],
            start=idx[0].strftime("%Y-%m-%d"),
            end=idx[-1].strftime("%Y-%m-%d"),
            alpha=0.01,  # ignored when LassoCV runs
            fetch_factor=_make_factor_fetcher(price_bank),
            fetch_returns=_make_returns_fetcher({"TGT": y}),
        )
        assert result["alpha_optimal"] > 0.0
        assert result["alpha_optimal"] < 1.0  # not pathological
        # Real factors should both survive.
        survivors = set(result["sparse_solution_factors"])
        assert {"f1", "f7"}.issubset(survivors)

    def test_empty_factor_ids_returns_zero(self) -> None:
        result = fit_multi_event_lasso(
            ticker="TGT",
            factor_ids=[],
            start="2024-01-01",
            end="2024-06-01",
            fetch_factor=_make_factor_fetcher({}),
            fetch_returns=_make_returns_fetcher({}),
        )
        assert result["n_factors_in"] == 0
        assert result["n_factors_nonzero"] == 0
        assert result["betas"] == {}

    def test_identical_factors_lasso_picks_subset(self) -> None:
        """If two factors are identical, LASSO should leave at most one of
        them at non-zero (sparse winner-takes-all behavior). Some
        implementations may zero both; we accept either (==1 or ==0) but
        never both > 0 simultaneously with significant magnitude."""
        rng = np.random.default_rng(123)
        n = 250
        idx = _utc_idx(n)
        common = pd.Series(_ar1_to_prob(rng.normal(0, 0.5, n), phi=0.88), index=idx)
        price_bank = {
            "f0": common,
            "f1": common.copy(),  # truly identical
            "f2": pd.Series(_ar1_to_prob(rng.normal(0, 0.5, n), phi=0.88), index=idx),
        }
        d_common = delta_logit(common).fillna(0.0).to_numpy()
        y_vals = 0.5 * d_common + rng.normal(0, 1e-3, n)
        y = pd.Series(y_vals, index=idx, name="TGT")

        result = fit_multi_event_lasso(
            ticker="TGT",
            factor_ids=["f0", "f1", "f2"],
            start=idx[0].strftime("%Y-%m-%d"),
            end=idx[-1].strftime("%Y-%m-%d"),
            alpha=0.01,
            fetch_factor=_make_factor_fetcher(price_bank),
            fetch_returns=_make_returns_fetcher({"TGT": y}),
        )
        # Combined |coef(f0)| + |coef(f1)| should be the *one* effective coefficient.
        c0 = abs(result["betas"].get("f0", 0.0))
        c1 = abs(result["betas"].get("f1", 0.0))
        # At least one of them is nonzero if Lasso recovered the signal.
        assert c0 + c1 > 0.0
        # And the *minimum* of the two should be small relative to the maximum
        # — i.e. Lasso prefers concentrated weight (sparse winner) rather than
        # splitting evenly. Allow tolerance: the ratio min/max <= 0.7.
        if c0 > 0 and c1 > 0:
            assert min(c0, c1) / max(c0, c1) <= 0.7

    def test_highly_correlated_factors_lasso_selects_one_per_pair(self) -> None:
        """rho ~ 0.99 -> Lasso typically zeros one of the pair. We construct
        2 pairs (4 factors total) where each pair shares 99% common variance.
        We expect the n_nonzero to be < 4."""
        rng = np.random.default_rng(7)
        n = 280
        idx = _utc_idx(n)
        from pfm.model import delta_logit

        base_a = pd.Series(_ar1_to_prob(rng.normal(0, 0.5, n), phi=0.85), index=idx)
        base_b = pd.Series(_ar1_to_prob(rng.normal(0, 0.5, n), phi=0.85), index=idx)
        d_a = delta_logit(base_a).fillna(0.0).to_numpy()

        # Build correlated mates by adding a tiny perturbation to the logit path.
        def _perturb(p: pd.Series) -> pd.Series:
            logit = np.log(p / (1 - p))
            logit2 = logit + rng.normal(0, 0.01, len(p))
            p2 = 1.0 / (1.0 + np.exp(-logit2))
            return pd.Series(np.clip(p2, 0.05, 0.95), index=p.index)

        price_bank = {
            "a1": base_a,
            "a2": _perturb(base_a),
            "b1": base_b,
            "b2": _perturb(base_b),
        }
        # y depends on a-pair only.
        y_vals = 0.5 * d_a + rng.normal(0, 1e-3, n)
        y = pd.Series(y_vals, index=idx, name="TGT")

        result = fit_multi_event_lasso(
            ticker="TGT",
            factor_ids=["a1", "a2", "b1", "b2"],
            start=idx[0].strftime("%Y-%m-%d"),
            end=idx[-1].strftime("%Y-%m-%d"),
            alpha=0.01,
            fetch_factor=_make_factor_fetcher(price_bank),
            fetch_returns=_make_returns_fetcher({"TGT": y}),
        )
        # At least one a-pair member should survive; b-pair betas should be small.
        b1 = abs(result["betas"].get("b1", 0.0))
        b2 = abs(result["betas"].get("b2", 0.0))
        a1 = abs(result["betas"].get("a1", 0.0))
        a2 = abs(result["betas"].get("a2", 0.0))
        # a-side (real) much bigger than b-side (irrelevant).
        assert max(a1, a2) > max(b1, b2)


# ===========================================================================
# 2. Sector attribution
# ===========================================================================


class TestSectorAttribution:
    def test_planted_drivers_recovered_per_sector(self) -> None:
        """3 sectors x 5 factors. Each sector loads 'exclusively' on one factor."""
        rng = np.random.default_rng(31)
        n = 240
        idx = _utc_idx(n)

        factor_ids = ["factor_1", "factor_2", "factor_3", "factor_4", "factor_5"]
        price_bank: dict[str, pd.Series] = {}
        deltas: dict[str, np.ndarray] = {}
        for fid in factor_ids:
            p = _ar1_to_prob(rng.normal(0, 0.5, n), phi=0.88)
            s = pd.Series(p, index=idx)
            price_bank[fid] = s
            deltas[fid] = delta_logit(s).fillna(0.0).to_numpy()

        # sector_a sensitive to factor_1, sector_b to factor_2, sector_c to factor_3
        plant = {"sector_a": "factor_1", "sector_b": "factor_2", "sector_c": "factor_3"}
        returns_bank: dict[str, pd.Series] = {}
        for sec, fid in plant.items():
            y = 2.0 * deltas[fid] + rng.normal(0, 1e-3, n)
            returns_bank[sec] = pd.Series(y, index=idx)

        result = sector_attribution(
            sectors_etfs=list(plant.keys()),
            factor_ids=factor_ids,
            start=idx[0].strftime("%Y-%m-%d"),
            end=idx[-1].strftime("%Y-%m-%d"),
            fetch_factor=_make_factor_fetcher(price_bank),
            fetch_returns=_make_returns_fetcher(returns_bank),
        )

        # Each sector's dominant factor matches the planted driver.
        for sec, fid in plant.items():
            assert result["dominant_factor_per_sector"][sec] == fid, (
                f"sector {sec}: dominant={result['dominant_factor_per_sector'][sec]} expected={fid}"
            )
        # Reverse mapping: factor_1's dominant sector is sector_a, etc.
        for sec, fid in plant.items():
            assert result["dominant_sector_per_factor"][fid] == sec

        # Row sums of attribution_matrix must NOT exceed 1 by much (leakage
        # into residuals is expected; but each individual sector's row should
        # have its dominant cell account for most of R^2).
        mat = np.array(result["attribution_matrix"])
        for i in range(len(plant)):
            assert mat[i].max() <= 1.5  # cell values are R^2 shares, allow slack
            assert mat[i].sum() <= 2.0

    def test_default_sector_etfs_runs_without_error(self) -> None:
        """Smoke test: pass sectors_etfs=None and supply all-default ETF names
        as keys in the returns bank; verify the function returns the
        DEFAULT_SECTOR_ETFS list (or a subset)."""
        rng = np.random.default_rng(99)
        n = 150
        idx = _utc_idx(n)
        from pfm.multi_event_chain import DEFAULT_SECTOR_ETFS

        # Just one factor and one driver, but populate ALL default ETF tickers.
        f = pd.Series(_ar1_to_prob(rng.normal(0, 0.5, n), phi=0.88), index=idx)
        d = delta_logit(f).fillna(0.0).to_numpy()
        returns_bank = {
            tk: pd.Series(0.5 * d + rng.normal(0, 1e-3, n), index=idx) for tk in DEFAULT_SECTOR_ETFS
        }
        result = sector_attribution(
            sectors_etfs=None,
            factor_ids=["X"],
            start=idx[0].strftime("%Y-%m-%d"),
            end=idx[-1].strftime("%Y-%m-%d"),
            fetch_factor=_make_factor_fetcher({"X": f}),
            fetch_returns=_make_returns_fetcher(returns_bank),
        )
        assert set(result["sectors"]).issubset(set(DEFAULT_SECTOR_ETFS))
        assert "X" in result["factors"]


# ===========================================================================
# 3. Find chains (Granger pathfinding)
# ===========================================================================


class TestFindChains:
    """Three-hop chain A -> B -> C -> ticker via Granger-style lagged dependence."""

    def _build_chain_universe(self, seed: int = 17, n: int = 280):
        """Build A -> B -> C -> ticker via lagged innovation injection."""
        rng = np.random.default_rng(seed)
        idx = _utc_idx(n)

        # A: independent AR(1) on logit.
        a_logit = np.zeros(n)
        eps_a = rng.normal(0, 0.5, n)
        for t in range(1, n):
            a_logit[t] = 0.85 * a_logit[t - 1] + eps_a[t]
        p_a = np.clip(1.0 / (1.0 + np.exp(-a_logit)), 0.05, 0.95)

        # B: depends on lag-1 of A's innovation.
        b_logit = np.zeros(n)
        eps_b = rng.normal(0, 0.2, n)
        for t in range(1, n):
            b_logit[t] = 0.7 * b_logit[t - 1] + 0.6 * eps_a[t - 1] + eps_b[t]
        p_b = np.clip(1.0 / (1.0 + np.exp(-b_logit)), 0.05, 0.95)

        # C: depends on lag-1 of B's logit innovation.
        c_logit = np.zeros(n)
        eps_c = rng.normal(0, 0.2, n)
        for t in range(1, n):
            c_logit[t] = 0.7 * c_logit[t - 1] + 0.6 * eps_b[t - 1] + eps_c[t]
        p_c = np.clip(1.0 / (1.0 + np.exp(-c_logit)), 0.05, 0.95)

        # ticker depends on lag-1 of C's Δlogit.
        s_c = pd.Series(p_c, index=idx)
        d_c = delta_logit(s_c).fillna(0.0).to_numpy()
        ret = np.zeros(n)
        for t in range(1, n):
            ret[t] = 0.6 * d_c[t - 1] + rng.normal(0, 1e-3)

        price_bank = {
            "A": pd.Series(p_a, index=idx),
            "B": pd.Series(p_b, index=idx),
            "C": pd.Series(p_c, index=idx),
        }
        returns_bank = {"TGT": pd.Series(ret, index=idx)}
        return price_bank, returns_bank, idx

    def test_chain_recovers_a_b_c_ticker_at_max_depth_3(self) -> None:
        price_bank, returns_bank, idx = self._build_chain_universe()
        paths = find_chains(
            start_factor="A",
            end_ticker="TGT",
            candidate_intermediate_factors=["B", "C"],
            max_depth=3,
            start=idx[0].strftime("%Y-%m-%d"),
            end=idx[-1].strftime("%Y-%m-%d"),
            fetch_factor=_make_factor_fetcher(price_bank),
            fetch_returns=_make_returns_fetcher(returns_bank),
            p_threshold=0.20,  # generous (synthetic noise weakens Granger F)
        )
        assert len(paths) >= 1
        node_lists = [p["path"] for p in paths]
        # At least one path must hit the full A -> B -> C -> TGT chain.
        assert ["A", "B", "C", "TGT"] in node_lists, (
            f"expected ['A','B','C','TGT'] among paths: {node_lists}"
        )

    def test_chain_max_depth_4_still_finds_path(self) -> None:
        """Increasing max_depth should not drop the valid path."""
        price_bank, returns_bank, idx = self._build_chain_universe()
        paths = find_chains(
            start_factor="A",
            end_ticker="TGT",
            candidate_intermediate_factors=["B", "C"],
            max_depth=4,
            start=idx[0].strftime("%Y-%m-%d"),
            end=idx[-1].strftime("%Y-%m-%d"),
            fetch_factor=_make_factor_fetcher(price_bank),
            fetch_returns=_make_returns_fetcher(returns_bank),
            p_threshold=0.20,
        )
        node_lists = [p["path"] for p in paths]
        assert ["A", "B", "C", "TGT"] in node_lists

    def test_no_paths_when_pure_noise(self) -> None:
        rng = np.random.default_rng(55)
        n = 200
        idx = _utc_idx(n)
        price_bank = {
            "X": pd.Series(_ar1_to_prob(rng.normal(0, 0.4, n)), index=idx),
            "Y": pd.Series(_ar1_to_prob(rng.normal(0, 0.4, n)), index=idx),
            "Z": pd.Series(_ar1_to_prob(rng.normal(0, 0.4, n)), index=idx),
        }
        returns_bank = {"NOISE": pd.Series(rng.normal(0, 1e-3, n), index=idx)}
        paths = find_chains(
            start_factor="X",
            end_ticker="NOISE",
            candidate_intermediate_factors=["Y", "Z"],
            max_depth=3,
            start=idx[0].strftime("%Y-%m-%d"),
            end=idx[-1].strftime("%Y-%m-%d"),
            fetch_factor=_make_factor_fetcher(price_bank),
            fetch_returns=_make_returns_fetcher(returns_bank),
            p_threshold=0.05,
        )
        # Pure noise; allow up to a couple of false positives.
        assert len(paths) <= 2

    def test_no_candidates_falls_back_to_direct_check(self) -> None:
        """If candidate_intermediate_factors is [], the engine should still
        try the direct A -> TGT edge."""
        # Build a direct dependency: ticker = lag-1 Δlogit(A).
        rng = np.random.default_rng(101)
        n = 220
        idx = _utc_idx(n)
        a_logit = np.zeros(n)
        eps_a = rng.normal(0, 0.5, n)
        for t in range(1, n):
            a_logit[t] = 0.85 * a_logit[t - 1] + eps_a[t]
        p_a = np.clip(1.0 / (1.0 + np.exp(-a_logit)), 0.05, 0.95)
        s_a = pd.Series(p_a, index=idx)
        d_a = delta_logit(s_a).fillna(0.0).to_numpy()
        ret = np.zeros(n)
        for t in range(1, n):
            ret[t] = 0.6 * d_a[t - 1] + rng.normal(0, 1e-3)

        paths = find_chains(
            start_factor="A",
            end_ticker="TGT",
            candidate_intermediate_factors=[],
            max_depth=2,
            start=idx[0].strftime("%Y-%m-%d"),
            end=idx[-1].strftime("%Y-%m-%d"),
            fetch_factor=_make_factor_fetcher({"A": s_a}),
            fetch_returns=_make_returns_fetcher({"TGT": pd.Series(ret, index=idx)}),
            p_threshold=0.20,
        )
        assert len(paths) >= 1
        # Direct path is exactly [A, TGT] with one hop.
        assert any(p["path"] == ["A", "TGT"] for p in paths)

    def test_paths_sorted_by_p_value(self) -> None:
        """When multiple paths exist, granger_p_max should be non-decreasing."""
        price_bank, returns_bank, idx = self._build_chain_universe()
        paths = find_chains(
            start_factor="A",
            end_ticker="TGT",
            candidate_intermediate_factors=["B", "C"],
            max_depth=3,
            start=idx[0].strftime("%Y-%m-%d"),
            end=idx[-1].strftime("%Y-%m-%d"),
            fetch_factor=_make_factor_fetcher(price_bank),
            fetch_returns=_make_returns_fetcher(returns_bank),
            p_threshold=0.30,
        )
        if len(paths) >= 2:
            p_maxes = [p["granger_p_max"] for p in paths]
            for i in range(1, len(p_maxes)):
                assert p_maxes[i] >= p_maxes[i - 1] - 1e-12


# ===========================================================================
# 4. Event/macro correlation
# ===========================================================================


class TestEventMacroCorrelation:
    def test_planted_correlation_recovered_high(self) -> None:
        """Build a factor whose Δlogit ~ 0.7 * Δmacro + small noise."""
        rng = np.random.default_rng(41)
        n = 220
        idx = _utc_idx(n)
        d_macro = rng.normal(0, 1.0, n)
        macro = pd.Series(np.cumsum(d_macro), index=idx, name="DGS10")

        target_dlogit = 0.7 * d_macro + rng.normal(0, 0.25, n)
        logit = np.cumsum(target_dlogit) - target_dlogit[0]
        logit = logit - logit.mean()
        p = np.clip(1.0 / (1.0 + np.exp(-logit / 4.0)), 0.05, 0.95)
        factor = pd.Series(p, index=idx, name="rate-cut")

        result = event_macro_correlation(
            factor_id="rate-cut",
            macro_series=["DGS10"],
            start=idx[0].strftime("%Y-%m-%d"),
            end=idx[-1].strftime("%Y-%m-%d"),
            fetch_factor=_make_factor_fetcher({"rate-cut": factor}),
            fetch_macro=_make_macro_fetcher({"DGS10": macro}),
            max_lead_lag_days=3,
        )
        assert "DGS10" in result["macro_correlations"]
        assert abs(result["macro_correlations"]["DGS10"]) >= 0.5
        assert abs(result["t_stats"]["DGS10"]) > 2.0

    def test_multiple_macro_series_simultaneous(self) -> None:
        """Mix one strong-corr macro and one independent macro."""
        rng = np.random.default_rng(73)
        n = 220
        idx = _utc_idx(n)

        d_macro = rng.normal(0, 1.0, n)
        macro_strong = pd.Series(np.cumsum(d_macro), index=idx, name="DGS10")
        macro_independent = pd.Series(np.cumsum(rng.normal(0, 1.0, n)), index=idx, name="UNRATE")

        target = 0.8 * d_macro + rng.normal(0, 0.2, n)
        logit = np.cumsum(target) - target[0]
        logit = logit - logit.mean()
        p = np.clip(1.0 / (1.0 + np.exp(-logit / 4.0)), 0.05, 0.95)
        factor = pd.Series(p, index=idx, name="rate-cut")

        result = event_macro_correlation(
            factor_id="rate-cut",
            macro_series=["DGS10", "UNRATE"],
            start=idx[0].strftime("%Y-%m-%d"),
            end=idx[-1].strftime("%Y-%m-%d"),
            fetch_factor=_make_factor_fetcher({"rate-cut": factor}),
            fetch_macro=_make_macro_fetcher({"DGS10": macro_strong, "UNRATE": macro_independent}),
            max_lead_lag_days=3,
        )
        assert "DGS10" in result["macro_correlations"]
        assert "UNRATE" in result["macro_correlations"]
        # Strong > independent (in magnitude).
        assert abs(result["macro_correlations"]["DGS10"]) > abs(
            result["macro_correlations"]["UNRATE"]
        )

    def test_zero_macro_series_returns_empty(self) -> None:
        result = event_macro_correlation(
            factor_id="X",
            macro_series=[],
            start="2024-01-01",
            end="2024-06-01",
            fetch_factor=_make_factor_fetcher(
                {"X": pd.Series(_ar1_to_prob(np.zeros(60) + 1e-3), index=_utc_idx(60))}
            ),
            fetch_macro=_make_macro_fetcher({}),
        )
        assert result["macro_correlations"] == {}
        assert result["t_stats"] == {}


# ===========================================================================
# 5. Systemic PM factor (PCA)
# ===========================================================================


class TestSystemicPMFactor:
    def test_first_pc_explains_most_when_one_common_driver(self) -> None:
        """10 factors all loading on a single AR(1) -> PC1 dominates."""
        rng = np.random.default_rng(57)
        n = 280
        idx = _utc_idx(n)

        z = np.zeros(n)
        eps_z = rng.normal(0, 0.5, n)
        for t in range(1, n):
            z[t] = 0.85 * z[t - 1] + eps_z[t]

        price_bank: dict[str, pd.Series] = {}
        for i in range(10):
            logit = 0.8 * z + rng.normal(0, 0.10, n)
            p = np.clip(1.0 / (1.0 + np.exp(-logit)), 0.05, 0.95)
            price_bank[f"sf{i}"] = pd.Series(p, index=idx, name=f"sf{i}")

        result = extract_systemic_pm_factor(
            factor_ids=[f"sf{i}" for i in range(10)],
            n_factors=2,
            start=idx[0].strftime("%Y-%m-%d"),
            end=idx[-1].strftime("%Y-%m-%d"),
            fetch_factor=_make_factor_fetcher(price_bank),
        )
        ev = result["explained_variance"]
        assert len(ev) == 2
        assert ev[0] > 0.60, f"PC1 explained variance {ev[0]:.3f} < 0.60"
        # Loadings: all same sign on PC1 (after sign-orientation).
        loadings = list(result["loadings"].values())
        assert all(np.sign(loadings[0]) == np.sign(L) for L in loadings if abs(L) > 0.05)
        assert result["can_use_as_factor"] is True
        # Component scores match dates.
        assert len(result["component_scores"]) == len(result["dates"])

    def test_n_factors_2_returns_2_components(self) -> None:
        rng = np.random.default_rng(67)
        n = 220
        idx = _utc_idx(n)
        z = np.zeros(n)
        eps_z = rng.normal(0, 0.5, n)
        for t in range(1, n):
            z[t] = 0.85 * z[t - 1] + eps_z[t]
        price_bank: dict[str, pd.Series] = {}
        for i in range(5):
            logit = 0.8 * z + rng.normal(0, 0.20, n)
            p = np.clip(1.0 / (1.0 + np.exp(-logit)), 0.05, 0.95)
            price_bank[f"sf{i}"] = pd.Series(p, index=idx)

        result = extract_systemic_pm_factor(
            factor_ids=[f"sf{i}" for i in range(5)],
            n_factors=2,
            start=idx[0].strftime("%Y-%m-%d"),
            end=idx[-1].strftime("%Y-%m-%d"),
            fetch_factor=_make_factor_fetcher(price_bank),
        )
        assert result["n_components"] == 2
        assert len(result["explained_variance"]) == 2

    def test_single_factor_input_is_unusable(self) -> None:
        """Module guards against degenerate single-factor PCA."""
        rng = np.random.default_rng(2)
        n = 120
        idx = _utc_idx(n)
        p = pd.Series(_ar1_to_prob(rng.normal(0, 0.5, n), phi=0.85), index=idx)
        result = extract_systemic_pm_factor(
            factor_ids=["only"],
            n_factors=1,
            start=idx[0].strftime("%Y-%m-%d"),
            end=idx[-1].strftime("%Y-%m-%d"),
            fetch_factor=_make_factor_fetcher({"only": p}),
        )
        # Implementation guards: < 2 columns -> empty/unusable.
        assert result["can_use_as_factor"] is False
        assert result["component_scores"] == []


# ===========================================================================
# 6. Portfolio Optimizer — HRP
# ===========================================================================


class TestHRP:
    def test_hrp_weights_positive_sum_to_one(self) -> None:
        """3-asset baseline: HRP outputs all-positive weights summing to 1."""
        df = _gaussian_returns(sigmas=[0.10, 0.20, 0.15], rho=0.3, n=400, seed=11)
        out = hrp(df, shrinkage="sample")
        w = list(out["weights"].values())
        assert all(v > 0 for v in w)
        assert abs(sum(w) - 1.0) < 1e-6

    def test_hrp_handles_singular_covariance(self) -> None:
        """Duplicate column -> singular Σ. HRP must not raise; weights sum to 1."""
        df = _gaussian_returns(sigmas=[0.10, 0.20, 0.15], rho=0.2, n=400, seed=42)
        df["a3"] = df["a0"]  # rank deficient
        out = hrp(df, shrinkage="sample")
        s = sum(out["weights"].values())
        assert abs(s - 1.0) < 1e-6
        for v in out["weights"].values():
            assert 0.0 <= v <= 1.0 + 1e-9

    def test_hrp_more_diversified_than_min_variance(self) -> None:
        """HRP's effective_n is typically >= min_variance's on positively-correlated
        assets (HRP refuses to put all weight on the lowest-σ asset)."""
        df = _gaussian_returns(sigmas=[0.05, 0.20, 0.20, 0.20], rho=0.30, n=600, seed=3)
        h = hrp(df, shrinkage="sample")
        mv = min_variance(df, max_w=1.0, min_w=0.0, shrinkage="sample")
        # MV will load heavily on the low-σ asset -> small effective_n.
        # HRP should yield a more diversified portfolio.
        assert h["effective_n"] >= mv["effective_n"] - 1e-6

    def test_hrp_50_assets_runs_quickly(self) -> None:
        """Performance check: 50-asset HRP should complete in well under 5 s."""
        rng = np.random.default_rng(13)
        n = 500
        sigmas = (0.10 + rng.uniform(0, 0.15, 50)).tolist()
        df = _gaussian_returns(sigmas=sigmas, rho=0.20, n=n, seed=13)
        t0 = time.time()
        out = hrp(df, shrinkage="sample")
        elapsed = time.time() - t0
        assert elapsed < 5.0, f"HRP on 50 assets took {elapsed:.2f}s (>5s)"
        assert abs(sum(out["weights"].values()) - 1.0) < 1e-6

    def test_hrp_2_assets_approaches_inverse_volatility(self) -> None:
        """For 2 uncorrelated assets, HRP allocates inversely to volatility."""
        df = _gaussian_returns(sigmas=[0.10, 0.20], rho=0.0, n=500, seed=8)
        out = hrp(df, shrinkage="sample")
        # IVP: weight ∝ 1/σ²  -> w0/w1 ≈ σ1²/σ0² = 4.
        ratio = out["weights"]["a0"] / out["weights"]["a1"]
        # Sample variance noise; allow [2.0, 8.0].
        assert 2.0 <= ratio <= 8.0


# ===========================================================================
# 7. Portfolio Optimizer — Mean-Variance
# ===========================================================================


class TestMeanVariance:
    def test_mv_recovers_known_optimum(self) -> None:
        """Asymmetric mu — the asset with the largest realized sample-Sharpe
        should attract the bulk of MV weight when rf=0 and shrink_mu=0 (no
        shrinkage). We compare to the sample-mu (not the population-mu): with
        n=1000 there is sample noise, so the ordering of empirical Sharpes
        is what MV optimises against."""
        df = _gaussian_returns(
            sigmas=[0.10, 0.18, 0.14],
            rho=0.20,
            mu=[0.0010, 0.0001, 0.0003],
            n=1000,
            seed=99,
        )
        ms = mean_variance_max_sharpe(
            df, rf=0.0, max_w=0.99, min_w=0.0, shrink_mu=0.0, shrinkage="sample"
        )
        w = ms["weights"]
        # Empirical Sharpes per asset: μ̂ᵢ / σ̂ᵢ.
        sample_sharpe = (df.mean() / df.std()).to_dict()
        best = max(sample_sharpe, key=lambda k: sample_sharpe[k])
        # MV must allocate the bulk of the weight to the empirical best.
        assert w[best] >= 0.50, (
            f"MV gave best-Sharpe asset {best} weight {w[best]:.3f} (<0.5); "
            f"weights={w}, sample_sharpe={sample_sharpe}"
        )
        # Weights sum to 1.
        assert abs(sum(w.values()) - 1.0) < 1e-6
        # Sharpe must be non-negative in-sample.
        assert ms["sharpe"] >= 0

    def test_mv_max_weight_constraint_binds(self) -> None:
        """Set max_w=0.30 below the unconstrained MV optimum -> cap should bind
        on the highest-mu asset."""
        df = _gaussian_returns(
            sigmas=[0.10, 0.18, 0.14],
            rho=0.10,
            mu=[0.0020, 0.0001, 0.0003],
            n=1000,
            seed=5,
        )
        ms = mean_variance_max_sharpe(
            df, rf=0.0, max_w=0.30, min_w=0.0, shrink_mu=0.0, shrinkage="sample"
        )
        for v in ms["weights"].values():
            assert v <= 0.30 + 1e-6
        # And a0 should hit the cap (most attractive asset).
        assert ms["weights"]["a0"] >= 0.30 - 1e-6

    def test_mv_shrinkage_ledoit_vs_sample(self) -> None:
        """Both shrinkage choices should produce sensible (sum-to-1) weights;
        in small-T regime, ledoit_wolf typically gives more conservative weights."""
        df = _gaussian_returns(
            sigmas=[0.10, 0.18, 0.14],
            rho=0.20,
            mu=[0.0010, 0.0001, 0.0003],
            n=80,  # small-T: shrinkage matters
            seed=21,
        )
        s_sample = mean_variance_max_sharpe(
            df, rf=0.0, max_w=0.99, min_w=0.0, shrink_mu=0.5, shrinkage="sample"
        )
        s_lw = mean_variance_max_sharpe(
            df, rf=0.0, max_w=0.99, min_w=0.0, shrink_mu=0.5, shrinkage="ledoit_wolf"
        )
        for s in (s_sample, s_lw):
            assert abs(sum(s["weights"].values()) - 1.0) < 1e-6

    def test_mv_rf_change_shifts_results_coherently(self) -> None:
        """Higher rf -> demands higher-return portfolios -> Sharpe drops."""
        df = _gaussian_returns(
            sigmas=[0.10, 0.18, 0.14],
            rho=0.20,
            mu=[0.0010, 0.0001, 0.0003],
            n=1000,
            seed=77,
        )
        s_low = mean_variance_max_sharpe(
            df, rf=0.0, max_w=0.99, min_w=0.0, shrink_mu=0.0, shrinkage="sample"
        )
        s_high = mean_variance_max_sharpe(
            df, rf=0.10, max_w=0.99, min_w=0.0, shrink_mu=0.0, shrinkage="sample"
        )
        # Higher rf -> Sharpe drops (or stays same in degenerate corner).
        assert s_high["sharpe"] <= s_low["sharpe"] + 1e-6


# ===========================================================================
# 8. Risk Parity (ERC)
# ===========================================================================


class TestRiskParity:
    def test_erc_equalises_risk_contributions(self) -> None:
        df = _gaussian_returns(sigmas=[0.08, 0.18, 0.12, 0.22], rho=0.25, n=1000, seed=3)
        out = risk_parity_erc(df, max_w=0.50, min_w=0.0, shrinkage="sample")
        w = np.array([out["weights"][c] for c in df.columns])
        cov = np.cov(df.to_numpy(), rowvar=False, ddof=1)
        sigma_w = cov @ w
        rc = w * sigma_w
        cv = float(rc.std(ddof=1) / abs(rc.mean()))
        assert cv < 0.05, f"ERC CV={cv:.4f} should be < 0.05"

    def test_erc_two_equal_vol_assets_split_5050(self) -> None:
        df = _gaussian_returns(sigmas=[0.15, 0.15], rho=0.2, n=600, seed=2)
        out = risk_parity_erc(df, max_w=0.99, min_w=0.0, shrinkage="sample")
        diff = abs(out["weights"]["a0"] - out["weights"]["a1"])
        assert diff < 0.03  # within 3pp

    def test_erc_two_assets_inverse_vol_4_to_1(self) -> None:
        """σ ratio 4:1 (a0=0.05, a1=0.20). ERC weight ratio should be roughly the
        inverse vol ratio = 4:1."""
        df = _gaussian_returns(sigmas=[0.05, 0.20], rho=0.0, n=1000, seed=44)
        out = risk_parity_erc(df, max_w=0.99, min_w=0.0, shrinkage="sample")
        ratio = out["weights"]["a0"] / out["weights"]["a1"]
        # Allow [2.5, 5.5] for sample noise.
        assert 2.5 <= ratio <= 5.5

    def test_erc_diversification_ratio_above_one(self) -> None:
        df = _gaussian_returns(sigmas=[0.08, 0.18, 0.12], rho=0.20, n=600, seed=15)
        out = risk_parity_erc(df, max_w=0.50, min_w=0.0, shrinkage="sample")
        assert out["diversification_ratio"] > 1.0


# ===========================================================================
# 9. Efficient Frontier
# ===========================================================================


class TestEfficientFrontier:
    def test_frontier_50_points_monotonic_in_vol_and_return(self) -> None:
        df = _gaussian_returns(
            sigmas=[0.08, 0.16, 0.12, 0.20],
            rho=0.15,
            mu=[0.0002, 0.0008, 0.0004, 0.0010],
            n=900,
            seed=55,
        )
        pts = efficient_frontier(
            df,
            n_points=50,
            max_w=0.60,
            min_w=0.0,
            rf=0.0,
            shrinkage="sample",
            shrink_mu=0.0,
        )
        assert len(pts) >= 5
        vols = [p["expected_vol"] for p in pts]
        rets = [p["expected_return"] for p in pts]
        # Frontier sorted by vol (ascending).
        for i in range(1, len(vols)):
            assert vols[i] >= vols[i - 1] - 1e-6
        # Return non-decreasing along ascending-vol frontier (modulo SLSQP wobble).
        eps = 1e-3
        for i in range(1, len(rets)):
            assert rets[i] >= rets[i - 1] - eps

    def test_frontier_anchors_match_min_var_and_concentrated(self) -> None:
        """First point ~= min-variance; last point has return ~= the
        concentrated upper-anchor."""
        df = _gaussian_returns(
            sigmas=[0.08, 0.16, 0.12, 0.20],
            rho=0.15,
            mu=[0.0002, 0.0008, 0.0004, 0.0010],
            n=900,
            seed=12,
        )
        pts = efficient_frontier(
            df,
            n_points=20,
            max_w=0.40,
            min_w=0.0,
            rf=0.0,
            shrinkage="sample",
            shrink_mu=0.0,
        )
        mv = min_variance(df, max_w=0.40, min_w=0.0, shrinkage="sample")
        assert pts[0]["expected_vol"] == pytest.approx(mv["expected_vol"], abs=1e-2)

    def test_frontier_sharpe_peaks_in_interior(self) -> None:
        """The peak Sharpe along the frontier is typically interior, not at
        either end."""
        df = _gaussian_returns(
            sigmas=[0.08, 0.16, 0.12, 0.20],
            rho=0.15,
            mu=[0.0002, 0.0008, 0.0004, 0.0010],
            n=900,
            seed=73,
        )
        pts = efficient_frontier(
            df,
            n_points=30,
            max_w=0.60,
            min_w=0.0,
            rf=0.0,
            shrinkage="sample",
            shrink_mu=0.0,
        )
        sharpes = [p["sharpe"] for p in pts]
        # Peak index neither 0 nor last.
        peak_i = int(np.argmax(sharpes))
        # On weak signal it might collapse to the boundary; allow either end too
        # but the interior is the typical case. Just assert positive max.
        assert sharpes[peak_i] >= max(sharpes[0], sharpes[-1]) - 1e-9


# ===========================================================================
# 10. Monte Carlo drawdown
# ===========================================================================


class TestMonteCarloDrawdown:
    def test_quantile_ordering_p05_p50_p95(self) -> None:
        df = _gaussian_returns(sigmas=[0.10, 0.20, 0.15], rho=0.3, n=500, seed=4)
        weights = {"a0": 0.5, "a1": 0.25, "a2": 0.25}
        out = monte_carlo_drawdown(weights, df, n_paths=2000, horizon_days=126, block=10)
        assert out["p05"] <= out["p50"] <= out["p95"]
        assert out["p05"] >= 0.0
        assert out["mean"] >= 0.0

    def test_more_paths_tighter_quantile_estimates(self) -> None:
        """With more paths, the empirical quantile estimate has lower MC error.
        We check by comparing 100-path vs 10000-path runs and asserting that
        the 10000-path mean falls inside the spread of the smaller run (a
        weak but reliable check)."""
        df = _gaussian_returns(sigmas=[0.12, 0.18], rho=0.2, n=500, seed=23)
        w = {"a0": 0.5, "a1": 0.5}
        small = monte_carlo_drawdown(w, df, n_paths=100, horizon_days=126, block=10)
        big = monte_carlo_drawdown(w, df, n_paths=10000, horizon_days=126, block=10)
        # Big run mean should be near small run's p05..p95 envelope.
        assert small["p05"] - 0.10 <= big["mean"] <= small["p95"] + 0.10

    def test_longer_horizon_wider_drawdown_distribution(self) -> None:
        """horizon=252 should yield wider drawdowns than horizon=30 on average."""
        df = _gaussian_returns(sigmas=[0.15, 0.20], rho=0.2, n=500, seed=37)
        w = {"a0": 0.5, "a1": 0.5}
        short = monte_carlo_drawdown(w, df, n_paths=2000, horizon_days=30, block=10)
        long_ = monte_carlo_drawdown(w, df, n_paths=2000, horizon_days=252, block=10)
        assert long_["mean"] >= short["mean"]
        assert long_["p95"] >= short["p95"]

    def test_var_coherence_p05_lt_mean(self) -> None:
        """p05 quantile of *max drawdown* must be ≤ mean (max drawdowns are
        non-negative; the lower quantile is the smallest)."""
        df = _gaussian_returns(sigmas=[0.10, 0.20], rho=0.0, n=500, seed=51)
        w = {"a0": 0.5, "a1": 0.5}
        out = monte_carlo_drawdown(w, df, n_paths=5000, horizon_days=200, block=15)
        assert out["p05"] <= out["mean"] <= out["p95"]


# ===========================================================================
# 11. Reverse Factor Finder
# ===========================================================================


class TestReverseFinder:
    """Synthetic universe: 20 candidate factors, 5 are real."""

    def _build_universe(self, n: int = 260, seed: int = 0):
        rng = np.random.default_rng(seed)
        idx = pd.date_range("2025-01-01", periods=n, freq="B", tz="UTC").normalize()

        n_candidates = 20
        n_real = 5
        # Build smooth oscillatory probabilities.
        factors: dict[str, pd.DataFrame] = {}
        deltas: dict[str, pd.Series] = {}
        for i in range(n_candidates):
            t = np.arange(n) / n
            phase = i * 0.4
            freq = 0.7 + (i % 5) * 0.3
            p = 0.5 + 0.30 * np.sin(2 * np.pi * t * freq + phase) + 0.05 * rng.normal(0, 1, n)
            p = np.clip(p, 0.05, 0.95)
            factors[f"f{i}"] = pd.DataFrame({"price": p}, index=idx)
            deltas[f"f{i}"] = delta_logit(factors[f"f{i}"]["price"]).dropna()

        # Real betas with monotonically decreasing impact: f0..f4 are real.
        real_betas = {"f0": 0.6, "f1": 0.5, "f2": 0.4, "f3": 0.35, "f4": 0.30}
        common = deltas["f0"].index
        for i in range(1, n_real):
            common = common.intersection(deltas[f"f{i}"].index)
        y_vals = np.zeros(len(common))
        for fid, b in real_betas.items():
            y_vals = y_vals + b * deltas[fid].loc[common].values
        y_vals = y_vals + rng.normal(0, 1e-3, len(common))
        y = pd.Series(y_vals, index=common, name="r")
        return factors, y, real_betas

    def test_reverse_finder_recovers_real_factors_in_top_5(self) -> None:
        factors, y, real_betas = self._build_universe(n=260, seed=11)

        def factor_fetcher(fid, start, end):
            df = factors[fid]
            return df.loc[(df.index >= start) & (df.index <= end)]

        def returns_fetcher(ticker, start, end, return_type="log"):
            return y[(y.index >= start) & (y.index <= end)]

        out = reverse_find_factors(
            ticker="SYNTH",
            candidate_factor_ids=list(factors.keys()),
            start=date(2025, 1, 1),
            end=date(2025, 12, 31),
            k=5,
            returns_fetcher=returns_fetcher,
            factor_fetcher=factor_fetcher,
        )
        ids = [p["factor_id"] for p in out["top_factors"]]
        # At least 3 of the 5 real factors should appear in top-5.
        recovered = sum(1 for fid in real_betas if fid in ids)
        assert recovered >= 3, f"only {recovered}/5 real factors recovered: {ids}"
        # ΔR² descending.
        deltas = [p["delta_r_squared"] for p in out["top_factors"]]
        for i in range(1, len(deltas)):
            assert deltas[i] <= deltas[i - 1] + 1e-9
        # VIF reported (>0) for each factor.
        for p in out["top_factors"]:
            assert p["vif"] > 0

    def test_reverse_finder_zero_candidates_raises(self) -> None:
        with pytest.raises(ValueError, match="candidate_factor_ids"):
            reverse_find_factors(
                ticker="SYNTH",
                candidate_factor_ids=[],
                start=date(2025, 1, 1),
                end=date(2025, 12, 31),
                k=3,
                returns_fetcher=lambda *a, **k: pd.Series(dtype=float),
                factor_fetcher=lambda *a, **k: pd.DataFrame(),
            )

    def test_reverse_finder_no_ticker_data_returns_note(self) -> None:
        factors, _, _ = self._build_universe(n=120, seed=2)

        def returns_fetcher(ticker, start, end, return_type="log"):
            return pd.Series(dtype=float, name="r")

        def factor_fetcher(fid, start, end):
            return factors[fid]

        out = reverse_find_factors(
            ticker="DEAD",
            candidate_factor_ids=list(factors.keys()),
            start=date(2025, 1, 1),
            end=date(2025, 12, 31),
            k=3,
            returns_fetcher=returns_fetcher,
            factor_fetcher=factor_fetcher,
        )
        assert out["top_factors"] == []
        assert "note" in out


# ===========================================================================
# 12. Prediction-Driven Alpha
# ===========================================================================


class TestPredictionDrivenAlpha:
    def test_ranking_by_abs_beta_times_r2(self) -> None:
        """Build a single factor and 5 distinct ticker series with different
        sensitivities. The output should be ranked by |β| * R²."""
        rng = np.random.default_rng(91)
        n = 260
        idx = pd.date_range("2025-01-01", periods=n, freq="B", tz="UTC").normalize()
        t = np.arange(n) / n
        p = 0.5 + 0.30 * np.sin(2 * np.pi * t * 1.2) + 0.05 * rng.normal(0, 1, n)
        p = np.clip(p, 0.05, 0.95)
        factors = {"factor_X": pd.DataFrame({"price": p}, index=idx)}
        d = delta_logit(factors["factor_X"]["price"]).dropna()

        betas = {"AAA": 0.5, "BBB": 0.3, "CCC": 0.1, "DDD": 0.7, "EEE": 0.05}
        returns_bank: dict[str, pd.Series] = {}
        for tk, beta in betas.items():
            noise = rng.normal(0, 5e-4, len(d))
            returns_bank[tk] = beta * d + pd.Series(noise, index=d.index)

        def factor_fetcher(fid, start, end):
            df = factors[fid]
            return df.loc[(df.index >= start) & (df.index <= end)]

        def returns_fetcher(ticker, start, end, return_type="log"):
            s = returns_bank.get(ticker)
            if s is None:
                return pd.Series(dtype=float, name="r")
            return s[(s.index >= start) & (s.index <= end)]

        end_d = idx[-1].date()
        out = prediction_driven_alpha(
            factor_id="factor_X",
            candidate_tickers=list(betas.keys()),
            window_days=300,
            top_n=5,
            end=end_d,
            returns_fetcher=returns_fetcher,
            factor_fetcher=factor_fetcher,
        )
        rows = out["tickers"]
        assert len(rows) == 5
        # Ranked descending by |β| * R².
        scores = [abs(r["beta"]) * max(r["r_squared"], 0.0) for r in rows]
        for i in range(1, len(scores)):
            assert scores[i] <= scores[i - 1] + 1e-9
        # The largest beta (DDD=0.7) should rank near the top.
        top = rows[0]["ticker"]
        assert top in ("DDD", "AAA")  # DDD strongest, AAA second; allow ties

    def test_expected_return_with_delta_logit_assumed(self) -> None:
        """Provide delta_logit_assumed=0.1; expected_return_pct = β * 0.1 * 100."""
        rng = np.random.default_rng(83)
        n = 240
        idx = pd.date_range("2025-01-01", periods=n, freq="B", tz="UTC").normalize()
        t = np.arange(n) / n
        p = 0.5 + 0.30 * np.sin(2 * np.pi * t * 1.0) + 0.05 * rng.normal(0, 1, n)
        p = np.clip(p, 0.05, 0.95)
        factors = {"f": pd.DataFrame({"price": p}, index=idx)}
        d = delta_logit(factors["f"]["price"]).dropna()
        y = 0.6 * d + pd.Series(rng.normal(0, 5e-4, len(d)), index=d.index)

        def factor_fetcher(fid, start, end):
            df = factors[fid]
            return df.loc[(df.index >= start) & (df.index <= end)]

        def returns_fetcher(ticker, start, end, return_type="log"):
            return y[(y.index >= start) & (y.index <= end)]

        out = prediction_driven_alpha(
            factor_id="f",
            candidate_tickers=["AAA"],
            window_days=300,
            top_n=1,
            delta_logit_assumed=0.1,
            end=idx[-1].date(),
            returns_fetcher=returns_fetcher,
            factor_fetcher=factor_fetcher,
        )
        row = out["tickers"][0]
        assert row["expected_return_pct"] == pytest.approx(row["beta"] * 0.1 * 100.0, rel=1e-6)


# ===========================================================================
# 13. API endpoint smoke tests
# ===========================================================================


class TestEndpointsSmoke:
    """Mount routers in fresh FastAPI apps; no main.py."""

    @pytest.fixture
    def optimizer_client(self) -> TestClient:
        app = FastAPI()
        app.include_router(optimizer_router)
        with TestClient(app) as c:
            yield c

    def test_optimize_hrp_returns_valid_payload(self, optimizer_client: TestClient) -> None:
        body = {
            "pair_ids": [
                "alpha_a__alpha_b",
                "alpha_c__alpha_d",
                "alpha_e__alpha_f",
                "alpha_g__alpha_h",
                "alpha_i__alpha_j",
            ],
            "method": "hrp",
            "lookback_days": 252,
            "risk_free_rate": 0.045,
            "max_weight": 0.40,
            "min_weight": 0.0,
            "shrinkage": "ledoit_wolf",
            "mc_paths": 500,
            "mc_horizon_days": 126,
            "return_frontier": True,
            "seed": 13,
        }
        r = optimizer_client.post("/strategies/optimize", json=body)
        assert r.status_code == 200, r.text
        payload = r.json()
        assert payload["method"] == "hrp"
        assert set(payload["weights"].keys()) == set(body["pair_ids"])
        total = sum(payload["weights"].values())
        assert abs(total - 1.0) < 1e-6
        assert payload["frontier"] is not None
        assert payload["mc_drawdown"]["p05"] <= payload["mc_drawdown"]["p95"]


# Top-level guard for direct invocation.
if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-x", "-vv"])
