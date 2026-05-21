"""Tests for ``pfm.alpha_hunter`` — cross-pair alpha-discovery orchestrator.

Strategy: construct synthetic price dictionaries whose pairwise cointegration
structure is known by construction, then verify the report counts and labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pfm.alpha_hunter import AlphaHit, run_alpha_hunter

SEED = 42


def _idx(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")


def _random_walk(n: int, sigma: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(0, sigma, size=n))


def _ar1_series(n: int, rho: float, sigma: float, seed: int) -> np.ndarray:
    """Stationary AR(1) noise (cointegrating residual)."""
    rng = np.random.default_rng(seed)
    eps = rng.normal(0, sigma, size=n)
    out = np.empty(n)
    out[0] = eps[0] / np.sqrt(max(1.0 - rho * rho, 1e-12))
    for t in range(1, n):
        out[t] = rho * out[t - 1] + eps[t]
    return out


def test_no_cointegrated_pairs_filtered() -> None:
    """4 independent random walks → no pair survives ADF → n_real_alpha=0."""
    n = 250
    idx = _idx(n)
    prices: dict[str, pd.Series] = {
        f"f{i}": pd.Series(_random_walk(n, 0.02, seed=SEED + i), index=idx) for i in range(4)
    }

    report = run_alpha_hunter(prices, seed=SEED)

    assert report.n_factors == 4
    assert report.n_pairs_total == 6  # C(4, 2)
    assert report.n_real_alpha == 0
    # No hits at all should be REAL_ALPHA; verdicts must reflect filtering.
    assert all(h.verdict != "REAL_ALPHA" for h in report.hits)


def test_synthetic_cointegrated_pair_detected() -> None:
    """Series (c, d) cointegrated by construction; (a, b) independent.

    Asserts at least one REAL_ALPHA hit and that (c, d) is among the
    REAL_ALPHA pairs.
    """
    n = 400
    idx = _idx(n)
    # Two independent random walks.
    a = _random_walk(n, 0.02, seed=SEED + 1)
    b = _random_walk(n, 0.02, seed=SEED + 2)
    # c is a random walk; d = 0.7 * c + small stationary AR(1) noise.
    c = _random_walk(n, 0.02, seed=SEED + 3)
    eps = _ar1_series(n, rho=0.3, sigma=0.005, seed=SEED + 4)
    d = 0.7 * c + eps

    prices: dict[str, pd.Series] = {
        "a": pd.Series(a, index=idx),
        "b": pd.Series(b, index=idx),
        "c": pd.Series(c, index=idx),
        "d": pd.Series(d, index=idx),
    }

    report = run_alpha_hunter(
        prices,
        seed=SEED,
        perm_n_iters=100,  # smaller for test runtime
    )

    assert report.n_factors == 4
    assert report.n_pairs_total == 6
    assert report.n_pairs_passed_adf >= 1

    real_hits = [h for h in report.hits if h.verdict == "REAL_ALPHA"]
    # The (c, d) pair was constructed to be cointegrated; expect at least
    # one REAL_ALPHA, and (c, d) should be among them.
    assert report.n_real_alpha >= 1
    real_pairs = {tuple(sorted((h.a_id, h.b_id))) for h in real_hits}
    assert ("c", "d") in real_pairs


def test_max_pairs_caps_evaluation() -> None:
    """6 series with max_pairs=3 → only 3 pairs evaluated."""
    n = 200
    idx = _idx(n)
    prices: dict[str, pd.Series] = {
        f"s{i}": pd.Series(_random_walk(n, 0.02, seed=SEED + i), index=idx) for i in range(6)
    }

    report = run_alpha_hunter(prices, max_pairs=3, seed=SEED)

    # C(6, 2) = 15, but capped at 3.
    assert report.n_factors == 6
    assert report.n_pairs_total == 3


def test_min_obs_filter() -> None:
    """A series shorter than min_obs is not paired."""
    n_long = 250
    n_short = 20  # well below default min_obs=60
    long_idx = _idx(n_long)
    short_idx = _idx(n_short)

    prices: dict[str, pd.Series] = {
        "long_a": pd.Series(_random_walk(n_long, 0.02, seed=SEED + 1), index=long_idx),
        "long_b": pd.Series(_random_walk(n_long, 0.02, seed=SEED + 2), index=long_idx),
        "short_c": pd.Series(_random_walk(n_short, 0.02, seed=SEED + 3), index=short_idx),
    }

    report = run_alpha_hunter(prices, min_obs=60, seed=SEED)

    # 3 pairs total are formed, but any pair containing short_c is dropped
    # by the min_obs pre-check before EG runs.
    assert report.n_pairs_total == 3
    # Only pairs NOT involving short_c can produce hits.
    short_in_hits = [h for h in report.hits if "short_c" in (h.a_id, h.b_id)]
    assert short_in_hits == []


def test_verdict_label_consistency() -> None:
    """REAL_ALPHA iff perm_p ≤ perm_p_threshold AND oos_sharpe ≥ perm_oos_sharpe_threshold."""
    n = 400
    idx = _idx(n)
    # Build a mixed factor set: one cointegrated pair + a couple of random walks
    # to exercise multiple verdict branches.
    a = _random_walk(n, 0.02, seed=SEED + 1)
    b = _random_walk(n, 0.02, seed=SEED + 2)
    c = _random_walk(n, 0.02, seed=SEED + 3)
    eps = _ar1_series(n, rho=0.3, sigma=0.005, seed=SEED + 4)
    d = 0.7 * c + eps

    prices: dict[str, pd.Series] = {
        "a": pd.Series(a, index=idx),
        "b": pd.Series(b, index=idx),
        "c": pd.Series(c, index=idx),
        "d": pd.Series(d, index=idx),
    }

    perm_p_thresh = 0.10
    oos_thresh = 1.0
    report = run_alpha_hunter(
        prices,
        seed=SEED,
        perm_p_threshold=perm_p_thresh,
        perm_oos_sharpe_threshold=oos_thresh,
        perm_n_iters=100,
    )

    for h in report.hits:
        assert isinstance(h, AlphaHit)
        if h.verdict == "REAL_ALPHA":
            # REAL_ALPHA requires both: perm test was run and passed.
            assert h.perm_p is not None
            assert h.perm_p <= perm_p_thresh
            assert h.oos_sharpe >= oos_thresh
        elif h.verdict == "marginal":
            # Permutation ran (above oos threshold) but perm_p exceeded threshold.
            assert h.perm_p is not None
            assert h.perm_p > perm_p_thresh
            assert h.oos_sharpe >= oos_thresh
        elif h.verdict == "promising":
            # Did not trigger the permutation gate.
            assert h.oos_sharpe < oos_thresh
            assert h.perm_p is None
