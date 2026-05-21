"""White's Reality Check, Hansen's SPA, and Romano-Wolf Stepwise SPA.

Data-snooping bias is the elephant in any "we tested 100 strategies and
found one with t = 2.5" claim: the maximum t over many tests is
mechanically larger than any single t even under the null. White (2000),
"A Reality Check for Data Snooping", *Econometrica* 68:1097-1126, gave
the first formal bootstrap test for the null

.. math::

    H_0: \\max_{k=1..K} \\mathbb{E}[ r_{k,t} - r_{bench,t} ] \\le 0

against the alternative that at least one strategy has a positive
expected excess return relative to the benchmark. Hansen (2005), "A Test
for Superior Predictive Ability", *Journal of Business & Economic
Statistics* 23:365-380, refines White's RC by recentering only the
non-poorly-performing strategies, which makes the test less conservative
when the family contains losers (the standard case).

Romano & Wolf (2005), "Stepwise Multiple Testing as Formalized Data
Snooping", *Econometrica* 73:1237-1282, generalise the framework to
identify the *full subset* of strategies that beat the benchmark, with
strong control of the family-wise error rate.

We implement all three on top of a common stationary block bootstrap
(Politis & Romano 1994) so the resamples are valid for serially-dependent
return series.

References
----------
White, H. (2000). "A Reality Check for Data Snooping",
    *Econometrica* 68:1097-1126.
Hansen, P. R. (2005). "A Test for Superior Predictive Ability",
    *Journal of Business & Economic Statistics* 23:365-380.
Romano, J. P. & Wolf, M. (2005). "Stepwise Multiple Testing as Formalized
    Data Snooping", *Econometrica* 73:1237-1282.
Politis, D. N. & Romano, J. P. (1994). "The Stationary Bootstrap",
    *Journal of the American Statistical Association* 89:1303-1313.
"""

from __future__ import annotations

import math

import numpy as np

# ---------------------------------------------------------------------------
# Stationary block bootstrap (Politis-Romano 1994)
# ---------------------------------------------------------------------------


def _stationary_bootstrap_indices(
    n: int,
    block_size: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw one stationary-bootstrap index sequence of length ``n``.

    Block lengths are i.i.d. Geometric(1/block_size); blocks wrap around
    the original series. Returns a length-``n`` integer index vector.
    """
    p = 1.0 / max(block_size, 1.0)
    out = np.empty(n, dtype=np.int64)
    i = 0
    while i < n:
        start = int(rng.integers(0, n))
        block_len = int(rng.geometric(p))
        block_len = max(1, block_len)
        for k in range(block_len):
            if i >= n:
                break
            out[i] = (start + k) % n
            i += 1
    return out


# ---------------------------------------------------------------------------
# White's Reality Check + Hansen's SPA
# ---------------------------------------------------------------------------


def whites_reality_check(
    strategy_returns_matrix: np.ndarray | list[list[float]],
    benchmark_returns: np.ndarray | list[float],
    *,
    n_bootstrap: int = 1000,
    block_size: int | float | None = None,
    seed: int = 42,
) -> dict[str, object]:
    """Bootstrap White's RC and Hansen's SPA on a panel of strategies.

    Args:
        strategy_returns_matrix: ``T x K`` array. Column k is strategy k's
            per-period return.
        benchmark_returns: length-``T`` benchmark return series. Use zeros
            for an excess-return-vs-zero test.
        n_bootstrap: number of bootstrap replications.
        block_size: average block length for the stationary bootstrap.
            Default = ``max(2, int(T**(1/3)))``.
        seed: RNG seed.

    Returns:
        Dict with ``n_strategies``, ``best_strategy_idx``,
        ``best_excess_return`` (the max sample mean of ``r_k - r_bench``),
        ``white_pvalue`` (the conventional RC p-value), ``hansen_spa_pvalue``
        (Hansen's recentered version), ``n_strategies_significant_at_05``
        (count of per-strategy nominal-0.05 t-test rejections; not FWER-
        controlled — see :func:`stepwise_spa` for that), and the
        ``bootstrap_max_distribution`` for downstream inspection.
    """
    rng = np.random.default_rng(seed)
    arr = np.asarray(strategy_returns_matrix, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"strategy_returns_matrix must be 2D, got shape {arr.shape}")
    n, k = arr.shape
    bench = np.asarray(benchmark_returns, dtype=float).ravel()
    if bench.size != n:
        raise ValueError(f"benchmark length {bench.size} != T {n}")
    if k < 1:
        raise ValueError("need at least one strategy")
    if n < 10:
        raise ValueError(f"need T >= 10, got {n}")

    excess = arr - bench[:, None]  # T x K
    means = excess.mean(axis=0)  # K
    sds = excess.std(axis=0, ddof=1)
    sds = np.where(sds <= 0, 1e-12, sds)

    test_stat = float(np.max(np.sqrt(n) * means))  # V_T
    best_idx = int(np.argmax(means))
    best_excess = float(means[best_idx])

    if block_size is None:
        block_size = max(2, int(round(n ** (1.0 / 3.0))))
    block_size = float(block_size)

    boot_max = np.empty(n_bootstrap, dtype=float)
    boot_max_hansen = np.empty(n_bootstrap, dtype=float)

    # Hansen's recentering threshold: only strategies whose sample mean
    # exceeds  -sd * sqrt(2 * log(log T) / T)  are kept "in the family"
    # at full strength; the rest are softly excluded.
    log_log_t = math.log(math.log(max(n, 3)))
    hansen_thresh = -sds * math.sqrt(2.0 * log_log_t / n)
    keep_full = means >= hansen_thresh  # bool, K

    for i in range(n_bootstrap):
        idx = _stationary_bootstrap_indices(n, block_size, rng)
        resampled = excess[idx, :]
        bmeans = resampled.mean(axis=0)
        # White RC: bootstrap quantity is sqrt(T) * (bmean_k - mean_k),
        # max over k (full family).
        boot_stats = np.sqrt(n) * (bmeans - means)
        boot_max[i] = float(np.max(boot_stats))
        # Hansen SPA: zero-out strategies that fail the recentering screen.
        hansen_stats = np.where(keep_full, boot_stats, np.sqrt(n) * bmeans)
        boot_max_hansen[i] = float(np.max(hansen_stats))

    white_pvalue = float(np.mean(boot_max >= test_stat))
    hansen_pvalue = float(np.mean(boot_max_hansen >= test_stat))

    # Per-strategy nominal-0.05 t-test (NOT FWER-controlled; just a count).
    t_stats = np.sqrt(n) * means / sds
    from scipy.stats import norm

    n_sig_05 = int(np.sum((1.0 - norm.cdf(t_stats)) < 0.05))

    return {
        "n_strategies": int(k),
        "n_obs": int(n),
        "best_strategy_idx": best_idx,
        "best_excess_return": best_excess,
        "test_statistic_v_t": test_stat,
        "white_pvalue": white_pvalue,
        "hansen_spa_pvalue": hansen_pvalue,
        "n_strategies_significant_at_05": n_sig_05,
        "block_size": float(block_size),
        "n_bootstrap": int(n_bootstrap),
    }


# ---------------------------------------------------------------------------
# Romano-Wolf Stepwise SPA
# ---------------------------------------------------------------------------


def stepwise_spa(
    strategy_returns_matrix: np.ndarray | list[list[float]],
    benchmark_returns: np.ndarray | list[float],
    *,
    alpha: float = 0.05,
    n_bootstrap: int = 1000,
    block_size: int | float | None = None,
    seed: int = 42,
    max_steps: int = 50,
) -> dict[str, object]:
    """Romano-Wolf (2005) stepwise procedure controlling FWER at ``alpha``.

    At each step:

    1. Form the bootstrap distribution of the *max* studentised
       excess-return over the **surviving** strategies (those not yet
       rejected).
    2. Find the (1-alpha) quantile of this max distribution.
    3. Reject any surviving strategy whose own studentised statistic
       exceeds that quantile.
    4. If no strategy is newly rejected, stop.

    The construction strongly controls FWER at level ``alpha`` under
    standard regularity (Romano & Wolf 2005, Theorem 3.1).

    Args:
        strategy_returns_matrix: ``T x K`` array.
        benchmark_returns: length-``T`` series.
        alpha: target FWER level.
        n_bootstrap: bootstrap replications per step.
        block_size: stationary-bootstrap block length.
        seed: RNG seed.
        max_steps: hard ceiling on stepwise iterations.

    Returns:
        Dict with ``rejected_strategy_indices`` (sorted ascending) and
        ``p_values_per_strategy`` (single-step bootstrap p-values, one per
        column; useful for ranking even outside the FWER cutoff).
    """
    rng = np.random.default_rng(seed)
    arr = np.asarray(strategy_returns_matrix, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"strategy_returns_matrix must be 2D, got shape {arr.shape}")
    n, k = arr.shape
    bench = np.asarray(benchmark_returns, dtype=float).ravel()
    if bench.size != n:
        raise ValueError(f"benchmark length {bench.size} != T {n}")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    excess = arr - bench[:, None]
    means = excess.mean(axis=0)
    sds = excess.std(axis=0, ddof=1)
    sds = np.where(sds <= 0, 1e-12, sds)
    t_stats = np.sqrt(n) * means / sds

    if block_size is None:
        block_size = max(2, int(round(n ** (1.0 / 3.0))))
    block_size = float(block_size)

    # Pre-draw bootstrap centred excess returns: B x K matrix of
    # (sqrt(T) * (bmean_k - mean_k)) / sd_k.
    boot_centered_t = np.empty((n_bootstrap, k), dtype=float)
    for i in range(n_bootstrap):
        idx = _stationary_bootstrap_indices(n, block_size, rng)
        resampled = excess[idx, :]
        bmeans = resampled.mean(axis=0)
        boot_centered_t[i, :] = np.sqrt(n) * (bmeans - means) / sds

    # Single-step p-values: P(boot_t_k >= t_stat_k)  per strategy.
    pvals_single = np.empty(k, dtype=float)
    for j in range(k):
        pvals_single[j] = float(np.mean(boot_centered_t[:, j] >= t_stats[j]))

    surviving = np.ones(k, dtype=bool)
    rejected: list[int] = []
    for _ in range(max_steps):
        if not surviving.any():
            break
        max_over_surv = np.max(boot_centered_t[:, surviving], axis=1)
        q = float(np.quantile(max_over_surv, 1.0 - alpha))
        # Reject any surviving strategy whose t-stat exceeds the quantile.
        new_rejects = [j for j in range(k) if surviving[j] and t_stats[j] > q]
        if not new_rejects:
            break
        for j in new_rejects:
            surviving[j] = False
            rejected.append(j)
    rejected.sort()

    return {
        "rejected_strategy_indices": rejected,
        "n_rejected": len(rejected),
        "p_values_per_strategy": [float(x) for x in pvals_single],
        "alpha": float(alpha),
        "n_strategies": int(k),
        "n_obs": int(n),
        "n_bootstrap": int(n_bootstrap),
        "block_size": float(block_size),
    }


__all__ = ["stepwise_spa", "whites_reality_check"]
