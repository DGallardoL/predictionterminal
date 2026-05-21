"""Advanced quant primitives: CUSUM structural-break, walk-forward
backtest, bootstrap Sharpe CI, permutation Sharpe p-value.

These are the "rigour layer" on top of pfm.cointegration / pfm.pairs.
Each function operates on a univariate spread series and is independent
of how that spread was constructed (Engle-Granger residuals, Kalman
innovations, basket residuals, etc.).

References:
    Brown, R., Durbin, J., Evans, J. (1975). "Techniques for Testing the
        Constancy of Regression Relationships over Time." JRSS-B 37, 149-192.
    Politis, D., Romano, J. (1994). "The Stationary Bootstrap."
        Journal of the American Statistical Association 89, 1303-1313.
    Lo, A. (2002). "The Statistics of Sharpe Ratios."
        Financial Analysts Journal 58(4), 36-52.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from math import sqrt

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────── CUSUM ────────────────────────────────────


@dataclass(frozen=True)
class CusumResult:
    """Output of :func:`cusum_test`.

    Attributes:
        n_obs: sample size.
        cusum_series: per-bar normalised cumulative sum of residuals.
        max_abs_cusum: peak absolute deviation.
        break_point: index of the maximum (≈ structural break date).
        threshold_95: critical value at 95% (Brown-Durbin-Evans).
        rejected: True if max_abs_cusum > threshold_95.
        verdict: ``"stable"`` / ``"break_detected"`` / ``"insufficient-data"``.
    """

    n_obs: int
    cusum_series: pd.Series
    max_abs_cusum: float
    break_point: pd.Timestamp | None
    threshold_95: float
    rejected: bool
    verdict: str


def cusum_test(spread: pd.Series) -> CusumResult:
    """Brown-Durbin-Evans CUSUM-OLS structural-break test.

    Compute the cumulative sum of standardised residuals
    ``W_t = (1/σ̂)·Σ_{i=1..t} ε_i / √n``. Under no break, W_t is a Brownian
    bridge and stays within the parabolic 95% band ``±a·(1 + 2·t/n)`` with
    ``a ≈ 0.948`` (5% level for the OLS-CUSUM statistic).

    A series excursion outside the band is evidence the spread underwent
    a level shift / regime change. The bar where ``|W_t|`` peaks is a good
    proxy for the break date.
    """
    s = spread.dropna()
    n = len(s)
    if n < 30:
        return CusumResult(
            n_obs=n,
            cusum_series=pd.Series(dtype=float),
            max_abs_cusum=float("nan"),
            break_point=None,
            threshold_95=float("nan"),
            rejected=False,
            verdict="insufficient-data",
        )
    eps = s.values - s.values.mean()
    sigma_hat = float(np.std(eps, ddof=1))
    if sigma_hat <= 0:
        return CusumResult(
            n_obs=n,
            cusum_series=pd.Series(0.0, index=s.index),
            max_abs_cusum=0.0,
            break_point=None,
            threshold_95=float("nan"),
            rejected=False,
            verdict="stable",
        )
    w = np.cumsum(eps) / (sigma_hat * sqrt(n))
    cusum = pd.Series(w, index=s.index, name="cusum")
    max_abs = float(np.max(np.abs(w)))
    bp_idx = int(np.argmax(np.abs(w)))
    bp = s.index[bp_idx]
    # 5% critical value for the OLS-CUSUM with parabolic boundary at the
    # mid-sample is approximately 0.948 (Brown-Durbin-Evans 1975 Table 1).
    crit = 0.948 * (1.0 + 2.0 * (bp_idx + 1) / n)
    rejected = max_abs > crit
    return CusumResult(
        n_obs=n,
        cusum_series=cusum,
        max_abs_cusum=max_abs,
        break_point=bp if rejected else None,
        threshold_95=float(crit),
        rejected=rejected,
        verdict="break_detected" if rejected else "stable",
    )


# ───────────────────────── walk-forward ───────────────────────────────


@dataclass(frozen=True)
class WalkForwardFold:
    """One fold of a walk-forward backtest."""

    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_sharpe: float
    test_sharpe: float
    n_train: int
    n_test: int


@dataclass(frozen=True)
class WalkForwardResult:
    """Output of :func:`walk_forward_backtest`."""

    n_obs: int
    n_folds: int
    folds: list[WalkForwardFold]
    train_sharpe_mean: float
    test_sharpe_mean: float
    test_sharpe_median: float
    test_sharpe_min: float
    test_sharpe_max: float
    test_sharpe_std: float
    stability: str  # "stable" / "borderline" / "unstable"


def walk_forward_backtest(
    spread: pd.Series,
    *,
    n_folds: int = 5,
    window: int = 20,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float = 4.0,
    annualisation: float = 252.0,
    embargo_size: int | None = None,
) -> WalkForwardResult:
    """Rolling-window K-fold backtest of a z-score pairs trade.

    Splits the spread into ``n_folds`` contiguous chunks. For each fold ``k``,
    train on the cumulative prefix (folds 0..k-1) — strictly speaking we
    don't refit anything because the strategy is parameter-free given the
    rolling z-window — and evaluate Sharpe on fold ``k``. Reports the
    distribution of *test* Sharpes; high mean + low std + min > 0 = robust.

    The implementation reuses the per-bar PnL formula from
    :mod:`pfm.pairs` so results are commensurable with single-shot
    backtest output.

    Walk-forward embargo (Lopez de Prado, 2018, *Advances in Financial Machine
    Learning*, ch. 7): financial features auto-correlate, so adjacent train
    bars leak information into a contiguous test window.  We therefore drop
    ``embargo_size`` bars on *both* sides of every test fold from the training
    set before computing the train Sharpe.  The default is
    ``max(5, ceil(fold_size * 0.10))``, large enough to cover typical
    rolling-window dependencies while preserving most of the training sample.

    Args:
        embargo_size: Number of training bars to embargo on each side of the
            test fold.  Pass ``0`` to disable the embargo (legacy behaviour).
            ``None`` selects the default ``max(5, ceil(fold_size * 0.10))``.
    """
    s = spread.dropna().sort_index()
    n = len(s)
    if n < n_folds * (window + 5):
        raise ValueError(
            f"walk_forward_backtest: need ≥ n_folds·(window+5) = "
            f"{n_folds * (window + 5)} bars, got {n}"
        )
    if entry_z <= exit_z:
        raise ValueError("entry_z must be greater than exit_z")
    if stop_z <= entry_z:
        raise ValueError("stop_z must be greater than entry_z")

    # Rolling z-score over the *full* spread (no look-ahead since the
    # rolling window only uses past bars).
    mu = s.rolling(window=window, min_periods=max(5, window // 2)).mean()
    sd = s.rolling(window=window, min_periods=max(5, window // 2)).std(ddof=1)
    z = (s - mu) / sd

    # Vectorised position generator that mimics the state machine.
    state = 0
    pos = np.zeros(n, dtype=int)
    for i, zi in enumerate(z.values):
        if np.isnan(zi):
            pos[i] = state
            continue
        if state == 0:
            if zi <= -entry_z:
                state = 1
            elif zi >= entry_z:
                state = -1
        elif state == 1:
            if abs(zi) < exit_z or zi <= -stop_z:
                state = 0
        elif state == -1 and (abs(zi) < exit_z or zi >= stop_z):
            state = 0
        pos[i] = state

    dspread = s.diff().fillna(0.0).values
    pnl = np.concatenate([[0.0], pos[:-1].astype(float)]) * dspread
    sqrt_ann = sqrt(annualisation)

    fold_size = n // n_folds
    if embargo_size is None:
        embargo_size = max(5, int(np.ceil(fold_size * 0.10)))
    if embargo_size < 0:
        raise ValueError("embargo_size must be non-negative")
    folds: list[WalkForwardFold] = []
    train_sh: list[float] = []
    test_sh: list[float] = []
    for k in range(n_folds):
        test_start_i = k * fold_size
        test_end_i = (k + 1) * fold_size if k < n_folds - 1 else n
        # Lopez de Prado embargo: drop ``embargo_size`` training bars on each
        # side of the test fold to suppress leakage from auto-correlated bars.
        train_pnl = np.concatenate(
            [
                pnl[: max(0, test_start_i - embargo_size)],
                pnl[test_end_i + embargo_size :],
            ]
        )
        test_pnl = pnl[test_start_i:test_end_i]
        train_sd = float(np.std(train_pnl, ddof=1)) if len(train_pnl) > 1 else 0.0
        test_sd = float(np.std(test_pnl, ddof=1)) if len(test_pnl) > 1 else 0.0
        train_s = (float(np.mean(train_pnl)) / train_sd) * sqrt_ann if train_sd > 0 else 0.0
        test_s = (float(np.mean(test_pnl)) / test_sd) * sqrt_ann if test_sd > 0 else 0.0
        folds.append(
            WalkForwardFold(
                fold=k,
                train_start=s.index[0],
                train_end=s.index[-1],
                test_start=s.index[test_start_i],
                test_end=s.index[test_end_i - 1],
                train_sharpe=train_s,
                test_sharpe=test_s,
                n_train=len(train_pnl),
                n_test=len(test_pnl),
            )
        )
        train_sh.append(train_s)
        test_sh.append(test_s)

    test_arr = np.array(test_sh)
    test_min = float(np.min(test_arr))
    test_std = float(np.std(test_arr, ddof=1)) if len(test_arr) > 1 else 0.0
    test_mean = float(np.mean(test_arr))
    if test_min > 0 and test_std < abs(test_mean):
        stability = "stable"
    elif test_min > -0.5 and test_std < 1.5 * abs(test_mean) + 0.5:
        stability = "borderline"
    else:
        stability = "unstable"

    return WalkForwardResult(
        n_obs=n,
        n_folds=n_folds,
        folds=folds,
        train_sharpe_mean=float(np.mean(train_sh)),
        test_sharpe_mean=test_mean,
        test_sharpe_median=float(np.median(test_arr)),
        test_sharpe_min=test_min,
        test_sharpe_max=float(np.max(test_arr)),
        test_sharpe_std=test_std,
        stability=stability,
    )


# ─────────────────────── bootstrap Sharpe CI ──────────────────────────


def _stationary_block_bootstrap(
    pnl: np.ndarray,
    *,
    block_size: int,
    n_iters: int,
    seed: int,
) -> np.ndarray:
    """Politis-Romano stationary block bootstrap. Returns (n_iters, n) array."""
    rng = np.random.default_rng(seed)
    n = len(pnl)
    out = np.empty((n_iters, n), dtype=float)
    for i in range(n_iters):
        bars: list[float] = []
        while len(bars) < n:
            start = int(rng.integers(0, n))
            blk = min(block_size, n - len(bars))
            idx = (np.arange(blk) + start) % n
            bars.extend(pnl[idx].tolist())
        out[i, :] = bars[:n]
    return out


@dataclass(frozen=True)
class BootstrapSharpeResult:
    """Output of :func:`bootstrap_sharpe_ci`."""

    sharpe_point: float
    sharpe_mean: float
    sharpe_std: float
    sharpe_ci_lo_90: float
    sharpe_ci_hi_90: float
    sharpe_ci_lo_95: float
    sharpe_ci_hi_95: float
    n_bootstrap: int
    block_size: int


def bootstrap_sharpe_ci(
    pnl: np.ndarray | pd.Series,
    *,
    annualisation: float = 252.0,
    n_iters: int = 500,
    block_size: int | None = None,
    seed: int = 42,
) -> BootstrapSharpeResult:
    """Stationary-block bootstrap CI for the Sharpe ratio of a per-bar PnL series."""
    arr = np.asarray(pnl)
    n = len(arr)
    if n < 20:
        raise ValueError(f"bootstrap_sharpe_ci: need ≥20 bars, got {n}")
    if block_size is None:
        block_size = max(5, int(round(np.sqrt(n))))
    sqrt_ann = sqrt(annualisation)
    pnl_std = float(np.std(arr, ddof=1))
    point_sharpe = (float(np.mean(arr)) / pnl_std) * sqrt_ann if pnl_std > 0 else 0.0
    boots = _stationary_block_bootstrap(arr, block_size=block_size, n_iters=n_iters, seed=seed)
    means = boots.mean(axis=1)
    stds = boots.std(axis=1, ddof=1)
    sharpes = np.where(stds > 0, (means / stds) * sqrt_ann, 0.0)
    ci_lo_90, ci_hi_90 = np.percentile(sharpes, [5, 95])
    ci_lo_95, ci_hi_95 = np.percentile(sharpes, [2.5, 97.5])
    return BootstrapSharpeResult(
        sharpe_point=point_sharpe,
        sharpe_mean=float(np.mean(sharpes)),
        sharpe_std=float(np.std(sharpes, ddof=1)),
        sharpe_ci_lo_90=float(ci_lo_90),
        sharpe_ci_hi_90=float(ci_hi_90),
        sharpe_ci_lo_95=float(ci_lo_95),
        sharpe_ci_hi_95=float(ci_hi_95),
        n_bootstrap=n_iters,
        block_size=block_size,
    )


# ──────────────────────── permutation Sharpe ──────────────────────────


@dataclass(frozen=True)
class PermutationSharpeResult:
    """Output of :func:`permutation_sharpe_test`."""

    real_sharpe: float
    null_sharpes: list[float]
    null_median: float
    null_pct95: float
    p_value: float
    n_iters: int


def permutation_sharpe_test(
    spread: np.ndarray | pd.Series,
    *,
    pnl_strategy_fn,
    annualisation: float = 252.0,
    n_iters: int = 200,
    seed: int = 42,
) -> PermutationSharpeResult:
    """Null distribution of Sharpe under random permutation of the spread.

    For each permutation, shuffle the spread's *first differences* (or
    multiply random ±1), reconstruct a synthetic spread, run the same
    strategy, compute Sharpe. ``p = P(Sharpe ≥ observed | null)`` is the
    fraction of nulls that beat the real Sharpe.

    ``pnl_strategy_fn`` accepts a 1-D spread array and returns a 1-D PnL
    array — typically a thin wrapper around the z-score state machine in
    pfm.pairs.
    """
    s = np.asarray(spread)
    n = len(s)
    if n < 30:
        raise ValueError(f"permutation_sharpe_test: need ≥30 bars, got {n}")
    rng = np.random.default_rng(seed)
    sqrt_ann = sqrt(annualisation)

    def _sharpe(pnl: np.ndarray) -> float:
        if len(pnl) < 2:
            return 0.0
        sd = float(np.std(pnl, ddof=1))
        return (float(np.mean(pnl)) / sd) * sqrt_ann if sd > 0 else 0.0

    real_pnl = pnl_strategy_fn(s)
    real_sharpe = _sharpe(real_pnl)

    diffs = np.diff(s)
    null_sharpes: list[float] = []
    for _ in range(n_iters):
        signs = rng.choice([1.0, -1.0], size=len(diffs))
        # Sign-flipped first differences preserve magnitude distribution
        # but break temporal structure. Reconstruct by cumsum from spread[0].
        permuted_diffs = diffs * signs
        permuted = np.concatenate([[s[0]], s[0] + np.cumsum(permuted_diffs)])
        # Just keep length n (we have n+1 with the initial concatenation).
        permuted = permuted[:n]
        try:
            pnl = pnl_strategy_fn(permuted)
            null_sharpes.append(_sharpe(pnl))
        except (ValueError, RuntimeError, ArithmeticError) as exc:
            # User-supplied pnl_strategy_fn may blow up on degenerate
            # permutations (zero variance, numpy LinAlg, etc.). Treat as
            # a 0-Sharpe sample so the null distribution stays well-defined.
            logger.debug("permutation pnl_fn raised on null sample: %s", exc)
            null_sharpes.append(0.0)

    p_value = float(np.mean([ns >= real_sharpe for ns in null_sharpes]))
    return PermutationSharpeResult(
        real_sharpe=real_sharpe,
        null_sharpes=null_sharpes,
        null_median=float(np.median(null_sharpes)),
        null_pct95=float(np.percentile(null_sharpes, 95)),
        p_value=p_value,
        n_iters=n_iters,
    )


__all__ = [
    "BootstrapSharpeResult",
    "CusumResult",
    "PermutationSharpeResult",
    "WalkForwardFold",
    "WalkForwardResult",
    "bootstrap_sharpe_ci",
    "cusum_test",
    "permutation_sharpe_test",
    "walk_forward_backtest",
]
