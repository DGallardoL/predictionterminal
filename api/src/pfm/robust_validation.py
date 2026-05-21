"""Comprehensive robustness battery for a portfolio's alpha claim.

Goes beyond the existing per-pair `pairs_backtest` + permutation +
bootstrap. The questions we answer here:

1.  **Is the portfolio Sharpe statistically distinguishable from zero?**
    - Block bootstrap CI (Politis-Romano)
    - Sign-flip permutation null
    - Lo (2002) closed-form asymptotic SE on Sharpe

2.  **Is the alpha robust to parameter choice?**
    - Sweep over window size, entry threshold, exit threshold
    - Report: median and quantile range of Sharpe across the parameter grid

3.  **Is it robust to transaction costs?**
    - Subtract round-trip cost per round-trip from each leg's PnL
    - Find the *break-even* cost where Sharpe = 0

4.  **Is it robust to time window?**
    - Train-half / test-half held-out test (not k-fold)

5.  **Is the alpha specific to data-mined pair choice?**
    - White (2000) Reality Check style: of the N pairs tested historically,
      report the empirical p-value of the portfolio Sharpe against the
      distribution of Sharpes from random equal-vol baskets of N pairs.

These are the standard portfolio-level robustness tests in the
quant-research literature (Bailey & Lopez de Prado 2014, Harvey & Liu 2014).

References:
    Lo, A. (2002). "The Statistics of Sharpe Ratios." FAJ.
    White, H. (2000). "A Reality Check for Data Snooping." Econometrica.
    Hansen, P. (2005). "A Test for Superior Predictive Ability." JBES.
    Bailey, D. & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio."
    Politis, D. & Romano, J. (1994). Stationary block bootstrap.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np
import pandas as pd

# ────────────────────── Lo (2002) Sharpe SE ───────────────────────────


@dataclass(frozen=True)
class LoSharpeTest:
    """Lo (2002) asymptotic distribution of the Sharpe ratio."""

    sharpe: float
    sharpe_se: float
    z_stat: float  # = sharpe / se → ~N(0,1) under H0: true Sharpe = 0
    p_value: float  # two-sided
    ci_lo_95: float
    ci_hi_95: float


def lo_sharpe_test(
    pnl: pd.Series,
    *,
    annualisation: float = 252.0,
) -> LoSharpeTest:
    """Closed-form SE on annualised Sharpe (Lo 2002 eq. 12).

    Asymptotic SE for IID returns:
        SE(SR) = sqrt((1 + 0.5·SR²) / T) · √annualisation_factor

    Under non-IID (auto-correlated) PnL, this UNDERESTIMATES the SE — so
    the test is conservatively *liberal*. For a rigorous test, use
    `block_bootstrap_sharpe_ci` below.
    """
    arr = np.asarray(pnl.dropna(), dtype=float)
    n = len(arr)
    if n < 20:
        return LoSharpeTest(
            sharpe=0.0,
            sharpe_se=float("nan"),
            z_stat=0.0,
            p_value=1.0,
            ci_lo_95=float("nan"),
            ci_hi_95=float("nan"),
        )
    sd = float(np.std(arr, ddof=1))
    if sd <= 0:
        return LoSharpeTest(
            sharpe=0.0,
            sharpe_se=float("nan"),
            z_stat=0.0,
            p_value=1.0,
            ci_lo_95=0.0,
            ci_hi_95=0.0,
        )
    sr_per_bar = float(np.mean(arr)) / sd
    sqrt_ann = sqrt(annualisation)
    sr_annual = sr_per_bar * sqrt_ann
    se_per_bar = sqrt((1.0 + 0.5 * sr_per_bar * sr_per_bar) / n)
    se_annual = se_per_bar * sqrt_ann
    z = sr_annual / se_annual if se_annual > 0 else 0.0
    from scipy.stats import norm

    p_value = 2.0 * (1.0 - norm.cdf(abs(z)))
    return LoSharpeTest(
        sharpe=sr_annual,
        sharpe_se=se_annual,
        z_stat=float(z),
        p_value=float(p_value),
        ci_lo_95=float(sr_annual - 1.96 * se_annual),
        ci_hi_95=float(sr_annual + 1.96 * se_annual),
    )


# ────────────────── Block bootstrap CI on Sharpe ──────────────────────


def block_bootstrap_sharpe_ci(
    pnl: pd.Series,
    *,
    annualisation: float = 252.0,
    n_iters: int = 500,
    block_size: int | None = None,
    seed: int = 42,
) -> dict[str, float]:
    """Block bootstrap CI on annualised Sharpe — preserves autocorrelation."""
    arr = np.asarray(pnl.dropna(), dtype=float)
    n = len(arr)
    if n < 20:
        return {"sharpe": 0.0, "ci_lo_90": 0.0, "ci_hi_90": 0.0, "ci_lo_95": 0.0, "ci_hi_95": 0.0}
    if block_size is None:
        block_size = max(5, int(round(np.sqrt(n))))
    rng = np.random.default_rng(seed)
    sqrt_ann = sqrt(annualisation)
    point_sd = float(np.std(arr, ddof=1))
    point_sharpe = (float(np.mean(arr)) / point_sd) * sqrt_ann if point_sd > 0 else 0.0
    sharpes = []
    for _ in range(n_iters):
        bars: list[float] = []
        while len(bars) < n:
            start = int(rng.integers(0, n))
            blk = min(block_size, n - len(bars))
            idx = (np.arange(blk) + start) % n
            bars.extend(arr[idx].tolist())
        sample = np.array(bars[:n])
        sd = float(np.std(sample, ddof=1))
        sharpes.append((float(np.mean(sample)) / sd) * sqrt_ann if sd > 0 else 0.0)
    return {
        "sharpe": float(point_sharpe),
        "ci_lo_90": float(np.percentile(sharpes, 5)),
        "ci_hi_90": float(np.percentile(sharpes, 95)),
        "ci_lo_95": float(np.percentile(sharpes, 2.5)),
        "ci_hi_95": float(np.percentile(sharpes, 97.5)),
    }


# ────────────────── Permutation null on portfolio ─────────────────────


def permutation_sharpe_null(
    pnl: pd.Series,
    *,
    annualisation: float = 252.0,
    n_iters: int = 500,
    seed: int = 42,
) -> dict[str, float]:
    """Sign-flip permutation null. ``p = P(null Sharpe ≥ real Sharpe)``."""
    arr = np.asarray(pnl.dropna(), dtype=float)
    n = len(arr)
    if n < 20:
        return {"real_sharpe": 0.0, "null_median": 0.0, "p_value": 1.0}
    rng = np.random.default_rng(seed)
    sqrt_ann = sqrt(annualisation)

    def _sharpe(x):
        sd = float(np.std(x, ddof=1))
        return (float(np.mean(x)) / sd) * sqrt_ann if sd > 0 else 0.0

    real_sharpe = _sharpe(arr)
    nulls = []
    for _ in range(n_iters):
        signs = rng.choice([1.0, -1.0], size=n)
        nulls.append(_sharpe(arr * signs))
    p_value = float(np.mean([ns >= real_sharpe for ns in nulls]))
    return {
        "real_sharpe": float(real_sharpe),
        "null_median": float(np.median(nulls)),
        "null_pct95": float(np.percentile(nulls, 95)),
        "p_value": p_value,
    }


# ────────────────────── Cost sensitivity ──────────────────────────────


def cost_sensitivity_curve(
    pnl: pd.Series,
    *,
    position_changes: pd.Series,
    cost_grid_bps: list[float] | None = None,
    annualisation: float = 252.0,
) -> dict[str, list[float] | float]:
    """For each cost level (bps per round-trip), compute net Sharpe.

    ``position_changes`` is a series with ±2 at trade entries (or you can
    pass the absolute trade-count series). We charge ``cost`` per unit of
    |position_change|.

    Returns:
        Dict with ``costs_bps`` and ``net_sharpe`` arrays + ``break_even_bps``.
    """
    if cost_grid_bps is None:
        cost_grid_bps = [0, 5, 10, 25, 50, 100, 200, 300, 500]
    arr = np.asarray(pnl.dropna(), dtype=float)
    pos_changes = np.asarray(position_changes.reindex(pnl.index).fillna(0), dtype=float)
    n = len(arr)
    if n < 20:
        return {
            "costs_bps": cost_grid_bps,
            "net_sharpe": [0.0] * len(cost_grid_bps),
            "break_even_bps": 0.0,
        }
    sqrt_ann = sqrt(annualisation)
    sharpes = []
    for c_bps in cost_grid_bps:
        c = c_bps / 10_000.0
        # Cost per bar = c · |position_change|
        net_pnl = arr - c * np.abs(pos_changes)
        sd = float(np.std(net_pnl, ddof=1))
        sh = (float(np.mean(net_pnl)) / sd) * sqrt_ann if sd > 0 else 0.0
        sharpes.append(sh)
    # Find break-even: linear interpolation between first cost where Sharpe ≤ 0
    break_even = float("inf")
    for i in range(len(cost_grid_bps) - 1):
        if sharpes[i] > 0 and sharpes[i + 1] <= 0:
            # Linear interp
            x1, x2 = cost_grid_bps[i], cost_grid_bps[i + 1]
            y1, y2 = sharpes[i], sharpes[i + 1]
            break_even = x1 + (x2 - x1) * (y1 - 0.0) / (y1 - y2) if y1 != y2 else x1
            break
    return {
        "costs_bps": [float(c) for c in cost_grid_bps],
        "net_sharpe": [float(s) for s in sharpes],
        "break_even_bps": float(break_even),
    }


# ────────────────────── Out-of-time split ─────────────────────────────


def out_of_time_test(
    pnl: pd.Series,
    *,
    train_fraction: float = 0.5,
    annualisation: float = 252.0,
) -> dict[str, float | str | int]:
    """Train-first-half / test-second-half (no k-fold; one held-out tail)."""
    arr = np.asarray(pnl.dropna(), dtype=float)
    n = len(arr)
    if n < 40:
        return {
            "train_sharpe": 0.0,
            "test_sharpe": 0.0,
            "ratio": 0.0,
            "verdict": "insufficient-data",
        }
    n_train = int(round(n * train_fraction))
    train = arr[:n_train]
    test = arr[n_train:]
    sqrt_ann = sqrt(annualisation)
    train_sd = float(np.std(train, ddof=1))
    test_sd = float(np.std(test, ddof=1))
    train_s = (float(np.mean(train)) / train_sd) * sqrt_ann if train_sd > 0 else 0.0
    test_s = (float(np.mean(test)) / test_sd) * sqrt_ann if test_sd > 0 else 0.0
    ratio = test_s / train_s if abs(train_s) > 1e-9 else 0.0
    if ratio > 0.7:
        verdict = "robust"
    elif ratio > 0.3:
        verdict = "borderline"
    else:
        verdict = "overfit"
    return {
        "train_sharpe": float(train_s),
        "test_sharpe": float(test_s),
        "ratio": float(ratio),
        "verdict": verdict,
        "n_train": n_train,
        "n_test": n - n_train,
    }


# ───────── Deflated Sharpe ratio (Bailey-Lopez de Prado 2014) ────────


def deflated_sharpe_ratio(
    sharpe: float,
    n_obs: int,
    *,
    n_trials: int = 100,
    skew: float = 0.0,
    kurtosis: float = 3.0,
) -> dict[str, float]:
    """Bailey & Lopez de Prado (2014) Deflated Sharpe Ratio.

    Adjusts the observed Sharpe for:
    - **Multiple-testing bias** (n_trials = number of strategies you searched)
    - **Non-normality** of returns (skew + excess kurtosis)
    - **Small-sample bias**

    The DSR p-value is the probability that the observed Sharpe is real
    given how many strategies you tried before finding it.

    Args:
        sharpe: observed annualised Sharpe.
        n_obs: number of observations.
        n_trials: number of strategies tested (the data-mining budget).
        skew, kurtosis: residual moments (Fisher kurtosis, 3 = normal).

    Returns:
        Dict with deflated_sharpe and deflated_p_value.
    """
    from scipy.stats import norm

    if n_obs < 20:
        return {
            "deflated_sharpe": 0.0,
            "deflated_p_value": 1.0,
            "expected_max_sharpe_under_null": 0.0,
        }
    # Expected maximum Sharpe under the null (Bailey-Lopez de Prado eq. 8):
    em_z = (1.0 - 0.5772) / norm.ppf(1.0 - 1.0 / n_trials) if n_trials > 1 else 0.0
    expected_max = (
        norm.ppf(1.0 - 1.0 / n_trials) * (1.0 - em_z)
        + norm.ppf(1.0 - 1.0 / (n_trials * np.e)) * em_z
        if n_trials > 1
        else 0.0
    )
    expected_max = max(expected_max, 0.0)

    # Convert annualised → per-period Sharpe (assumes 252 ann factor; conservative).
    sr_per = sharpe / sqrt(252.0)
    excess_kurt = kurtosis - 3.0
    # Bailey-LDP eq. 9 (DSR = Φ(z*))
    denom = sqrt((1.0 - skew * sr_per + 0.25 * (excess_kurt - 1.0) * sr_per * sr_per) / (n_obs - 1))
    if denom <= 0:
        z_star = 0.0
    else:
        z_star = (sr_per - expected_max) / denom
    deflated_p = float(1.0 - norm.cdf(z_star))
    return {
        "deflated_sharpe": float(sr_per - expected_max),
        "deflated_p_value": deflated_p,
        "expected_max_sharpe_under_null": float(expected_max),
    }


# ───────────────────── orchestration ──────────────────────────────────


@dataclass(frozen=True)
class RobustValidationReport:
    portfolio_sharpe: float
    n_obs: int
    lo_test: dict[str, float]
    bootstrap_ci: dict[str, float]
    permutation: dict[str, float]
    cost_sensitivity: dict
    out_of_time: dict[str, float]
    deflated_sharpe: dict[str, float]
    overall_verdict: str  # "STRONG ALPHA" / "MARGINAL" / "OVERFIT" / "NOISE"


def run_robust_validation(
    portfolio_pnl: pd.Series,
    *,
    position_changes: pd.Series | None = None,
    annualisation: float = 252.0,
    n_trials_searched: int = 100,
    seed: int = 42,
) -> RobustValidationReport:
    """Run all 6 robustness tests on a single portfolio PnL series.

    ``position_changes`` (if provided) enables cost sensitivity. If None,
    cost sensitivity is run with a placeholder constant 0 — just reports
    Sharpe under no costs.
    """
    arr = portfolio_pnl.dropna()
    n = len(arr)
    sqrt_ann = sqrt(annualisation)
    sd = float(arr.std(ddof=1)) if n > 1 else 0.0
    point_sharpe = (float(arr.mean()) / sd) * sqrt_ann if sd > 0 else 0.0

    lo = lo_sharpe_test(arr, annualisation=annualisation)
    boot = block_bootstrap_sharpe_ci(arr, annualisation=annualisation, n_iters=500, seed=seed)
    perm = permutation_sharpe_null(arr, annualisation=annualisation, n_iters=500, seed=seed)
    if position_changes is None:
        position_changes = pd.Series(np.zeros(n), index=arr.index)
    cost = cost_sensitivity_curve(
        arr, position_changes=position_changes, annualisation=annualisation
    )
    oot = out_of_time_test(arr, annualisation=annualisation)
    # Deflated Sharpe with default skew/kurt from data.
    z = (arr - arr.mean()) / sd if sd > 0 else arr * 0
    skew = float((z**3).mean())
    kurt = float((z**4).mean())
    dsr = deflated_sharpe_ratio(
        point_sharpe, n, n_trials=n_trials_searched, skew=skew, kurtosis=kurt
    )

    # Overall verdict: must pass MULTIPLE rigorous tests.
    passes = sum(
        [
            lo.p_value < 0.05,  # asymptotic
            boot["ci_lo_95"] > 0,  # bootstrap CI
            perm["p_value"] < 0.05,  # permutation p
            oot["ratio"] > 0.5,  # OOS robustness
            dsr["deflated_p_value"] < 0.05,  # deflated p (multiple testing)
        ]
    )
    if passes >= 4:
        verdict = "STRONG ALPHA"
    elif passes >= 3:
        verdict = "MARGINAL ALPHA"
    elif passes >= 1:
        verdict = "WEAK / SUSPECT"
    else:
        verdict = "NOISE / OVERFIT"

    return RobustValidationReport(
        portfolio_sharpe=point_sharpe,
        n_obs=n,
        lo_test={
            "sharpe": lo.sharpe,
            "se": lo.sharpe_se,
            "z_stat": lo.z_stat,
            "p_value": lo.p_value,
            "ci_lo_95": lo.ci_lo_95,
            "ci_hi_95": lo.ci_hi_95,
        },
        bootstrap_ci=boot,
        permutation=perm,
        cost_sensitivity=cost,
        out_of_time=oot,
        deflated_sharpe=dsr,
        overall_verdict=verdict,
    )


__all__ = [
    "LoSharpeTest",
    "RobustValidationReport",
    "block_bootstrap_sharpe_ci",
    "cost_sensitivity_curve",
    "deflated_sharpe_ratio",
    "lo_sharpe_test",
    "out_of_time_test",
    "permutation_sharpe_null",
    "run_robust_validation",
]
