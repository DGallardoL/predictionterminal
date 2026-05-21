"""Basket statistical arbitrage on a portfolio of related events.

The classic Avellaneda-Lee (2010) construction extended to prediction-market
probability series:

1.  Stack ``k`` related markets into a matrix ``P`` (T × k). Demean.
2.  PCA decomposes ``P`` into ``r`` factor components + idiosyncratic
    residuals ``ε``.
3.  The residuals — what's *left* after subtracting common factor moves
    — should be mean-reverting. Trade the z-score of each market's
    residual: long when residual is far below its mean, short when above.

The "alpha" is the inefficiency-corrected idiosyncratic component of one
market relative to its peers. This generalises pairs-trading from k=2 to
arbitrary k.

Inputs are probability series, not log-returns, so the construction is
slightly different from equity stat-arb where one differences first. We
operate directly on probabilities (which have bounded support and don't
need a log-transform); the residuals are then standardised on a rolling
window for entry/exit.

Kelly criterion: given an estimated edge μ and variance σ² of a per-bar
PnL, the optimal fraction of capital to deploy is ``f* = μ / σ²`` (full
Kelly) or ``f*/2`` (half-Kelly, more practical given parameter
uncertainty).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


@dataclass(frozen=True)
class BasketPCAResult:
    """Output of :func:`basket_pca_residuals`.

    Attributes:
        factor_ids: order of columns in the input.
        n_obs: aligned sample size.
        n_components_used: number of principal components subtracted.
        explained_variance_ratio: first-``n_components_used`` ratios.
        loadings: (k, n_components_used) matrix of PCA loadings.
        residuals: per-date DataFrame of idiosyncratic residuals (one
            column per factor).
        z_residuals: rolling-window z-scores of each residual series.
        kelly_fraction_per_market: Kelly-optimal capital fraction per
            market under a unit-variance assumption (``μ_residual / σ²``).
            Useful as a *relative* sizing signal, not an absolute claim.
    """

    factor_ids: list[str]
    n_obs: int
    n_components_used: int
    explained_variance_ratio: list[float]
    loadings: list[list[float]]
    residuals: pd.DataFrame
    z_residuals: pd.DataFrame
    kelly_fraction_per_market: dict[str, float]


def basket_pca_residuals(
    prices: pd.DataFrame,
    *,
    n_components: int | None = None,
    explained_variance_target: float = 0.70,
    z_window: int = 20,
) -> BasketPCAResult:
    """Run PCA on a probability matrix and return idiosyncratic residuals.

    Args:
        prices: T × k DataFrame of probability series, columns = factor ids.
        n_components: if set, use exactly this many components. If ``None``,
            we keep the smallest number of components that exceeds
            ``explained_variance_target`` of total variance.
        explained_variance_target: target cumulative variance (used when
            ``n_components`` is ``None``).
        z_window: rolling window for residual z-scoring.

    Returns:
        :class:`BasketPCAResult`.

    Raises:
        ValueError: if fewer than 2 columns or fewer than ``z_window+5`` rows
            after dropna.
    """
    if prices.shape[1] < 2:
        raise ValueError("basket_pca_residuals requires ≥2 columns")
    aligned = prices.dropna()
    if len(aligned) < z_window + 5:
        raise ValueError(f"need ≥ z_window+5 = {z_window + 5} aligned bars, got {len(aligned)}")

    centred = aligned - aligned.mean(axis=0)
    k = centred.shape[1]
    pca_full = PCA(n_components=k).fit(centred.values)
    evr_full = pca_full.explained_variance_ratio_

    if n_components is None:
        cum = np.cumsum(evr_full)
        # Number of components to clear the target (at least 1, at most k-1).
        n_use = int(np.searchsorted(cum, explained_variance_target) + 1)
        n_use = max(1, min(n_use, k - 1))
    else:
        n_use = int(np.clip(n_components, 1, k - 1))

    pca = PCA(n_components=n_use).fit(centred.values)
    factors = pca.transform(centred.values)  # T × n_use
    reconstructed = factors @ pca.components_  # T × k
    resid = centred.values - reconstructed
    resid_df = pd.DataFrame(resid, index=aligned.index, columns=aligned.columns)

    mu = resid_df.rolling(window=z_window, min_periods=max(5, z_window // 2)).mean()
    sd = resid_df.rolling(window=z_window, min_periods=max(5, z_window // 2)).std(ddof=1)
    z = (resid_df - mu) / sd
    z = z.replace([np.inf, -np.inf], np.nan)

    # Kelly fraction per market: stat-arb sizing on the LATEST residual.
    # The previous formula (whole-sample mean / variance) was always ~0
    # because PCA residuals are mean-zero by construction — every market
    # came back as ~1e-13. The actionable Kelly for a mean-reverting
    # residual r_t with equilibrium variance σ² is f_t = -r_t / σ²
    # (short when residual is above mean, long when below). Truncate to a
    # ±5x cap so a thin-variance market doesn't blow up sizing.
    eps = 1e-6
    latest = resid_df.iloc[-1] if len(resid_df) else pd.Series(0.0, index=aligned.columns)
    kelly = (-latest / resid_df.var(axis=0).clip(lower=eps)).clip(-5.0, 5.0)
    kelly_dict = {fid: float(v) for fid, v in kelly.items()}

    return BasketPCAResult(
        factor_ids=list(aligned.columns),
        n_obs=len(aligned),
        n_components_used=n_use,
        explained_variance_ratio=[float(v) for v in evr_full[:n_use]],
        loadings=[[float(x) for x in row] for row in pca.components_.T],
        residuals=resid_df,
        z_residuals=z,
        kelly_fraction_per_market=kelly_dict,
    )


def kelly_fraction(
    sharpe: float,
    *,
    fractional: float = 0.5,
) -> float:
    """Approximate Kelly fraction from a Sharpe ratio.

    For a normally-distributed PnL with mean μ and std σ, the full-Kelly
    fraction is ``f* = μ/σ²``. Per-bar Sharpe is ``μ/σ``, so
    ``f* = sharpe / σ``. We don't have σ standalone here — the practical
    rule used is

        f_recommended = fractional · sharpe / 2

    i.e. half-Kelly (``fractional=0.5``) at Sharpe = 1 sizes 25% of capital.
    Caller can dial ``fractional`` down further for safety.

    This is a heuristic — the strict Kelly formula needs ``μ`` and ``σ``
    separately. For pairs-trading PnL it's fine because we don't have a
    well-calibrated per-bar μ either.
    """
    return float(fractional * sharpe / 2.0)


__all__ = ["BasketPCAResult", "basket_pca_residuals", "kelly_fraction"]
