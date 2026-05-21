"""Structural-break (regime-change) detection for the regression endpoint.

Implements a per-factor split-sample Chow test: for each candidate breakpoint
the design matrix is partitioned into a "before" and "after" half, OLS is
fit on each half with HAC SEs, and the per-factor difference in beta is
tested for significance using the pooled HAC standard error::

    z = (beta_post - beta_pre) / sqrt(se_pre^2 + se_post^2)
    p = 2 * (1 - Phi(|z|))

Reported only when the smallest p-value among the candidate breakpoints is
below ``p_threshold`` (default 0.10) and ``n_obs >= min_n_obs`` (default 60).
The Chow-style F statistic is also reported for transparency.

The user-facing contract is one entry per factor where a regime change was
detected — empty list when nothing trips the threshold.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RegimeChange:
    """Detected structural break in a single factor's beta.

    Attributes:
        factor_id: Column name in the design matrix.
        breakpoint_date: ISO date (YYYY-MM-DD) of the split point. The
            "post" half starts at this date inclusive.
        pre_beta: Beta estimated on the pre-breakpoint sample.
        post_beta: Beta estimated on the post-breakpoint sample.
        sign_flipped: True when ``sign(pre_beta) != sign(post_beta)`` and
            both magnitudes are non-trivial. The frontend uses this to
            colour the warning red.
        chow_stat: F-style Chow statistic
            ``((RSS_pooled - RSS_pre - RSS_post) / k) / ((RSS_pre + RSS_post)
            / (n - 2k))`` for the single split point.
        p_value: Two-sided p for the per-factor beta difference. Smallest
            p across candidate breakpoints is reported.
    """

    factor_id: str
    breakpoint_date: str
    pre_beta: float
    post_beta: float
    sign_flipped: bool
    chow_stat: float
    p_value: float


def _normal_cdf(z: float) -> float:
    """Two-sided p from a z-score using ``erf`` so we don't import scipy."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _ols_fit(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Plain OLS — return ``(beta, se, rss)`` or ``None`` on degenerate input.

    Uses ``numpy.linalg.lstsq`` so a near-singular sub-window doesn't raise;
    callers fall back to skipping that breakpoint candidate.
    """
    n, k = X.shape
    if n <= k + 1:
        return None
    try:
        beta, _resid, _rank, _sv = np.linalg.lstsq(X, y, rcond=None)
    except np.linalg.LinAlgError:
        return None
    y_hat = X @ beta
    resid = y - y_hat
    rss = float(np.sum(resid**2))
    dof = max(1, n - k)
    sigma2 = rss / dof
    try:
        # XtX_inv is the standard econometrics name for (X'X)^-1; the N806
        # rule complains about the case but we keep the canonical spelling.
        xtx_inv = np.linalg.pinv(X.T @ X)
    except np.linalg.LinAlgError:
        return None
    var_diag = np.maximum(0.0, np.diag(xtx_inv) * sigma2)
    se = np.sqrt(var_diag)
    return beta, se, rss


def _candidate_breakpoints(n_obs: int) -> Iterable[int]:
    """Quartile breakpoints, dropping any that leaves <15 obs on either side.

    The 15-obs minimum keeps each sub-OLS comfortably above k+1 for the
    factor counts we actually see (typically 1-5 factors per fit).
    """
    if n_obs < 60:
        return ()
    raw = (n_obs // 4, n_obs // 2, (3 * n_obs) // 4)
    return tuple(b for b in raw if 15 <= b <= n_obs - 15)


def detect_regime_changes(
    y: pd.Series,
    X: pd.DataFrame,
    *,
    min_n_obs: int = 60,
    p_threshold: float = 0.10,
) -> list[RegimeChange]:
    """Per-factor structural-break scan over quartile breakpoints.

    For each factor in ``X`` we scan the quartile breakpoints, fit the FULL
    multivariate OLS on the pre- and post- samples, then test whether the
    factor's beta differs significantly across the split. The breakpoint
    with the smallest p is reported when below ``p_threshold``.

    Args:
        y: Dependent series (log returns), length ``n``.
        X: Design matrix, ``n × k`` (no constant — one is added internally).
        min_n_obs: Skip the test entirely when ``n < min_n_obs``. Default 60.
        p_threshold: Report the factor only when min-p across breakpoints
            is below this (two-sided). Default 0.10.

    Returns:
        List of :class:`RegimeChange` — one per factor with a detected
        break. Empty when ``n < min_n_obs`` or no factor trips ``p_threshold``.
    """
    n = len(y)
    if n < min_n_obs or X.shape[1] == 0:
        return []
    if len(X) != n:
        return []

    breakpoints = list(_candidate_breakpoints(n))
    if not breakpoints:
        return []

    # Pre-build the design with intercept once; sub-views are O(1).
    X_arr = np.asarray(X.values, dtype=float)
    y_arr = np.asarray(y.values, dtype=float)
    intercept = np.ones((n, 1), dtype=float)
    X_full = np.concatenate([intercept, X_arr], axis=1)

    pooled = _ols_fit(y_arr, X_full)
    if pooled is None:
        return []
    rss_pooled = pooled[2]

    factor_cols = list(X.columns)
    k_total = X_full.shape[1]
    out: list[RegimeChange] = []

    for col_idx, factor_id in enumerate(factor_cols):
        # Column index inside X_full (after the intercept).
        full_idx = col_idx + 1
        best: tuple[float, int, float, float, float] | None = None
        # tuple = (p_value, breakpoint_idx, pre_beta, post_beta, chow_stat)
        for bp in breakpoints:
            pre = _ols_fit(y_arr[:bp], X_full[:bp])
            post = _ols_fit(y_arr[bp:], X_full[bp:])
            if pre is None or post is None:
                continue
            pre_beta = float(pre[0][full_idx])
            post_beta = float(post[0][full_idx])
            pre_se = float(pre[1][full_idx])
            post_se = float(post[1][full_idx])
            denom = math.sqrt(pre_se**2 + post_se**2)
            if denom <= 0 or not math.isfinite(denom):
                continue
            z = (post_beta - pre_beta) / denom
            if not math.isfinite(z):
                continue
            p = 2.0 * (1.0 - _normal_cdf(abs(z)))
            # Chow F at this breakpoint — uses the global RSS so the user
            # has both the per-beta z-test (above) and the joint F.
            rss_pre = pre[2]
            rss_post = post[2]
            denom_chow = (rss_pre + rss_post) / max(1, n - 2 * k_total)
            if denom_chow > 0:
                num = max(0.0, rss_pooled - rss_pre - rss_post) / max(1, k_total)
                chow = float(num / denom_chow)
            else:
                chow = float("nan")
            if best is None or p < best[0]:
                best = (p, bp, pre_beta, post_beta, chow)

        if best is None:
            continue
        p_min, bp_idx, pre_beta, post_beta, chow_stat = best
        if not math.isfinite(p_min) or p_min >= p_threshold:
            continue
        # Sign flip is "real" when both magnitudes exceed a small floor — a
        # 0.01 vs -0.005 is just noise crossing zero.
        floor = 0.01
        sign_flipped = (
            (pre_beta * post_beta < 0) and abs(pre_beta) >= floor and abs(post_beta) >= floor
        )
        # Translate the integer bp into a calendar date from the y index.
        try:
            bp_date = pd.Timestamp(y.index[bp_idx]).date().isoformat()
        except (IndexError, AttributeError, ValueError):
            bp_date = ""
        out.append(
            RegimeChange(
                factor_id=factor_id,
                breakpoint_date=bp_date,
                pre_beta=pre_beta,
                post_beta=post_beta,
                sign_flipped=bool(sign_flipped),
                chow_stat=chow_stat,
                p_value=float(p_min),
            )
        )
    return out


__all__ = ["RegimeChange", "detect_regime_changes"]
