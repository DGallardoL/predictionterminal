"""Cointegration tests for prediction-market probability pairs.

Two events whose prices share a stochastic trend can be tradeable as a
mean-reverting pair. We use the Engle-Granger 2-step (Engle & Granger 1987)
which is appropriate for the bivariate case and which all the downstream
pairs-trading code consumes:

    Step 1: P_A_t = α + β · P_B_t + ε_t          (OLS hedge ratio)
    Step 2: ADF on {ε_t}                          (residuals must be I(0))

If we reject the ADF null at p < α (default 0.05), the pair is cointegrated
and ε is the *spread* whose dynamics drive the trade. We additionally fit
an AR(1) on the spread to derive a half-life:

    ε_t = ρ · ε_{t-1} + η_t      → half-life = −ln(2) / ln(ρ)        for ρ ∈ (0, 1)

A pair with a half-life over a few months is uninteresting for daily-bar
pairs trading (capital tied up too long for the realised reversion).

For sets of >2 series we expose a thin wrapper around the Johansen test
(Johansen 1991) — useful for "is *any* basket cointegrating combination
of these N events?". The Johansen test reports the trace and max-eigen
statistics; we surface both.

References:
    Engle, R. F. & Granger, C. W. J. (1987). "Co-integration and Error
        Correction: Representation, Estimation, and Testing." Econometrica.
    Johansen, S. (1991). "Estimation and Hypothesis Testing of
        Cointegration Vectors in Gaussian Vector Autoregressive Models."
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log

import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller


@dataclass(frozen=True)
class CointegrationResult:
    """Output of :func:`engle_granger`.

    Attributes:
        n_obs: jointly-observed sample size.
        beta_hedge: OLS slope β. Long 1 unit of A, short β units of B
            yields the spread time series.
        intercept: OLS intercept α from step 1.
        adf_stat: ADF test statistic on the residuals.
        adf_pvalue: ADF p-value. Reject H0 (unit root) ⇒ cointegrated.
        adf_used_lag: lag selected by AIC.
        spread: residual series ε_t.
        half_life_days: AR(1)-derived mean-reversion half-life. ``None``
            if AR(1) coefficient is non-stationary or non-positive.
        rho: AR(1) autoregressive coefficient on the spread.
        cointegrated: convenience flag (``adf_pvalue < significance``).
        verdict: one of ``"cointegrated"`` / ``"not_cointegrated"`` /
            ``"insufficient-data"``.
    """

    n_obs: int
    beta_hedge: float
    intercept: float
    adf_stat: float
    adf_pvalue: float
    adf_used_lag: int
    spread: pd.Series
    half_life_days: float | None
    rho: float | None
    cointegrated: bool
    verdict: str


def _align(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
    df = pd.concat({"a": a, "b": b}, axis=1).dropna()
    return df["a"], df["b"]


def _half_life_ar1(spread: pd.Series) -> tuple[float | None, float | None]:
    """Fit AR(1) on the spread and return ``(half_life, rho)``.

    Returns ``(None, rho)`` when ``rho`` is non-positive or ≥ 1 (the AR(1)
    is non-stationary or anti-persistent — half-life is undefined).
    """
    s = spread.dropna()
    if len(s) < 5:
        return None, None
    y = s.iloc[1:].to_numpy()
    x = s.iloc[:-1].to_numpy()
    X = sm.add_constant(x)
    res = sm.OLS(y, X).fit()
    rho = float(res.params[1])
    if rho <= 0.0 or rho >= 1.0:
        return None, rho
    return -log(2.0) / log(rho), rho


def trim_leading_flat(
    series: pd.Series,
    *,
    window: int = 30,
    min_std: float = 0.005,
) -> pd.Series:
    """Drop the leading "off" period where the market hasn't started moving.

    Many prediction markets sit near 0 or 1 for weeks/months before the event
    becomes news-relevant. Cointegration tests on such series get dominated
    by the boring period and produce false positives. This drops bars
    before the rolling-30-day std first exceeds ``min_std``.

    Args:
        series: probability series (any [0, 1] series, really).
        window: rolling-window length for the std measure.
        min_std: minimum std required for the series to be "active".

    Returns:
        Trimmed series; if no bar has rolling std ≥ min_std, returns the
        original series (don't truncate to nothing).
    """
    s = series.dropna()
    if len(s) < window + 5:
        return s
    rolling_std = s.rolling(window=window, min_periods=window // 2).std(ddof=1)
    active = rolling_std >= min_std
    if not active.any():
        return s
    first_active = active.idxmax()  # first True
    return s.loc[first_active:]


def engle_granger(
    p_a: pd.Series,
    p_b: pd.Series,
    *,
    significance: float = 0.05,
    adf_max_lag: int | None = None,
    transform: str = "raw",
    trim_leading: bool = False,
) -> CointegrationResult:
    """Run the Engle-Granger 2-step cointegration test.

    Args:
        p_a: probability series for event A (target leg of the spread).
        p_b: probability series for event B (hedging leg).
        significance: ADF p-value threshold to declare cointegration.
        adf_max_lag: optional override for ``adfuller``'s lag selection.
        transform: ``"raw"`` (default), ``"logit"`` (log-odds; clips to
            [0.01, 0.99] to avoid ±∞), or ``"diff"`` (first differences).
            Logit transforms can amplify signal in [0,1]-bounded series.
        trim_leading: if ``True``, drop the leading "flat" period of each
            series via :func:`trim_leading_flat` before alignment. Removes
            false-positive cointegration from co-flat early bars.

    Returns:
        :class:`CointegrationResult`.
    """
    if trim_leading:
        p_a = trim_leading_flat(p_a)
        p_b = trim_leading_flat(p_b)
    if transform == "logit":
        from numpy import log

        clipped_a = p_a.clip(lower=0.01, upper=0.99)
        clipped_b = p_b.clip(lower=0.01, upper=0.99)
        p_a = log(clipped_a / (1.0 - clipped_a))
        p_b = log(clipped_b / (1.0 - clipped_b))
    elif transform == "diff":
        p_a = p_a.diff().dropna()
        p_b = p_b.diff().dropna()
    elif transform != "raw":
        raise ValueError(f"unknown transform {transform!r}")
    a, b = _align(p_a, p_b)
    if len(a) < 30:
        return CointegrationResult(
            n_obs=len(a),
            beta_hedge=float("nan"),
            intercept=float("nan"),
            adf_stat=float("nan"),
            adf_pvalue=float("nan"),
            adf_used_lag=0,
            spread=pd.Series(dtype=float),
            half_life_days=None,
            rho=None,
            cointegrated=False,
            verdict="insufficient-data",
        )

    X = sm.add_constant(b.values)
    ols = sm.OLS(a.values, X).fit()
    # `add_constant(has_constant='skip')` is the statsmodels default — if `b`
    # is degenerate (all-equal values) the constant column collides with `b`
    # itself, statsmodels drops it, and `ols.params` ends up length-1. Indexing
    # [1] then raises and 500s the whole scan. Treat that as "not enough
    # variation to fit a hedge ratio" and bail with the same insufficient-data
    # shape used for short series above.
    if len(ols.params) < 2:
        return CointegrationResult(
            n_obs=len(a),
            beta_hedge=float("nan"),
            intercept=float("nan"),
            adf_stat=float("nan"),
            adf_pvalue=float("nan"),
            adf_used_lag=0,
            spread=pd.Series(dtype=float),
            half_life_days=None,
            rho=None,
            cointegrated=False,
            verdict="insufficient-variation",
        )
    intercept = float(ols.params[0])
    beta = float(ols.params[1])
    spread = pd.Series(ols.resid, index=a.index, name="spread")

    adf_kwargs: dict[str, object] = (
        {"autolag": "AIC"} if adf_max_lag is None else {"maxlag": adf_max_lag, "autolag": None}
    )
    adf_stat, adf_p, adf_lag, _, _, _ = adfuller(spread.values, **adf_kwargs)
    cointeg = bool(adf_p < significance)
    half_life, rho = _half_life_ar1(spread)

    verdict = "cointegrated" if cointeg else "not_cointegrated"

    return CointegrationResult(
        n_obs=len(a),
        beta_hedge=beta,
        intercept=intercept,
        adf_stat=float(adf_stat),
        adf_pvalue=float(adf_p),
        adf_used_lag=int(adf_lag),
        spread=spread,
        half_life_days=half_life,
        rho=rho,
        cointegrated=cointeg,
        verdict=verdict,
    )


@dataclass(frozen=True)
class JohansenResult:
    """Output of :func:`johansen_test`.

    Attributes:
        n_obs: sample size.
        rank_trace: estimated cointegration rank by trace test (number of
            cointegrating vectors at the chosen significance).
        rank_eigen: rank by max-eigenvalue test.
        trace_stats: list of trace statistics for r=0, 1, ..., k-1.
        trace_crit_95: corresponding 95% critical values.
        eigen_stats: list of max-eigen statistics.
        eigen_crit_95: corresponding 95% critical values.
        eigvecs: cointegrating vectors (columns) in the *first* β.
    """

    n_obs: int
    rank_trace: int
    rank_eigen: int
    trace_stats: list[float]
    trace_crit_95: list[float]
    eigen_stats: list[float]
    eigen_crit_95: list[float]
    eigvecs: list[list[float]]


def johansen_test(
    df: pd.DataFrame,
    *,
    det_order: int = 0,
    k_ar_diff: int = 1,
) -> JohansenResult:
    """Johansen cointegration test for ≥2 series.

    Args:
        df: DataFrame whose columns are the probability series.
        det_order: deterministic-trend assumption (-1=no trend, 0=const,
            1=linear). Default 0 (const) matches typical pairs setup.
        k_ar_diff: lags in the VAR-in-differences. Default 1.

    Returns:
        :class:`JohansenResult`.

    Raises:
        ValueError: if fewer than 2 columns or fewer than 30 rows.
    """
    from statsmodels.tsa.vector_ar.vecm import coint_johansen

    if df.shape[1] < 2:
        raise ValueError("johansen_test requires ≥2 columns")
    aligned = df.dropna()
    if len(aligned) < 30:
        raise ValueError(f"johansen_test needs ≥30 rows after dropna, got {len(aligned)}")

    res = coint_johansen(aligned.values, det_order=det_order, k_ar_diff=k_ar_diff)
    # 95% column index = 1 (statsmodels returns critical values at 90/95/99).
    trace_stats = [float(x) for x in res.lr1]
    trace_crit_95 = [float(x) for x in res.cvt[:, 1]]
    eigen_stats = [float(x) for x in res.lr2]
    eigen_crit_95 = [float(x) for x in res.cvm[:, 1]]
    rank_trace = int(sum(t > c for t, c in zip(trace_stats, trace_crit_95, strict=True)))
    rank_eigen = int(sum(t > c for t, c in zip(eigen_stats, eigen_crit_95, strict=True)))
    eigvecs = [[float(x) for x in row] for row in res.evec]

    return JohansenResult(
        n_obs=len(aligned),
        rank_trace=rank_trace,
        rank_eigen=rank_eigen,
        trace_stats=trace_stats,
        trace_crit_95=trace_crit_95,
        eigen_stats=eigen_stats,
        eigen_crit_95=eigen_crit_95,
        eigvecs=eigvecs,
    )


def spread_zscore(
    spread: pd.Series,
    *,
    window: int = 20,
    min_periods: int | None = None,
) -> pd.Series:
    """Rolling z-score of a spread series.

    Standard pairs-trading entry/exit signal generator. Default 20-bar
    window matches the typical "month-long memory" reversion horizon on
    daily probability series.
    """
    if min_periods is None:
        min_periods = max(5, window // 2)
    mu = spread.rolling(window=window, min_periods=min_periods).mean()
    sd = spread.rolling(window=window, min_periods=min_periods).std(ddof=1)
    z = (spread - mu) / sd
    return z.rename("zscore")


__all__ = [
    "CointegrationResult",
    "JohansenResult",
    "engle_granger",
    "johansen_test",
    "spread_zscore",
]
