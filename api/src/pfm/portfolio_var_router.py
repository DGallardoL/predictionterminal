"""``GET /portfolio/{handle}/var`` — Value at Risk for an imported portfolio.

Three VaR methodologies are computed side-by-side over the daily log
returns of the portfolio's constituents:

1.  **Parametric (variance-covariance)** — assumes returns are jointly
    Gaussian. Portfolio mean μ_p = wᵀμ, vol σ_p = √(wᵀΣw). VaR =
    portfolio_value × (z_α × σ_p × √h − μ_p × h) where z_α is the
    standard-normal upper-tail quantile at the chosen confidence and
    *h* is the horizon in trading days.
2.  **Historical simulation** — empirical α-quantile of the realised
    portfolio return series scaled by √h (square-root-of-time).
3.  **Monte Carlo** — `mc_paths` Cholesky-correlated Gaussian draws
    from (μ, Σ) compounded over *h* days; α-quantile of the simulated
    portfolio return.

Expected shortfall (conditional VaR) is reported at the **fixed 95%**
confidence level — the average loss in the worst (1 − 0.95) tail of the
historical series — regardless of the user-supplied `confidence`, so
the field is comparable across queries.

All VaR figures are reported as **positive dollar amounts** representing
the *loss* threshold (i.e. with probability `1 − confidence` the
portfolio loses at least this much over `horizon_days`).

Data source
-----------
Daily log returns come from :func:`pfm.sources.equity.get_log_returns`
which fans out yfinance → Tiingo → Stooq. Tests inject a stub via
``request.app.state.var_return_provider`` to avoid real network I/O —
the override signature is ``provider(ticker, start, end) -> pd.Series``.

Portfolio handle
----------------
The handle must have been produced by
:mod:`pfm.portfolio_import_router` (i.e. live in
``app.state.portfolios``). 404 is returned when the handle is unknown.

Weights
-------
Position weights are computed as ``shares × latest_price /
total_market_value``. When the latest-price lookup fails for a ticker
the row is dropped and a warning is emitted; if **all** rows fail the
endpoint returns 502.

Routing
-------
``main.py`` is owned by another wave-13 session, so this router is
left standalone. The next agent that owns ``main.py:routes`` should
mount it via::

    from pfm.portfolio_var_router import router as _portfolio_var_router
    app.include_router(_portfolio_var_router)
"""

from __future__ import annotations

import logging
import math
from collections import OrderedDict
from collections.abc import Callable
from typing import Any, Protocol

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Path, Query, Request, status
from pydantic import BaseModel, Field

from pfm.portfolio_import_router import Portfolio

logger = logging.getLogger(__name__)

router = APIRouter(tags=["portfolio"])


# ---------------------------------------------------------------------------
# constants / defaults
# ---------------------------------------------------------------------------

DEFAULT_LOOKBACK_DAYS = 252
"""One trading year of history powering μ, Σ, and the empirical quantile."""

DEFAULT_MC_PATHS = 10_000
"""Default Monte Carlo path count. Adjustable via ``mc_paths`` query param."""

MIN_OBSERVATIONS = 30
"""Minimum joint-aligned return observations required to fit a VaR model."""

ES_CONFIDENCE = 0.95
"""Fixed confidence level for Expected Shortfall (independent of `confidence`)."""


# ---------------------------------------------------------------------------
# pydantic response schema
# ---------------------------------------------------------------------------


class VaRResponse(BaseModel):
    """Result of ``GET /portfolio/{handle}/var``.

    All VaR / ES figures are **positive USD amounts** representing the
    *loss* threshold at the requested confidence over `horizon_days`.
    """

    handle: str = Field(..., examples=["pf_2026-05-16_a3f5d1"])
    confidence: float = Field(..., gt=0.5, lt=1.0, examples=[0.95])
    horizon_days: int = Field(..., ge=1, le=252, examples=[1])
    portfolio_value: float = Field(
        ...,
        ge=0.0,
        description="Sum of ``shares × latest_price`` across all tickers.",
    )
    parametric_var: float = Field(
        ...,
        ge=0.0,
        description="Variance-covariance Gaussian VaR in USD.",
    )
    historical_var: float = Field(
        ...,
        ge=0.0,
        description="Empirical α-quantile of portfolio returns in USD.",
    )
    monte_carlo_var: float = Field(
        ...,
        ge=0.0,
        description="MC-simulated VaR in USD.",
    )
    expected_shortfall_95: float = Field(
        ...,
        ge=0.0,
        description=(
            "Average loss in the worst 5% tail of the historical "
            "series — in USD over `horizon_days`."
        ),
    )
    weights: dict[str, float] = Field(
        ...,
        description="Position weights (market-value share, summing to ~1.0).",
    )
    n_observations: int = Field(..., ge=MIN_OBSERVATIONS)
    mc_paths: int = Field(..., ge=100)
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# data-provider protocol (injectable for tests)
# ---------------------------------------------------------------------------


class ReturnProvider(Protocol):
    """Callable that returns a Series of daily log returns.

    The default implementation wraps :func:`pfm.sources.equity.get_log_returns`.
    Tests can install a fake by setting
    ``app.state.var_return_provider = my_func``.
    """

    def __call__(
        self,
        ticker: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.Series: ...


def _default_return_provider(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Production fetch — yfinance/Tiingo/Stooq cascaded."""
    from pfm.sources.equity import get_log_returns

    return get_log_returns(ticker, start, end, return_type="log")


class PriceProvider(Protocol):
    """Returns the latest close price for a ticker (in USD)."""

    def __call__(self, ticker: str) -> float: ...


def _default_price_provider(ticker: str) -> float:
    """Production fetch — yfinance ``Ticker.history`` last close."""
    import yfinance as yf  # local import to keep module import cheap

    tk = yf.Ticker(ticker)
    hist = tk.history(period="5d", auto_adjust=False)
    if hist.empty:
        raise ValueError(f"no recent price for {ticker!r}")
    return float(hist["Close"].iloc[-1])


# ---------------------------------------------------------------------------
# pure-math helpers
# ---------------------------------------------------------------------------


def _inv_norm_cdf(p: float) -> float:
    """Standard-normal inverse CDF.

    Uses :class:`statistics.NormalDist` from the stdlib, which is
    available on every Python 3.8+ and avoids pulling in scipy. The
    underlying implementation is the rational-approximation algorithm
    that's accurate to ~1e-9.
    """
    if not (0.0 < p < 1.0):
        raise ValueError(f"p must be in (0, 1); got {p}")
    from statistics import NormalDist

    return NormalDist().inv_cdf(p)


def parametric_var(
    weights: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
    confidence: float,
    horizon_days: int,
    portfolio_value: float,
) -> float:
    """Variance-covariance Gaussian VaR.

    Args:
        weights: shape ``(N,)`` market-value weights summing to ~1.
        mu: shape ``(N,)`` *daily* mean log-return per asset.
        cov: shape ``(N, N)`` daily covariance matrix of log returns.
        confidence: e.g. ``0.95`` for 95% VaR.
        horizon_days: scaling horizon ``h``.
        portfolio_value: gross long market value in USD.

    Returns:
        VaR as a **positive** USD loss amount. Floored at 0 (a portfolio
        with strongly positive drift can have a negative loss-quantile —
        meaningless as a "VaR", so we clamp).
    """
    sigma_p = float(np.sqrt(max(weights @ cov @ weights, 0.0)))
    mu_p = float(weights @ mu)
    z = _inv_norm_cdf(confidence)
    # loss quantile at confidence: z*σ*√h − μ*h (positive ⇒ loss)
    var_return = z * sigma_p * math.sqrt(horizon_days) - mu_p * horizon_days
    return float(max(var_return, 0.0) * portfolio_value)


def historical_var(
    portfolio_returns: np.ndarray,
    confidence: float,
    horizon_days: int,
    portfolio_value: float,
) -> float:
    """Empirical α-quantile VaR scaled by √h.

    ``portfolio_returns`` is the dot product of the daily-return matrix
    with the weights vector — a 1-D series of historical 1-day P&L %.
    """
    alpha = 1.0 - confidence
    q = float(np.quantile(portfolio_returns, alpha))
    # quantile of a return is typically negative; loss = -q
    loss_return_1d = max(-q, 0.0)
    return float(loss_return_1d * math.sqrt(horizon_days) * portfolio_value)


def monte_carlo_var(
    weights: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
    confidence: float,
    horizon_days: int,
    portfolio_value: float,
    n_paths: int,
    seed: int | None = 7,
) -> float:
    """Monte Carlo VaR via correlated Gaussian draws over ``horizon_days``.

    Compounded log returns: r_h = Σ_t z_t @ chol.T + μ * h
    Portfolio path return: w @ r_h
    VaR = portfolio_value × max(−quantile(portfolio_returns, α), 0)
    """
    rng = np.random.default_rng(seed)
    n = len(weights)
    # Cholesky of cov; fall back to diagonal if not PD.
    try:
        chol = np.linalg.cholesky(cov + np.eye(n) * 1e-12)
    except np.linalg.LinAlgError:
        chol = np.diag(np.sqrt(np.clip(np.diag(cov), 0.0, None)))

    # Draw all (n_paths × horizon × N) at once then sum along time axis.
    z = rng.standard_normal(size=(n_paths, horizon_days, n))
    # daily return per asset = z @ chol.T + mu
    daily_ret = z @ chol.T + mu[None, None, :]
    horizon_ret = daily_ret.sum(axis=1)  # shape (n_paths, N)
    port_ret = horizon_ret @ weights  # shape (n_paths,)
    alpha = 1.0 - confidence
    q = float(np.quantile(port_ret, alpha))
    return float(max(-q, 0.0) * portfolio_value)


def expected_shortfall(
    portfolio_returns: np.ndarray,
    confidence: float,
    horizon_days: int,
    portfolio_value: float,
) -> float:
    """Average loss in the worst ``(1 − confidence)`` tail of the series.

    Sometimes called *Conditional VaR (CVaR)* or *Expected Tail Loss*.
    Reported as a positive USD amount. Scaled by √h.
    """
    alpha = 1.0 - confidence
    threshold = np.quantile(portfolio_returns, alpha)
    tail = portfolio_returns[portfolio_returns <= threshold]
    if tail.size == 0:
        return 0.0
    tail_mean = float(tail.mean())
    loss_1d = max(-tail_mean, 0.0)
    return float(loss_1d * math.sqrt(horizon_days) * portfolio_value)


# ---------------------------------------------------------------------------
# orchestration helpers
# ---------------------------------------------------------------------------


def _build_return_matrix(
    portfolio: Portfolio,
    lookback_days: int,
    return_provider: ReturnProvider,
) -> tuple[pd.DataFrame, list[str]]:
    """Fetch and align daily log returns for each ticker.

    Returns a tuple ``(df, warnings)``. ``df`` has shape
    ``(n_obs, n_tickers)`` with columns in portfolio order. Rows with
    any NaN are dropped (inner join across tickers). Tickers whose
    provider call fails are dropped with a warning; if no tickers
    survive an ``HTTPException(502)`` is raised.
    """
    warnings: list[str] = []
    end = pd.Timestamp.now(tz="UTC").normalize()
    # Buffer for weekends/holidays — request ~1.6× lookback in calendar days.
    start = end - pd.Timedelta(days=int(lookback_days * 1.6) + 30)

    series_map: dict[str, pd.Series] = {}
    for row in portfolio.rows:
        try:
            s = return_provider(row.ticker, start, end)
        except Exception as e:  # pragma: no cover — explicit warning path
            warnings.append(f"price_fetch_failed: {row.ticker}: {type(e).__name__}: {e}")
            continue
        if s is None or s.empty:
            warnings.append(f"price_fetch_empty: {row.ticker}")
            continue
        series_map[row.ticker] = s

    if not series_map:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="all return-series fetches failed; cannot compute VaR",
        )

    df = pd.concat(series_map, axis=1, join="inner").dropna(how="any")
    if df.shape[0] < MIN_OBSERVATIONS:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                f"insufficient aligned return observations: {df.shape[0]} < {MIN_OBSERVATIONS}"
            ),
        )

    # Trim to most recent ``lookback_days`` rows.
    if df.shape[0] > lookback_days:
        df = df.iloc[-lookback_days:].copy()

    return df, warnings


def _compute_weights(
    portfolio: Portfolio,
    surviving_tickers: list[str],
    price_provider: PriceProvider,
) -> tuple[np.ndarray, dict[str, float], float, list[str]]:
    """Market-value weights for the surviving tickers.

    Returns ``(weights_arr, weights_dict, portfolio_value, warnings)``.
    """
    warnings: list[str] = []
    row_by_ticker = {r.ticker: r for r in portfolio.rows}
    market_values: dict[str, float] = {}
    for tk in surviving_tickers:
        row = row_by_ticker[tk]
        try:
            px = float(price_provider(tk))
        except Exception as e:  # pragma: no cover — exercised in tests
            warnings.append(f"price_lookup_failed: {tk}: {type(e).__name__}: {e}")
            continue
        market_values[tk] = px * float(row.shares)

    if not market_values:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="latest-price lookup failed for every ticker",
        )

    pv = float(sum(market_values.values()))
    if pv <= 0.0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"non-positive portfolio value: {pv}",
        )

    # Build aligned weights vector in surviving_tickers order, zero-fill
    # any ticker whose price lookup failed.
    weights_arr = np.array(
        [market_values.get(tk, 0.0) / pv for tk in surviving_tickers],
        dtype=float,
    )
    weights_dict = {tk: float(w) for tk, w in zip(surviving_tickers, weights_arr, strict=True)}
    return weights_arr, weights_dict, pv, warnings


def _get_portfolio_store(request: Request) -> OrderedDict[str, Portfolio]:
    """Lookup the portfolio-import store. Always returns the live OrderedDict
    (never copies) so tests can mutate it on the same app."""
    store = getattr(request.app.state, "portfolios", None)
    if not isinstance(store, OrderedDict):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="no portfolios have been imported yet",
        )
    return store


def _get_providers(
    request: Request,
) -> tuple[ReturnProvider, PriceProvider]:
    """Resolve injected providers from ``app.state`` (tests) or defaults."""
    return_provider: ReturnProvider = getattr(
        request.app.state, "var_return_provider", _default_return_provider
    )
    price_provider: PriceProvider = getattr(
        request.app.state, "var_price_provider", _default_price_provider
    )
    return return_provider, price_provider


# ---------------------------------------------------------------------------
# endpoint
# ---------------------------------------------------------------------------


@router.get(
    "/portfolio/{handle}/var",
    response_model=VaRResponse,
    summary="Compute Value at Risk for an imported portfolio.",
    description=(
        "Returns parametric, historical, and Monte Carlo VaR plus 95% "
        "Expected Shortfall over `horizon_days`. All figures are positive "
        "USD loss amounts. The portfolio must have been previously imported "
        "via `POST /portfolio/import`."
    ),
)
def get_portfolio_var(
    request: Request,
    handle: str = Path(..., examples=["pf_2026-05-16_a3f5d1"]),
    confidence: float = Query(
        0.95,
        gt=0.5,
        lt=1.0,
        description="VaR confidence level — e.g. 0.95 for 95% VaR.",
    ),
    horizon_days: int = Query(
        1,
        ge=1,
        le=252,
        description="Forecast horizon in trading days.",
    ),
    lookback_days: int = Query(
        DEFAULT_LOOKBACK_DAYS,
        ge=MIN_OBSERVATIONS,
        le=2520,
        description="Trailing window of returns used to fit μ, Σ.",
    ),
    mc_paths: int = Query(
        DEFAULT_MC_PATHS,
        ge=100,
        le=200_000,
        description="Number of Monte Carlo paths.",
    ),
    seed: int | None = Query(
        7,
        description="RNG seed for Monte Carlo (None ⇒ non-deterministic).",
    ),
) -> VaRResponse:
    store = _get_portfolio_store(request)
    if handle not in store:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"unknown portfolio handle {handle!r}",
        )
    portfolio = store[handle]

    return_provider, price_provider = _get_providers(request)

    warnings: list[str] = []

    # 1. fetch + align return matrix
    ret_df, fetch_warns = _build_return_matrix(portfolio, lookback_days, return_provider)
    warnings.extend(fetch_warns)

    surviving = list(ret_df.columns)

    # 2. weights from latest prices for the surviving tickers
    weights_arr, weights_dict, pv, weight_warns = _compute_weights(
        portfolio, surviving, price_provider
    )
    warnings.extend(weight_warns)

    # Drop any ticker that didn't survive the price-lookup pass.
    nonzero_mask = weights_arr > 0.0
    if not nonzero_mask.all():
        keep = [tk for tk, m in zip(surviving, nonzero_mask, strict=True) if m]
        ret_df = ret_df[keep]
        weights_arr = weights_arr[nonzero_mask]
        weights_dict = {tk: weights_dict[tk] for tk in keep}
        surviving = keep

    # 3. moments
    returns = ret_df.to_numpy()
    mu = returns.mean(axis=0)
    if returns.shape[1] == 1:
        # 1-D edge case — np.cov returns a 0-D scalar.
        cov = np.array([[float(np.var(returns[:, 0], ddof=1))]])
    else:
        cov = np.cov(returns, rowvar=False, ddof=1)

    # 4. compute the three VaRs + ES
    port_ret_1d = returns @ weights_arr
    pvar = parametric_var(weights_arr, mu, cov, confidence, horizon_days, pv)
    hvar = historical_var(port_ret_1d, confidence, horizon_days, pv)
    mcvar = monte_carlo_var(weights_arr, mu, cov, confidence, horizon_days, pv, mc_paths, seed=seed)
    es95 = expected_shortfall(port_ret_1d, ES_CONFIDENCE, horizon_days, pv)

    return VaRResponse(
        handle=handle,
        confidence=confidence,
        horizon_days=horizon_days,
        portfolio_value=pv,
        parametric_var=pvar,
        historical_var=hvar,
        monte_carlo_var=mcvar,
        expected_shortfall_95=es95,
        weights=weights_dict,
        n_observations=int(ret_df.shape[0]),
        mc_paths=mc_paths,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# public re-exports
# ---------------------------------------------------------------------------


__all__ = [
    "DEFAULT_LOOKBACK_DAYS",
    "DEFAULT_MC_PATHS",
    "ES_CONFIDENCE",
    "MIN_OBSERVATIONS",
    "PriceProvider",
    "ReturnProvider",
    "VaRResponse",
    "expected_shortfall",
    "get_portfolio_var",
    "historical_var",
    "monte_carlo_var",
    "parametric_var",
    "router",
]


# Linter appeasement — Callable / Any kept available for downstream type
# checkers without affecting runtime behaviour.
_ = (Any, Callable)
