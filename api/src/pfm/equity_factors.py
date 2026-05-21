"""Equity ↔ prediction-market cointegration factors.

Designed in Alpha Report v14 as a queued-but-unbuilt module: cointegrate
yfinance equity prices against Polymarket "largest market cap by year-end"
markets and other equity-adjacent contracts, e.g.

    NVDA   ↔ nvda_largest_jun
    AAPL   ↔ aapl_largest_jun
    TSLA   ↔ tsla_largest_jun
    AMZN   ↔ amzn_largest_jun
    BP     ↔ bp_acquired
    BTC-USD ↔ btc_ath_jun

Methodology — *logit-log* spread
--------------------------------
Equity prices are strictly positive and unbounded above; prediction-market
probabilities are bounded in (0, 1). To put them on the same scale we
transform:

    x_t  = log(P_equity_t)                   (real line, I(1) ≈ random walk)
    y_t  = logit(prob_t) = log(p / (1 - p))  (real line, finite when clipped)

The Engle-Granger 2-step then regresses ``y_t = α + β · x_t + ε_t`` and
ADF-tests the residual ε_t. If ε_t is stationary we have a cointegrated
*belief-versus-fundamental* spread: the prediction-market participants'
implied belief about a milestone moves in step with the underlying equity
price plus a stationary disagreement term that mean-reverts.

We additionally call :func:`pfm.pairs.pairs_backtest` on the spread to
extract an OOS Sharpe — the trade is worth pursuing only if the spread is
both statistically cointegrated *and* economically tradable.

This module is the dependency-glue: it only knows about yfinance + the
existing ``pfm.cointegration`` / ``pfm.pairs`` machinery. The endpoint
wiring lives in :mod:`pfm.main`; live Polymarket fetches live in
:mod:`pfm.sources.polymarket`.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import yfinance as yf

from pfm.cointegration import engle_granger
from pfm.pairs import pairs_backtest

logger = logging.getLogger(__name__)

# Default clip applied to probabilities before the logit. Matches the
# project-wide default (see CLAUDE.md "clipping epsilon").
DEFAULT_CLIP_EPS: float = 0.01


class EquityFactorError(RuntimeError):
    """Raised on equity-factor data or fit error."""


def fetch_equity_history(
    ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    auto_adjust: bool = True,
) -> pd.Series:
    """Fetch a daily adjusted-close equity series via yfinance.

    Note: this is the *price level* fetcher used by the cointegration
    layer (which takes log inside). For *returns* see
    :func:`pfm.sources.equity.get_log_returns`.

    Args:
        ticker: yfinance-compatible symbol (e.g. ``"NVDA"``,
            ``"BTC-USD"``).
        start: Inclusive lower bound (UTC ``pd.Timestamp``).
        end: Inclusive upper bound — yfinance treats ``end`` as
            exclusive, so we add one day before the call.
        auto_adjust: pass-through to ``yf.download``; ``True`` returns
            split/dividend-adjusted closes.

    Returns:
        ``pd.Series`` of adjusted closes indexed by UTC-normalised dates,
        named after the ticker.

    Raises:
        EquityFactorError: if yfinance returns no data or no Close
            column.
    """
    start_d = start.date()
    end_d = end.date()
    end_excl = (end + pd.Timedelta(days=1)).date()

    df = yf.download(
        ticker,
        start=start_d,
        end=end_excl,
        auto_adjust=auto_adjust,
        progress=False,
        threads=False,
    )
    if df is None or df.empty:
        raise EquityFactorError(f"yfinance returned no data for {ticker!r} in [{start_d}, {end_d}]")

    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs(ticker, axis=1, level=-1, drop_level=True)

    if "Close" not in df.columns:
        raise EquityFactorError(f"yfinance response missing Close column for {ticker!r}")

    closes = df["Close"].dropna()
    if len(closes) < 2:
        raise EquityFactorError(f"too few closes for {ticker!r} to compute a price history")

    closes.index = pd.to_datetime(closes.index, utc=True).normalize()
    closes.name = ticker
    return closes


def _logit(p: pd.Series, *, clip_eps: float) -> pd.Series:
    """Element-wise logit with explicit clipping to (eps, 1 - eps).

    Probabilities outside (0, 1) are dropped (NaN) — they're almost
    certainly upstream data errors and silently clipping them masks the
    bug.
    """
    s = p.copy()
    s = s.where((s > 0.0) & (s < 1.0))
    clipped = s.clip(lower=clip_eps, upper=1.0 - clip_eps)
    return np.log(clipped / (1.0 - clipped))


def equity_market_cointegration(
    ticker_series: pd.Series,
    prob_series: pd.Series,
    *,
    clip_eps: float = DEFAULT_CLIP_EPS,
    significance: float = 0.05,
    backtest_window: int = 20,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    stop_z: float = 4.0,
) -> dict:
    """Engle-Granger cointegration of logit(prob) on log(price).

    Combines the existing :func:`pfm.cointegration.engle_granger` step-1
    regression and ADF residual test with a walk-forward backtest from
    :func:`pfm.pairs.pairs_backtest` to attach an out-of-sample Sharpe to
    the verdict.

    Args:
        ticker_series: equity price series indexed by date. Must be
            strictly positive (we take ``log``).
        prob_series: prediction-market probability series in (0, 1).
        clip_eps: probability clip for the logit transform. Default
            ``0.01``.
        significance: ADF p-value threshold for the cointegration
            verdict.
        backtest_window: rolling-window length for the z-score in the
            spread backtest.
        entry_z, exit_z, stop_z: z-score thresholds passed through to
            :func:`pairs_backtest`.

    Returns:
        Dict with keys

        * ``beta``      — OLS slope of logit(prob) on log(price)
        * ``alpha``     — OLS intercept
        * ``adf_stat``  — ADF test statistic on residual spread
        * ``adf_p``     — ADF p-value
        * ``half_life`` — AR(1)-derived half-life in days, or ``None``
        * ``sharpe_oos`` — out-of-sample Sharpe from the spread backtest
            (``None`` if too few bars to backtest)
        * ``verdict``   — ``"cointegrated"`` / ``"not_cointegrated"`` /
            ``"insufficient-data"``
        * ``n_obs``     — overlapping sample size after alignment

    Raises:
        EquityFactorError: if ``ticker_series`` has non-positive values.
    """
    if (ticker_series.dropna() <= 0).any():
        raise EquityFactorError("ticker_series must be strictly positive for log transform")

    log_price = np.log(ticker_series.dropna()).rename("log_price")
    logit_prob = _logit(prob_series, clip_eps=clip_eps).dropna().rename("logit_prob")

    # Note the leg order: logit(prob) is the *target* (A), log(price) is
    # the *hedging* leg (B). β is therefore "logit-points per log-dollar".
    cr = engle_granger(logit_prob, log_price, significance=significance)

    sharpe_oos: float | None = None
    if (
        cr.verdict == "cointegrated"
        and not cr.spread.empty
        and len(cr.spread.dropna()) >= backtest_window + 5
    ):
        try:
            bt = pairs_backtest(
                cr.spread,
                window=backtest_window,
                entry_z=entry_z,
                exit_z=exit_z,
                stop_z=stop_z,
            )
            sharpe_oos = float(bt.sharpe_oos) if np.isfinite(bt.sharpe_oos) else None
        except (ValueError, RuntimeError) as e:
            # Backtest can legitimately fail on degenerate spreads; log
            # and surface a None rather than crash the whole call.
            logger.debug("pairs_backtest failed: %s", e)
            sharpe_oos = None

    return {
        "beta": cr.beta_hedge,
        "alpha": cr.intercept,
        "adf_stat": cr.adf_stat,
        "adf_p": cr.adf_pvalue,
        "half_life": cr.half_life_days,
        "sharpe_oos": sharpe_oos,
        "verdict": cr.verdict,
        "n_obs": cr.n_obs,
    }


__all__ = [
    "DEFAULT_CLIP_EPS",
    "EquityFactorError",
    "equity_market_cointegration",
    "fetch_equity_history",
]
