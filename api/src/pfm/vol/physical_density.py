"""Physical (P-measure) terminal density of an equity index.

This is the *physical leg* of an empirical pricing kernel: the real-world,
objective-measure terminal density ``f_P(S_T)``
of an index at a short horizon, to be compared against the risk-neutral
density ``f_Q(S_T)`` recovered from option / prediction-market prices. The
ratio ``f_Q / f_P`` (suitably discounted) is the empirical pricing kernel.

Method
------
1. Fit a GARCH(1,1) to the index log-returns to obtain a
   one-step-ahead conditional volatility forecast ``σ̂_{T+1}`` (daily, in
   log-return units). We pass *log-prices* to :func:`pfm.garch.fit_garch_11`
   because that function internally first-differences its input, so
   ``Δ log P = log-return``.
2. Scale to the horizon: variance adds across days, so
   ``σ_T = σ_1d · √(horizon_days)`` (``horizon_days`` may be fractional).
3. Build a lognormal terminal density under the *physical* drift. Under
   the P-measure the log-return over ``t_years`` has mean
   ``m = (μ_P − ½ σ_ann²) · t_years`` where ``μ_P`` is the physical
   expected annual return (equity-risk-premium + risk-free), so
   ``log(S_T / S_0) ~ N(m, σ_T²)`` and ``S_T`` is lognormal.

The physical density differs from the risk-neutral one precisely through
the drift: ``f_Q`` uses the risk-free rate ``r``, ``f_P`` uses ``μ_P``. The
volatility is GARCH-forecast and shared in spirit (a POC simplification).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm

from pfm.garch import fit_garch_11

logger = logging.getLogger(__name__)

# NumPy 2.0 renamed ``trapz`` -> ``trapezoid``; keep both runtimes working.
_trapz = getattr(np, "trapezoid", None) or np.trapz

#: Trading days per year — the standard annualisation convention.
TRADING_DAYS_PER_YEAR: float = 252.0

#: Default minimum bars required to attempt a GARCH fit.
_MIN_GARCH_OBS: int = 60

#: yfinance ticker map for supported indices.
_TICKER_MAP: dict[str, str] = {
    "SPX": "^GSPC",
    "SP500": "^GSPC",
    "GSPC": "^GSPC",
    "NDX": "^NDX",
    "NASDAQ100": "^NDX",
}


@dataclass(frozen=True)
class PhysicalDensityResult:
    """Physical (P-measure) terminal density of an index.

    Attributes:
        spot: Reference spot level ``S_0``.
        t_years: Calendar horizon in years (for the drift term).
        horizon_days: Horizon in (fractional) days (for vol scaling).
        sigma_1d: One-day-ahead GARCH σ forecast (daily log-return units).
        sigma_ann: Annualised volatility ``σ_1d · √252``.
        sigma_T: Horizon volatility ``σ_1d · √horizon_days``.
        annual_drift: Physical expected annual return ``μ_P`` used.
        risk_free: Risk-free rate carried for reference / kernel use.
        grid: Strictly increasing strike grid (price levels of ``S_T``).
        pdf: Probability density on ``grid`` (integrates to ~1).
        cdf: Cumulative distribution on ``grid`` (monotone, in ``[0, 1]``).
        garch_persistence: Estimated ``α + β`` of the fitted GARCH(1,1).
        garch_converged: Whether the GARCH optimiser converged.
        n_obs: Number of price observations used.
        warnings: Non-fatal issues encountered (e.g. GARCH fallback).
        asset: Optional asset symbol.
        label: Optional human label.
    """

    spot: float
    t_years: float
    horizon_days: float
    sigma_1d: float
    sigma_ann: float
    sigma_T: float  # noqa: N815 — financial notation σ_T (horizon vol)
    annual_drift: float
    risk_free: float
    grid: np.ndarray
    pdf: np.ndarray
    cdf: np.ndarray
    garch_persistence: float
    garch_converged: bool
    n_obs: int
    warnings: list[str] = field(default_factory=list)
    asset: str | None = None
    label: str | None = None


def _sample_sigma_fallback(log_prices: pd.Series) -> float:
    """Rolling/full-sample stdev of log-returns as a σ_1d fallback.

    Args:
        log_prices: Series of log-prices.

    Returns:
        Sample standard deviation of the most-recent log-returns (daily).
    """
    rets = log_prices.dropna().diff().dropna()
    if len(rets) < 2:
        raise ValueError("physical_density: need >=2 returns for sample-σ fallback")
    window = rets.tail(60) if len(rets) >= 60 else rets
    sigma = float(window.std(ddof=1))
    if not np.isfinite(sigma) or sigma <= 0.0:
        sigma = float(rets.std(ddof=1))
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("physical_density: log-returns have zero variance")
    return sigma


def _build_grid(
    spot: float,
    m: float,
    sigma_T: float,
    *,
    grid_size: int,
) -> np.ndarray:
    """Build a strike grid of ``±6 σ_T`` around the lognormal centre.

    Args:
        spot: Spot ``S_0``.
        m: Mean of ``log(S_T / S_0)``.
        sigma_T: Horizon log-return volatility.
        grid_size: Number of grid points.

    Returns:
        Strictly increasing array of positive price levels.
    """
    lo_log = np.log(spot) + m - 6.0 * sigma_T
    hi_log = np.log(spot) + m + 6.0 * sigma_T
    log_grid = np.linspace(lo_log, hi_log, int(grid_size))
    return np.exp(log_grid)


def physical_density_from_returns(
    log_prices: pd.Series,
    spot: float,
    t_years: float,
    *,
    horizon_days: float,
    grid: np.ndarray | None = None,
    grid_size: int = 400,
    annual_drift: float = 0.06,
    risk_free: float = 0.045,
    asset: str | None = None,
    label: str | None = None,
) -> PhysicalDensityResult:
    """Estimate the physical terminal density from a log-price history.

    Pure math, no network. Fits GARCH(1,1) to recover ``σ_1d``, scales it to
    the horizon, and evaluates a lognormal physical density of ``S_T`` on a
    strike grid.

    Args:
        log_prices: Series of ``log(Close)`` (the GARCH input; first-
            differenced internally to log-returns).
        spot: Reference spot level ``S_0``.
        t_years: Calendar horizon in years (drives the drift term).
        horizon_days: Horizon in (possibly fractional) days (drives the
            volatility scaling ``σ_T = σ_1d · √horizon_days``).
        grid: Optional explicit strictly-increasing strike grid. When
            ``None``, a ``±6 σ_T`` lognormal-centred grid is built.
        grid_size: Number of grid points when ``grid`` is ``None``.
        annual_drift: Physical expected annual return ``μ_P``. Default 0.06.
        risk_free: Risk-free rate, carried for pricing-kernel use.
        asset: Optional asset symbol for the result.
        label: Optional human-readable label for the result.

    Returns:
        A :class:`PhysicalDensityResult`.

    Raises:
        ValueError: If inputs are degenerate (no valid σ recoverable, non-
            positive ``spot``/``horizon_days``, or empty grid).
    """
    warnings: list[str] = []

    if spot <= 0.0 or not np.isfinite(spot):
        raise ValueError(f"physical_density: spot must be positive, got {spot}")
    if horizon_days <= 0.0 or not np.isfinite(horizon_days):
        raise ValueError(f"physical_density: horizon_days must be positive, got {horizon_days}")
    if t_years < 0.0 or not np.isfinite(t_years):
        raise ValueError(f"physical_density: t_years must be non-negative, got {t_years}")

    s = log_prices.dropna()
    n_obs = int(len(s))
    if not np.all(np.isfinite(s.to_numpy(dtype=float))):
        raise ValueError("physical_density: log_prices contain non-finite values")

    sigma_1d: float
    persistence = float("nan")
    converged = False

    if n_obs < _MIN_GARCH_OBS:
        warnings.append(f"only {n_obs} obs (<{_MIN_GARCH_OBS}); using sample-σ instead of GARCH")
        sigma_1d = _sample_sigma_fallback(s)
    else:
        try:
            gres = fit_garch_11(s)
            sigma_1d = float(gres.last_sigma)
            persistence = float(gres.persistence)
            converged = bool(gres.converged)
            if not np.isfinite(sigma_1d) or sigma_1d <= 0.0:
                raise ValueError(f"GARCH produced non-positive σ ({sigma_1d})")
            if not converged:
                warnings.append("GARCH did not converge; σ forecast may be unreliable")
        except Exception as exc:  # fall back, never raise on convergence
            logger.warning("physical_density: GARCH failed (%s); using sample-σ", exc)
            warnings.append(f"GARCH fit failed ({exc}); fell back to sample-σ")
            sigma_1d = _sample_sigma_fallback(s)
            converged = False

    sigma_ann = sigma_1d * np.sqrt(TRADING_DAYS_PER_YEAR)
    sigma_T = sigma_1d * np.sqrt(horizon_days)

    # Physical lognormal drift over the calendar horizon.
    m = (annual_drift - 0.5 * sigma_ann**2) * t_years

    if grid is None:
        grid_arr = _build_grid(spot, m, sigma_T, grid_size=grid_size)
    else:
        grid_arr = np.asarray(grid, dtype=float)
        if grid_arr.ndim != 1 or grid_arr.size < 2:
            raise ValueError("physical_density: grid must be 1-D with >=2 points")
        if np.any(grid_arr <= 0.0):
            raise ValueError("physical_density: grid must contain only positive levels")
        if not np.all(np.diff(grid_arr) > 0.0):
            raise ValueError("physical_density: grid must be strictly increasing")

    # Lognormal: log(S_T/spot) ~ N(m, sigma_T^2).
    if sigma_T <= 0.0:
        raise ValueError(f"physical_density: non-positive sigma_T ({sigma_T})")
    log_moneyness = np.log(grid_arr / spot)
    z = (log_moneyness - m) / sigma_T
    # pdf of S_T: f(S) = phi(z) / (S * sigma_T)
    pdf = norm.pdf(z) / (grid_arr * sigma_T)
    cdf = norm.cdf(z)

    # Normalise pdf to integrate to 1 on the grid (clips ±6σ tail mass loss).
    area = float(_trapz(pdf, grid_arr))
    if np.isfinite(area) and area > 0.0:
        pdf = pdf / area
    else:
        warnings.append("pdf normalisation area non-positive; left unnormalised")

    # Clamp cdf strictly into [0, 1] and keep monotone.
    cdf = np.clip(cdf, 0.0, 1.0)
    cdf = np.maximum.accumulate(cdf)

    return PhysicalDensityResult(
        spot=float(spot),
        t_years=float(t_years),
        horizon_days=float(horizon_days),
        sigma_1d=float(sigma_1d),
        sigma_ann=float(sigma_ann),
        sigma_T=float(sigma_T),
        annual_drift=float(annual_drift),
        risk_free=float(risk_free),
        grid=grid_arr,
        pdf=pdf,
        cdf=cdf,
        garch_persistence=persistence,
        garch_converged=converged,
        n_obs=n_obs,
        warnings=warnings,
        asset=asset,
        label=label,
    )


def fetch_index_history(asset: str = "SPX", lookback_days: int = 400) -> pd.Series:
    """Fetch a daily log-Close history for an index via yfinance.

    Network call — isolated here so the pure math stays testable. yfinance is
    imported lazily so importing this module never triggers a network import.

    Args:
        asset: Index symbol (``"SPX"`` or ``"NDX"``; aliases accepted).
        lookback_days: Calendar days of history to request.

    Returns:
        A ``pd.Series`` of ``log(Close)`` indexed by date, dropna'd.

    Raises:
        ValueError: If ``asset`` is unknown or no data is returned.
    """
    import yfinance as yf  # lazy: keeps module import network-free

    key = asset.strip().upper().replace("^", "")
    ticker = _TICKER_MAP.get(key)
    if ticker is None:
        raise ValueError(
            f"fetch_index_history: unknown asset {asset!r}; supported: {sorted(set(_TICKER_MAP))}"
        )

    period = f"{int(max(lookback_days, 5))}d"
    raw = yf.Ticker(ticker).history(period=period, interval="1d")
    if raw is None or len(raw) == 0 or "Close" not in raw.columns:
        raise ValueError(f"fetch_index_history: no data returned for {ticker}")

    close = raw["Close"].dropna()
    close = close[close > 0.0]
    if len(close) == 0:
        raise ValueError(f"fetch_index_history: all-zero/empty closes for {ticker}")

    log_close = np.log(close.astype(float))
    log_close.name = f"log_close_{key}"
    # Normalise the index to plain dates (UTC calendar) for downstream joins.
    try:
        log_close.index = pd.DatetimeIndex(log_close.index).tz_localize(None)
    except (TypeError, AttributeError):
        pass
    return log_close


def estimate_physical_density(
    asset: str = "SPX",
    *,
    spot: float,
    t_years: float,
    horizon_days: float,
    lookback_days: int = 400,
    grid: np.ndarray | None = None,
    grid_size: int = 400,
    annual_drift: float = 0.06,
    risk_free: float = 0.045,
    label: str | None = None,
) -> PhysicalDensityResult:
    """Fetch index history then estimate its physical terminal density.

    Orchestrator: thin wrapper combining :func:`fetch_index_history` and
    :func:`physical_density_from_returns`.

    Args:
        asset: Index symbol (``"SPX"`` / ``"NDX"`` and aliases).
        spot: Reference spot level ``S_0``.
        t_years: Calendar horizon in years.
        horizon_days: Horizon in (fractional) days for vol scaling.
        lookback_days: History window to fetch.
        grid: Optional explicit strike grid.
        grid_size: Grid points when ``grid`` is ``None``.
        annual_drift: Physical expected annual return ``μ_P``.
        risk_free: Risk-free rate.
        label: Optional human label.

    Returns:
        A :class:`PhysicalDensityResult`.
    """
    log_prices = fetch_index_history(asset, lookback_days=lookback_days)
    return physical_density_from_returns(
        log_prices,
        spot=spot,
        t_years=t_years,
        horizon_days=horizon_days,
        grid=grid,
        grid_size=grid_size,
        annual_drift=annual_drift,
        risk_free=risk_free,
        asset=asset.strip().upper().replace("^", ""),
        label=label,
    )


__all__ = [
    "PhysicalDensityResult",
    "estimate_physical_density",
    "fetch_index_history",
    "physical_density_from_returns",
]
