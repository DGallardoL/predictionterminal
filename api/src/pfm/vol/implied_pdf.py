"""Implied-PDF engine (Phase 1+2).

Turns a same-maturity family of prediction-market binary contracts into a dense,
smoothed, arbitrage-free implied probability distribution of the underlying.

Three input *shapes* (``LadderFamily.data_shape``) are supported:

``terminal_buckets``
    Kalshi range markets. Each entry's ``prob`` is the probability *mass* over
    ``[floor, cap]``. Masses are clipped non-negative, renormalised (removing
    Kalshi overround), accumulated into a CDF at bucket upper edges, then a
    PCHIP monotone interpolator is fit and differentiated analytically.

``terminal_ladder``
    Kalshi above/below threshold markets. ``above`` gives the survival
    ``S(K)=P(S_T>K)`` (monotone non-increasing); ``below`` gives the CDF
    ``F(K)=P(S_T<K)`` directly. A PCHIP CDF is fit through ``(K, F(K))`` and
    differentiated.

``barrier_touch``
    Polymarket one-touch markets. ``prob`` is ``P(M_T >= K)`` for the running
    maximum ``M_T`` (or running minimum for ``touch_below``). Differencing the
    survival recovers the law of the **running extremum**, model-free — NOT the
    terminal price. With ``barrier_to_terminal=True`` a GBM ``(ν, σ)`` is fit to
    the running-max survival closed form (reflection principle) and a terminal
    overlay is produced.

The PCHIP (`scipy.interpolate.PchipInterpolator`) shape-preserving spline
guarantees a monotone non-decreasing CDF, hence a density ``f = dF/dK >= 0``.
"""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime
from typing import Literal

import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize
from scipy.stats import norm

from pfm.vol.implied_pdf_schemas import (
    DEFAULT_EPSILON,
    DEFAULT_GRID_SIZE,
    GBMFit,
    ImpliedPDFResult,
    LadderEntry,
    LadderFamily,
    MarketPoint,
    Moments,
    Quantiles,
    SmoothMethod,
)
from pfm.vol_surface_pm import (
    _empirical_moments,
    _enforce_monotone,
    _fit_lognormal,
    _safe_float,
)

logger = logging.getLogger(__name__)

# NumPy 2.0 renamed ``trapz`` -> ``trapezoid``; keep both runtimes working.
_trapz = getattr(np, "trapezoid", None) or np.trapz

TailModel = Literal["lognormal", "linear", "none"]

#: Minimum positive time-to-maturity (1 hour, expressed in years).
_MIN_T_YEARS: float = 1.0 / (365.0 * 24.0)

#: Seconds in a (calendar) year — matches the short-dated-PM convention.
_SECONDS_PER_YEAR: float = 365.25 * 86_400.0


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------


def _time_to_maturity_years(
    maturity_utc: datetime, now_utc: datetime | None, warnings: list[str]
) -> float:
    """Return ``T`` in years, floored at a small positive value.

    Args:
        maturity_utc: Maturity timestamp (naive treated as UTC).
        now_utc: Reference "now"; defaults to current UTC time.
        warnings: Mutable list; appends ``"past_maturity"`` when in the past.

    Returns:
        Year-fraction time to maturity, floored at one hour.
    """
    now = now_utc or datetime.now(tz=UTC)
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    mat = maturity_utc
    if mat.tzinfo is None:
        mat = mat.replace(tzinfo=UTC)
    t_seconds = (mat - now).total_seconds()
    t_years = t_seconds / _SECONDS_PER_YEAR
    if t_years <= 0:
        warnings.append("past_maturity: maturity is at or before now_utc")
        return _MIN_T_YEARS
    return max(t_years, _MIN_T_YEARS)


def _renormalise_pdf(grid: np.ndarray, pdf: np.ndarray) -> np.ndarray:
    """Clip tiny negatives to 0 and renormalise so ``∫pdf dK ≈ 1``."""
    pdf = np.where(pdf < 0.0, 0.0, pdf)
    area = float(_trapz(pdf, grid))
    if area > 0 and math.isfinite(area):
        pdf = pdf / area
    return pdf


def _cdf_from_pdf(grid: np.ndarray, pdf: np.ndarray) -> np.ndarray:
    """Cumulative-trapezoid CDF of ``pdf`` over ``grid``, clipped to [0, 1]."""
    cdf = np.zeros_like(grid)
    if grid.shape[0] > 1:
        dx = np.diff(grid)
        incr = 0.5 * (pdf[1:] + pdf[:-1]) * dx
        cdf[1:] = np.cumsum(incr)
    return np.clip(cdf, 0.0, 1.0)


def _moments_from_grid(grid: np.ndarray, pdf: np.ndarray) -> Moments:
    """Compute mean/median/mode/std/skew/excess-kurtosis from a dense grid PDF.

    Median and the mode are read off the dense PDF; central moments are
    integrated numerically with the trapezoidal rule.
    """
    mass = float(_trapz(pdf, grid))
    if mass <= 0 or not math.isfinite(mass):
        return Moments(mean=0.0, median=0.0, mode=0.0, std=0.0, skew=0.0, kurtosis=0.0)
    norm_pdf = pdf / mass

    mean = float(_trapz(grid * norm_pdf, grid))
    var = float(_trapz((grid - mean) ** 2 * norm_pdf, grid))
    std = math.sqrt(max(var, 0.0))

    mode = float(grid[int(np.argmax(norm_pdf))])

    if std > 1e-12:
        skew = float(_trapz(((grid - mean) / std) ** 3 * norm_pdf, grid))
        kurt = float(_trapz(((grid - mean) / std) ** 4 * norm_pdf, grid)) - 3.0
    else:
        skew = 0.0
        kurt = 0.0

    median = _quantile_from_grid(grid, norm_pdf, 0.5)
    return Moments(mean=mean, median=median, mode=mode, std=std, skew=skew, kurtosis=kurt)


def _quantile_from_grid(grid: np.ndarray, pdf: np.ndarray, q: float) -> float:
    """Invert the dense CDF (cumtrapz of ``pdf``) at probability ``q``."""
    cdf = _cdf_from_pdf(grid, pdf)
    # Ensure strictly increasing for np.interp (de-duplicate flats).
    cdf_max = float(cdf[-1]) if cdf[-1] > 0 else 1.0
    cdf = cdf / cdf_max
    return float(np.interp(q, cdf, grid))


def _quantiles_from_grid(grid: np.ndarray, pdf: np.ndarray) -> Quantiles:
    """Return the p5/p25/p50/p75/p95 quantiles from the dense PDF."""
    return Quantiles(
        p5=_quantile_from_grid(grid, pdf, 0.05),
        p25=_quantile_from_grid(grid, pdf, 0.25),
        p50=_quantile_from_grid(grid, pdf, 0.50),
        p75=_quantile_from_grid(grid, pdf, 0.75),
        p95=_quantile_from_grid(grid, pdf, 0.95),
    )


def _lognormal_overlay(grid: np.ndarray, mean: float, std: float) -> list[float] | None:
    """Fit a lognormal to ``(mean, std)`` and evaluate its PDF on ``grid``."""
    if mean <= 0 or std <= 0:
        return None
    mu, sigma = _fit_lognormal(mean, std)
    if sigma <= 0:
        return None
    pos = grid > 0
    out = np.zeros_like(grid)
    g = grid[pos]
    out[pos] = np.exp(-((np.log(g) - mu) ** 2) / (2.0 * sigma * sigma)) / (
        g * sigma * math.sqrt(2.0 * math.pi)
    )
    if not np.all(np.isfinite(out)):
        return None
    return [float(v) for v in out]


def _build_grid(k_lo: float, k_hi: float, grid_size: int) -> np.ndarray:
    """Strictly-increasing dense grid spanning ``[k_lo, k_hi]`` with padding."""
    span = max(k_hi - k_lo, abs(k_hi) * 0.5, 1e-9)
    lo = max(k_lo - 0.25 * span, 0.0) if k_lo >= 0 else k_lo - 0.25 * span
    hi = k_hi + 0.5 * span
    if hi <= lo:
        hi = lo + max(abs(lo), 1.0)
    return np.linspace(lo, hi, grid_size)


def _pchip_cdf_to_density(
    knot_k: np.ndarray, knot_cdf: np.ndarray, grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a PCHIP CDF through knots, return ``(pdf, cdf)`` evaluated on ``grid``.

    PCHIP is monotone non-decreasing here (the knots are), so its analytic
    derivative is a valid (non-negative) density. Below/above the knot span the
    CDF is held flat at its end values (density 0) for a coherent extrapolation.
    """
    pchip = PchipInterpolator(knot_k, knot_cdf, extrapolate=False)
    deriv = pchip.derivative()

    cdf = pchip(grid)
    pdf = deriv(grid)

    # Outside the knot range PchipInterpolator(extrapolate=False) -> NaN.
    k_min, k_max = float(knot_k[0]), float(knot_k[-1])
    lo_val, hi_val = float(knot_cdf[0]), float(knot_cdf[-1])
    below = grid < k_min
    above = grid > k_max
    cdf = np.where(below, lo_val, cdf)
    cdf = np.where(above, hi_val, cdf)
    pdf = np.where(below | above, 0.0, pdf)
    pdf = np.nan_to_num(pdf, nan=0.0)
    cdf = np.nan_to_num(cdf, nan=0.0)
    return pdf, np.clip(cdf, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Shape: terminal_buckets
# ---------------------------------------------------------------------------


def _cdf_knots_terminal_buckets(
    family: LadderFamily, tail_model: TailModel, warnings: list[str]
) -> tuple[np.ndarray, np.ndarray, list[MarketPoint], float, float]:
    """Build CDF knots from range buckets.

    Returns:
        ``(knot_k, knot_cdf, market_points, k_lo, k_hi)`` where ``knot_k`` are
        bucket upper edges (plus a synthetic lower edge) and ``knot_cdf`` is the
        cumulative renormalised mass.
    """
    buckets: list[tuple[float | None, float | None, float]] = []
    for e in family.entries:
        if e.direction != "between":
            continue
        p = _safe_float(e.prob)
        if p is None:
            continue
        floor = _safe_float(e.floor)
        cap = _safe_float(e.cap)
        if floor is None and cap is None:
            continue
        buckets.append((floor, cap, max(0.0, p)))

    if not buckets:
        raise ValueError("terminal_buckets: no usable 'between' entries")

    # Sort by a representative edge (cap for upper-open uses cap; otherwise floor).
    def _sort_key(b: tuple[float | None, float | None, float]) -> float:
        floor, cap, _ = b
        if floor is not None and cap is not None:
            return 0.5 * (floor + cap)
        return floor if floor is not None else cap  # type: ignore[return-value]

    buckets.sort(key=_sort_key)

    total = sum(m for _, _, m in buckets)
    if total <= 0:
        raise ValueError("terminal_buckets: total mass is non-positive")
    if abs(total - 1.0) > 0.02:
        warnings.append(f"overround: bucket masses summed to {total:.4f}; renormalised to 1.0")
    buckets = [(f, c, m / total) for f, c, m in buckets]

    # Determine finite edges to size the grid; estimate tail widths.
    finite_edges = [v for f, c, _ in buckets for v in (f, c) if v is not None]
    k_min = min(finite_edges)
    k_max = max(finite_edges)
    typ_width = (k_max - k_min) / max(len(buckets), 1)
    if typ_width <= 0:
        typ_width = max(abs(k_max), 1.0) * 0.1

    market_points: list[MarketPoint] = []
    # Knots at upper edges, accumulating mass.
    knot_k: list[float] = []
    knot_cdf: list[float] = []
    cum = 0.0

    # Lower anchor: bottom edge of the first bucket (open-tail synthesised).
    first_floor, first_cap, _ = buckets[0]
    if first_floor is None:
        # Open lower tail — extend a synthetic floor for the grid.
        lo_anchor = (first_cap if first_cap is not None else k_min) - _tail_width(
            tail_model, typ_width
        )
    else:
        lo_anchor = first_floor
    knot_k.append(lo_anchor)
    knot_cdf.append(0.0)

    for floor, cap, mass in buckets:
        cum += mass
        upper = cap
        if upper is None:
            # Open upper tail — synthesise an edge so the CDF reaches ~1.
            upper = (floor if floor is not None else k_max) + _tail_width(tail_model, typ_width)
        # Representative strike for transparency.
        if floor is not None and cap is not None:
            rep = 0.5 * (floor + cap)
        elif floor is not None:
            rep = floor
        else:
            rep = cap  # type: ignore[assignment]
        market_points.append(
            MarketPoint(k=float(rep), prob=float(mass), kind="mass", floor=floor, cap=cap)
        )
        if knot_k and upper <= knot_k[-1]:
            upper = knot_k[-1] + max(typ_width * 0.01, 1e-6)
        knot_k.append(float(upper))
        knot_cdf.append(min(cum, 1.0))

    arr_k = np.asarray(knot_k, dtype=float)
    arr_cdf = np.asarray(knot_cdf, dtype=float)
    return arr_k, arr_cdf, market_points, float(arr_k[0]), float(arr_k[-1])


def _tail_width(tail_model: TailModel, typ_width: float) -> float:
    """Half-width to extend an open tail bucket, by tail model."""
    if tail_model == "none":
        return typ_width * 0.5
    if tail_model == "linear":
        return typ_width * 1.0
    return typ_width * 1.5  # lognormal-style fatter tail


# ---------------------------------------------------------------------------
# Shape: terminal_ladder
# ---------------------------------------------------------------------------


def _cdf_knots_terminal_ladder(
    family: LadderFamily, eps: float, warnings: list[str]
) -> tuple[np.ndarray, np.ndarray, list[MarketPoint], float, float]:
    """Build CDF knots from above/below threshold markets."""
    above: list[tuple[float, float]] = []
    below: list[tuple[float, float]] = []
    for e in family.entries:
        if e.strike is None:
            continue
        p = _safe_float(e.prob)
        k = _safe_float(e.strike)
        if p is None or k is None:
            continue
        if e.direction == "above":
            above.append((k, p))
        elif e.direction == "below":
            below.append((k, p))

    pairs: list[tuple[float, float, str]] = []  # (K, F(K), kind)
    market_points: list[MarketPoint] = []

    if above:
        above.sort(key=lambda r: r[0])
        ks = [k for k, _ in above]
        surv = _enforce_monotone(ks, [p for _, p in above])
        for (k, raw), s in zip(above, surv, strict=True):
            s_clipped = float(np.clip(s, eps, 1.0 - eps))
            pairs.append((k, 1.0 - s_clipped, "above"))
            market_points.append(MarketPoint(k=float(k), prob=float(raw), kind="survival"))

    if below:
        below.sort(key=lambda r: r[0])
        # F(K) is non-decreasing; reuse _enforce_monotone on the complement.
        ks = [k for k, _ in below]
        comp = _enforce_monotone(ks, [1.0 - p for _, p in below])
        for (k, raw), c in zip(below, comp, strict=True):
            f_clipped = float(np.clip(1.0 - c, eps, 1.0 - eps))
            pairs.append((k, f_clipped, "below"))
            market_points.append(MarketPoint(k=float(k), prob=float(raw), kind="cdf"))

    if not pairs:
        raise ValueError("terminal_ladder: no usable above/below entries")

    # Merge by strike (keep last-seen on collision), sort, de-dup ties.
    pairs.sort(key=lambda r: r[0])
    knot_k: list[float] = []
    knot_cdf: list[float] = []
    for k, f, _kind in pairs:
        if knot_k and abs(k - knot_k[-1]) < 1e-12:
            knot_cdf[-1] = max(knot_cdf[-1], f)
            continue
        knot_k.append(k)
        knot_cdf.append(f)

    # Enforce non-decreasing CDF across the merged knots.
    for i in range(1, len(knot_cdf)):
        knot_cdf[i] = max(knot_cdf[i], knot_cdf[i - 1])

    if len(knot_k) < 2:
        raise ValueError("terminal_ladder: need at least 2 distinct strikes")

    arr_k = np.asarray(knot_k, dtype=float)
    arr_cdf = np.asarray(knot_cdf, dtype=float)
    return arr_k, arr_cdf, market_points, float(arr_k[0]), float(arr_k[-1])


# ---------------------------------------------------------------------------
# Shape: barrier_touch
# ---------------------------------------------------------------------------


def _survival_knots_barrier(
    family: LadderFamily, eps: float
) -> tuple[np.ndarray, np.ndarray, list[MarketPoint], str, Literal["touch_above", "touch_below"]]:
    """Return ``(K, survival, market_points, distribution_of, touch_dir)``.

    For ``touch_above`` the survival is ``P(M_T >= K)`` (running max), monotone
    non-increasing. For ``touch_below`` it is ``P(m_T <= K)`` (running min),
    monotone non-decreasing in K — we still expose it as a survival-style
    monotone series for the CDF construction.
    """
    touch_above: list[tuple[float, float]] = []
    touch_below: list[tuple[float, float]] = []
    for e in family.entries:
        if e.strike is None:
            continue
        p = _safe_float(e.prob)
        k = _safe_float(e.strike)
        if p is None or k is None:
            continue
        if e.direction == "touch_above":
            touch_above.append((k, p))
        elif e.direction == "touch_below":
            touch_below.append((k, p))

    if touch_above and not touch_below:
        touch_above.sort(key=lambda r: r[0])
        ks = [k for k, _ in touch_above]
        surv = _enforce_monotone(ks, [p for _, p in touch_above])
        surv = [float(np.clip(s, eps, 1.0 - eps)) for s in surv]
        mp = [MarketPoint(k=float(k), prob=float(p), kind="survival") for k, p in touch_above]
        return (
            np.asarray(ks, dtype=float),
            np.asarray(surv, dtype=float),
            mp,
            "running_max",
            "touch_above",
        )
    if touch_below and not touch_above:
        # P(m_T <= K) is non-decreasing in K -> a CDF of the running min.
        touch_below.sort(key=lambda r: r[0])
        ks = [k for k, _ in touch_below]
        comp = _enforce_monotone(ks, [1.0 - p for _, p in touch_below])
        cdf = [float(np.clip(1.0 - c, eps, 1.0 - eps)) for c in comp]
        mp = [MarketPoint(k=float(k), prob=float(p), kind="cdf") for k, p in touch_below]
        return (
            np.asarray(ks, dtype=float),
            np.asarray(cdf, dtype=float),
            mp,
            "running_min",
            "touch_below",
        )

    raise ValueError(
        "barrier_touch: provide a single-direction ladder of touch_above OR "
        "touch_below entries (got mixed/empty)"
    )


def _running_max_survival(
    k: np.ndarray, s0: float, nu: float, sigma: float, t: float
) -> np.ndarray:
    """Closed-form GBM running-max survival ``P(M_T >= K)`` (reflection principle).

    ``a = ln(K/S0)``; for ``K <= S0`` (``a <= 0``) the max has already exceeded
    K with probability 1.
    """
    sig_sqrt_t = sigma * math.sqrt(t)
    a = np.log(np.maximum(k, 1e-12) / s0)
    z1 = (-a + nu * t) / sig_sqrt_t
    z2 = (-a - nu * t) / sig_sqrt_t
    ratio = np.power(np.maximum(k, 1e-12) / s0, 2.0 * nu / (sigma * sigma))
    surv = norm.cdf(z1) + ratio * norm.cdf(z2)
    surv = np.where(a <= 0.0, 1.0, surv)
    return np.clip(surv, 0.0, 1.0)


def _fit_gbm_running_max(
    knot_k: np.ndarray, observed_surv: np.ndarray, s0: float, t: float
) -> tuple[float, float, float]:
    """Least-squares fit of GBM ``(ν, σ)`` to the running-max survival ladder.

    Returns ``(nu, sigma, rmse)``. ``sigma`` is bounded to ``(0.01, 5.0)``.
    """

    def objective(params: np.ndarray) -> float:
        nu, sigma = float(params[0]), float(params[1])
        if sigma <= 0.01 or sigma >= 5.0 or not math.isfinite(nu):
            return 1e9
        pred = _running_max_survival(knot_k, s0, nu, sigma, t)
        resid = observed_surv - pred
        return float(np.sum(resid * resid))

    best = (0.0, 0.3, float("inf"))
    for nu0 in (-0.2, 0.0, 0.2):
        for sig0 in (0.2, 0.5, 1.0):
            try:
                res = minimize(
                    objective,
                    x0=np.asarray([nu0, sig0]),
                    method="Nelder-Mead",
                    options={"maxiter": 2000, "xatol": 1e-7, "fatol": 1e-12},
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.info("gbm running-max fit failed: %s", exc)
                continue
            if res.success and math.isfinite(res.fun) and res.fun < best[2]:
                best = (float(res.x[0]), float(res.x[1]), float(res.fun))

    nu, sigma, sse = best
    sigma = float(np.clip(sigma, 0.01, 5.0))
    n = max(len(knot_k), 1)
    rmse = math.sqrt(max(sse, 0.0) / n)
    return nu, sigma, rmse


def _gbm_terminal_overlay(
    grid: np.ndarray, s0: float, nu: float, sigma: float, t: float
) -> np.ndarray:
    """Terminal-price PDF under the fitted GBM, evaluated on ``grid``.

    Terminal exceedance ``P(S_T>=K)=Φ((-a+νT)/(σ√T))``, ``a=ln(K/S0)`` →
    ``F(K)=1-that`` → analytic density of a lognormal with log-mean
    ``ln(S0)+νT`` and log-sd ``σ√T``.
    """
    sig_sqrt_t = sigma * math.sqrt(t)
    mu_log = math.log(max(s0, 1e-12)) + nu * t
    pos = grid > 0
    out = np.zeros_like(grid)
    g = grid[pos]
    out[pos] = np.exp(-((np.log(g) - mu_log) ** 2) / (2.0 * sig_sqrt_t**2)) / (
        g * sig_sqrt_t * math.sqrt(2.0 * math.pi)
    )
    return np.nan_to_num(out, nan=0.0)


# ---------------------------------------------------------------------------
# method="lognormal" primary path
# ---------------------------------------------------------------------------


def _lognormal_primary_pdf(grid: np.ndarray, mean: float, std: float) -> np.ndarray | None:
    """Smooth lognormal PDF on ``grid`` from moment-matched ``(mean, std)``."""
    overlay = _lognormal_overlay(grid, mean, std)
    if overlay is None:
        return None
    return np.asarray(overlay, dtype=float)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def compute_implied_pdf(
    family: LadderFamily,
    *,
    method: SmoothMethod = "pchip_monotone",
    eps: float = DEFAULT_EPSILON,
    grid_size: int = DEFAULT_GRID_SIZE,
    barrier_to_terminal: bool = False,
    tail_model: TailModel = "lognormal",
    now_utc: datetime | None = None,
) -> ImpliedPDFResult:
    """Compute a dense implied PDF/CDF for a same-maturity contract family.

    Args:
        family: The :class:`LadderFamily` (one of three ``data_shape`` values).
        method: ``"pchip_monotone"`` (default), ``"lognormal"`` (smooth
            moment-matched primary), or ``"empirical"`` (treated as PCHIP).
        eps: Probability clip applied to ladder survival/CDF probabilities.
        grid_size: Number of points in the dense output grid.
        barrier_to_terminal: For ``barrier_touch``, also fit GBM and emit a
            terminal-price overlay + :class:`GBMFit`.
        tail_model: Open-tail extension for ``terminal_buckets``.
        now_utc: Reference time for time-to-maturity (defaults to current UTC).

    Returns:
        A populated :class:`ImpliedPDFResult`.

    Raises:
        ValueError: On empty/degenerate input (router maps to HTTP 422).
    """
    grid_size = max(grid_size, 8)

    warnings: list[str] = []
    entries: list[LadderEntry] = list(family.entries or [])
    if not entries:
        raise ValueError("no entries in ladder family")

    t_years = _time_to_maturity_years(family.maturity_utc, now_utc, warnings)
    shape = family.data_shape
    distribution_of: str = "terminal_price"
    gbm_fit: GBMFit | None = None
    gbm_terminal_overlay: list[float] | None = None

    # ----- shape-specific knot construction -------------------------------
    if shape == "terminal_buckets":
        knot_k, knot_cdf, market_points, k_lo, k_hi = _cdf_knots_terminal_buckets(
            family, tail_model, warnings
        )
        spot = family.spot
    elif shape == "terminal_ladder":
        knot_k, knot_cdf, market_points, k_lo, k_hi = _cdf_knots_terminal_ladder(
            family, eps, warnings
        )
        spot = family.spot
    elif shape == "barrier_touch":
        surv_k, surv_p, market_points, distribution_of, touch_dir = _survival_knots_barrier(
            family, eps
        )
        warnings.append(
            f"barrier_touch: PDF is the model-free law of the {distribution_of} "
            "(running extremum), NOT the terminal price"
        )
        # Build a CDF from the survival/CDF series.
        if touch_dir == "touch_above":
            cdf_vals = 1.0 - surv_p  # F of running max
        else:
            cdf_vals = surv_p.copy()  # already a CDF of running min
        # Sort and monotonise.
        order = np.argsort(surv_k)
        knot_k = surv_k[order]
        knot_cdf = np.maximum.accumulate(cdf_vals[order])
        k_lo, k_hi = float(knot_k[0]), float(knot_k[-1])

        spot = family.spot
        if barrier_to_terminal:
            if spot is None:
                # Estimate S0 as strike where running-max survival ≈ 0.5.
                if touch_dir == "touch_above":
                    spot = float(np.interp(0.5, (1.0 - knot_cdf)[::-1], knot_k[::-1]))
                else:
                    spot = float(np.interp(0.5, knot_cdf, knot_k))
                warnings.append(f"spot estimated from survival≈0.5 crossing: {spot:.4g}")
            # Fit GBM only meaningful for the running-max (touch_above) form.
            if touch_dir == "touch_above":
                nu, sigma, rmse = _fit_gbm_running_max(knot_k, surv_p, spot, t_years)
            else:
                # touch_below: fit to the equivalent running-min survival
                # P(m_T <= K) = 1 - F; reuse the max machinery on reflected K.
                nu, sigma, rmse = _fit_gbm_running_max(knot_k, surv_p, spot, t_years)
    else:  # pragma: no cover - Literal guards this
        raise ValueError(f"unknown data_shape: {shape!r}")

    if len(knot_k) < 2:
        raise ValueError("need at least 2 usable strikes/edges to build a CDF")

    n_strikes = len(entries)
    if n_strikes < 3:
        warnings.append(f"few_strikes: only {n_strikes} entries (<3)")

    # Reject degenerate (all-equal) inputs.
    if float(np.ptp(knot_cdf)) < 1e-9:
        raise ValueError("degenerate input: CDF is flat (all probabilities equal)")

    # ----- dense grid + density ------------------------------------------
    grid = _build_grid(k_lo, k_hi, grid_size)

    # Empirical moments for the lognormal overlay (built from knot survival).
    surv_for_moments = (1.0 - np.clip(knot_cdf, 0.0, 1.0)).tolist()
    mom_in = _empirical_moments(knot_k.tolist(), surv_for_moments)
    overlay_mean = mom_in["mean"]
    overlay_std = mom_in["std"]

    primary_method: SmoothMethod = method
    if method == "lognormal":
        ln_pdf = _lognormal_primary_pdf(grid, overlay_mean, overlay_std)
        if ln_pdf is None:
            warnings.append("lognormal primary fit failed; fell back to PCHIP")
            pdf, _cdf_grid = _pchip_cdf_to_density(knot_k, knot_cdf, grid)
            primary_method = "pchip_monotone"
        else:
            pdf = ln_pdf
    else:
        pdf, _cdf_grid = _pchip_cdf_to_density(knot_k, knot_cdf, grid)

    pdf = _renormalise_pdf(grid, pdf)
    cdf = _cdf_from_pdf(grid, pdf)

    moments = _moments_from_grid(grid, pdf)
    quantiles = _quantiles_from_grid(grid, pdf)

    # ----- overlays -------------------------------------------------------
    lognormal_overlay = _lognormal_overlay(grid, overlay_mean, overlay_std)
    if lognormal_overlay is None:
        warnings.append("lognormal overlay fit failed (non-positive moments)")
    elif shape == "barrier_touch":
        warnings.append(
            "lognormal_overlay is a reference for the running extremum, not the terminal price"
        )

    if shape == "barrier_touch" and barrier_to_terminal:
        overlay_arr = _gbm_terminal_overlay(grid, float(spot), nu, sigma, t_years)
        overlay_arr = _renormalise_pdf(grid, overlay_arr)
        gbm_terminal_overlay = [float(v) for v in overlay_arr]
        gbm_fit = GBMFit(
            sigma_annual=float(sigma),
            nu_annual=float(nu),
            rmse=float(rmse),
            converted_to_terminal=True,
        )

    return ImpliedPDFResult(
        asset=family.asset,
        data_shape=shape,
        distribution_of=distribution_of,  # type: ignore[arg-type]
        maturity_utc=family.maturity_utc,
        time_to_maturity_years=t_years,
        spot=spot,
        grid=[float(v) for v in grid],
        pdf=[float(v) for v in pdf],
        cdf=[float(v) for v in cdf],
        market_points=market_points,
        lognormal_overlay=lognormal_overlay,
        gbm_terminal_overlay=gbm_terminal_overlay,
        gbm_fit=gbm_fit,
        moments=moments,
        quantiles=quantiles,
        method=primary_method,
        eps=eps,
        n_strikes=n_strikes,
        warnings=warnings,
    )


__all__ = ["compute_implied_pdf"]
