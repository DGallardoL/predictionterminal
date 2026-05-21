"""Cross-venue risk-neutral comparison and the empirical pricing kernel.

Brings three terminal-distribution estimates of the same equity index onto one
strike grid and contrasts them:

* **Kalshi-Q** — prediction-market risk-neutral density (:mod:`pfm.vol.implied_pdf`).
* **Options-Q** — option-implied risk-neutral density from the second derivative
  of the call-price curve under shape restrictions (monotone + convex)
  (:mod:`pfm.vol.options_rn`).
* **Physical-P** — real-world density from a GARCH vol forecast
  (:mod:`pfm.vol.physical_density`).

From these we report the empirical pricing kernel and implied risk aversion:

* **Cross-venue divergence** between the two *risk-neutral* measures —
  ``KL(Kalshi‖Options)``, symmetric Jensen-Shannon, and the pointwise ratio
  ``f_Kalshi/f_Options`` (where Kalshi over/under-prices vs listed options — a
  tradeable cross-venue mispricing signal).
* **Pricing kernel / SDF** ``M(S) = e^{-rτ}·f_Q(S)/f_P(S)`` (options-Q over the
  physical density).
* **Implied absolute risk aversion** ``ρ(S) = f_P'/f_P − f_Q'/f_Q = −d/dS ln M``,
  and relative risk aversion ``S·ρ(S)``.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_trapz = getattr(np, "trapezoid", None) or np.trapz
_EPS = 1e-12


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class DensitySeries(BaseModel):
    """One density on the shared grid plus its first two moments."""

    label: str
    measure: str  # "risk_neutral" | "physical"
    venue: str  # "kalshi" | "options" | "garch"
    pdf: list[float]
    cdf: list[float]
    mean: float
    std: float
    note: str | None = None


class CrossVenue(BaseModel):
    """Divergence between the two risk-neutral measures (Kalshi vs options)."""

    kl_kalshi_given_options: float = Field(..., description="KL(Kalshi ‖ Options), nats")
    kl_options_given_kalshi: float = Field(..., description="KL(Options ‖ Kalshi), nats")
    jensen_shannon: float = Field(..., description="Symmetric JS divergence, nats")
    mean_gap: float = Field(..., description="mean(Kalshi) − mean(Options)")
    std_ratio: float = Field(..., description="std(Kalshi) / std(Options)")
    ratio: list[float] = Field(..., description="pointwise f_Kalshi / f_Options on the grid")


class OpportunityModel(BaseModel):
    """One executable relative-value opportunity vs the options fair value."""

    slug: str
    kind: str
    lo: float | None
    hi: float | None
    fair_value: float = Field(..., description="discounted options-implied fair price")
    physical_prob: float
    kalshi_bid: float | None
    kalshi_ask: float | None
    spread: float | None
    volume: float | None
    open_interest: float | None
    action: str  # "BUY @ask" | "SELL @bid"
    edge: float = Field(..., description="executable edge in probability points")
    confidence: str  # "high" | "low"
    note: str = ""


class MarketQuality(BaseModel):
    """How tradeable the chosen Kalshi ladder actually is."""

    n_contracts: int
    n_executable: int
    n_opportunities: int
    median_spread: float | None
    total_volume: float
    tradeable: bool


class FairRow(BaseModel):
    """Per-contract theoretical fair value vs the live Kalshi quote."""

    slug: str
    kind: str
    lo: float | None
    hi: float | None
    kalshi_bid: float | None
    kalshi_ask: float | None
    kalshi_mid: float | None
    fair_value: float
    physical_prob: float
    gap: float | None  # kalshi_mid − fair_value (>0 = Kalshi rich vs options)


class PricingKernelResult(BaseModel):
    """Full cross-venue + pricing-kernel response for one asset."""

    asset: str
    ladder_key: str = Field("", description="Kalshi series/asset the ladder came from")
    maturity_label: str = Field("", description="human label for the chosen expiry")
    spot: float
    forward: float
    maturity_utc: datetime
    time_to_maturity_years: float
    risk_free: float
    annual_drift: float
    grid: list[float]
    densities: list[DensitySeries]
    cross_venue: CrossVenue
    opportunities: list[OpportunityModel] = Field(default_factory=list)
    fair_table: list[FairRow] = Field(default_factory=list)
    market_quality: MarketQuality | None = None
    pricing_kernel: list[float | None] = Field(
        ...,
        description="M(S)=e^{-rτ} f_Q^options/f_P; null where the physical density is negligible",
    )
    implied_risk_aversion: list[float | None] = Field(
        ..., description="absolute risk aversion ρ(S)=f_P'/f_P − f_Q'/f_Q"
    )
    relative_risk_aversion: list[float | None] = Field(..., description="S·ρ(S)")
    variance_risk_premium: float = Field(
        ..., description="options-Q variance minus physical variance (annualised vol² gap)"
    )
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------


def _interp_density(grid: np.ndarray, src_grid: np.ndarray, src_pdf: np.ndarray) -> np.ndarray:
    """Resample a density onto ``grid`` and renormalise to integrate to 1."""
    pdf = np.interp(grid, src_grid, src_pdf, left=0.0, right=0.0)
    pdf = np.clip(pdf, 0.0, None)
    area = float(_trapz(pdf, grid))
    return pdf / area if area > 0 else pdf


def _moments(grid: np.ndarray, pdf: np.ndarray) -> tuple[float, float]:
    mean = float(_trapz(grid * pdf, grid))
    var = float(_trapz((grid - mean) ** 2 * pdf, grid))
    return mean, float(np.sqrt(max(var, 0.0)))


def _cdf(grid: np.ndarray, pdf: np.ndarray) -> np.ndarray:
    c = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(grid))])
    return c / c[-1] if c[-1] > 0 else c


def _kl(p: np.ndarray, q: np.ndarray, grid: np.ndarray) -> float:
    """KL(p‖q) in nats, with a floor to keep the log finite."""
    p = np.clip(p, _EPS, None)
    q = np.clip(q, _EPS, None)
    return float(_trapz(p * np.log(p / q), grid))


def _support_range(
    grid: np.ndarray, cdf: np.ndarray, lo_q: float, hi_q: float
) -> tuple[float, float]:
    lo = float(grid[np.searchsorted(cdf, lo_q)]) if cdf[-1] > 0 else float(grid[0])
    hi_i = min(np.searchsorted(cdf, hi_q), len(grid) - 1)
    return lo, float(grid[hi_i])


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def compute_pricing_kernel(
    asset: str,
    kalshi_client: Any,
    *,
    ladder_key: str | None = None,
    maturity: str | None = None,
    now_utc: datetime | None = None,
    risk_free: float = 0.045,
    annual_drift: float = 0.06,
    grid_size: int = 240,
    lookback_days: int = 400,
) -> PricingKernelResult:
    """Build the three densities, compare the RN measures, and recover the kernel.

    Args:
        asset: ``"SPX"`` or ``"NDX"`` — drives the options/physical underlying.
        kalshi_client: Kalshi client for the prediction-market ladder.
        ladder_key: Kalshi series/asset to pull the ladder from (e.g. ``"KXINXY"``
            for the year-end market). Defaults to ``asset`` (the daily ladder).
        maturity: ISO-date prefix to disambiguate which Kalshi event/expiry.
        now_utc: Reference now (for maturities); defaults to current UTC.
        risk_free: Annualised risk-free rate.
        annual_drift: Physical expected return for the P-measure density.
        grid_size: Shared-grid resolution.
        lookback_days: History window for the GARCH physical-vol fit.

    Returns:
        :class:`PricingKernelResult`.

    Raises:
        ValueError: if the Kalshi ladder or option chain cannot be priced.
    """
    from pfm.sources.kalshi import discover_index_ladder
    from pfm.vol.implied_pdf import compute_implied_pdf
    from pfm.vol.options_rn import extract_options_rn
    from pfm.vol.physical_density import estimate_physical_density
    from pfm.vol.pricing_kernel_strategies import fair_value_rows, scan_opportunities

    now = now_utc or datetime.now(UTC)
    warnings: list[str] = []

    # --- Kalshi-Q ----------------------------------------------------------
    key = ladder_key or asset
    family = discover_index_ladder(key, client=kalshi_client, maturity_filter=maturity)
    kalshi = compute_implied_pdf(family)
    expiry = kalshi.maturity_utc
    t_years = max(float(kalshi.time_to_maturity_years), 1e-4)
    kalshi_grid = np.asarray(kalshi.grid, dtype=float)
    kalshi_pdf = np.asarray(kalshi.pdf, dtype=float)

    # --- Options-Q (shape-restricted, from the call-curve second derivative) ----------------
    target_expiry = expiry.date().isoformat()
    # Reprice the options density to the Kalshi horizon so the two RN measures are
    # compared at the SAME maturity (Kalshi index markets are not always 0/1DTE).
    options = extract_options_rn(
        asset,
        target_expiry=target_expiry,
        risk_free=risk_free,
        now_utc=now,
        price_t_years=t_years,
    )
    warnings.extend(options.warnings)
    spot = float(kalshi.spot or family.spot or options.forward)

    # --- Physical-P (GARCH) ------------------------------------------------
    horizon_days = max(t_years * 365.25, 0.04)
    physical = estimate_physical_density(
        asset,
        spot=spot,
        t_years=t_years,
        horizon_days=horizon_days,
        annual_drift=annual_drift,
        risk_free=risk_free,
        lookback_days=lookback_days,
    )
    warnings.extend(physical.warnings)

    # --- Shared grid: anchor on the (clean, tight) options support ---------
    lo, hi = _support_range(options.grid, options.cdf, 0.004, 0.996)
    # widen a touch so the wider Kalshi density isn't clipped at its shoulders
    pad = 0.35 * (hi - lo)
    grid = np.linspace(lo - pad, hi + pad, grid_size)

    q_kalshi = _interp_density(grid, kalshi_grid, kalshi_pdf)
    q_options = _interp_density(grid, np.asarray(options.grid), np.asarray(options.pdf))
    p_phys = _interp_density(grid, np.asarray(physical.grid), np.asarray(physical.pdf))

    m_k, s_k = _moments(grid, q_kalshi)
    m_o, s_o = _moments(grid, q_options)
    m_p, s_p = _moments(grid, p_phys)

    # --- Cross-venue divergence (RN vs RN) ---------------------------------
    js = 0.5 * _kl(q_kalshi, 0.5 * (q_kalshi + q_options), grid) + 0.5 * _kl(
        q_options, 0.5 * (q_kalshi + q_options), grid
    )
    ratio = np.clip(q_kalshi, _EPS, None) / np.clip(q_options, _EPS, None)
    cross = CrossVenue(
        kl_kalshi_given_options=_kl(q_kalshi, q_options, grid),
        kl_options_given_kalshi=_kl(q_options, q_kalshi, grid),
        jensen_shannon=js,
        mean_gap=m_k - m_o,
        std_ratio=(s_k / s_o) if s_o > 0 else float("nan"),
        ratio=_round_list(ratio, where=(q_options > _EPS) | (q_kalshi > _EPS)),
    )

    # --- Pricing kernel + implied risk aversion (options-Q over physical) --
    disc = float(np.exp(-risk_free * t_years))
    # Kernel/risk-aversion are only meaningful in the central region where BOTH
    # densities carry real mass — in the tails f_P → 0 makes the ratio and the
    # log-derivatives explode into numerical noise, so we restrict to the
    # data-rich interior. 2% of each peak ≈ the central ~95%.
    mask = (p_phys > 0.02 * float(p_phys.max())) & (q_options > 0.02 * float(q_options.max()))
    kernel: list[float | None] = [None] * grid.size
    rra: list[float | None] = [None] * grid.size
    rel_rra: list[float | None] = [None] * grid.size
    if mask.sum() >= 5:
        log_p = np.log(np.clip(p_phys, _EPS, None))
        log_q = np.log(np.clip(q_options, _EPS, None))
        d_log_p = np.gradient(log_p, grid)
        d_log_q = np.gradient(log_q, grid)
        rho = d_log_p - d_log_q  # absolute risk aversion
        m_vals = disc * np.clip(q_options, _EPS, None) / np.clip(p_phys, _EPS, None)
        for i in np.where(mask)[0]:
            kernel[i] = round(float(m_vals[i]), 6)
            rra[i] = round(float(rho[i]), 8)
            rel_rra[i] = round(float(grid[i] * rho[i]), 4)

    # --- Variance risk premium (annualised σ² gap) -------------------------
    sig_q_ann = (s_o / spot) / np.sqrt(t_years) if spot > 0 and t_years > 0 else 0.0
    sig_p_ann = physical.sigma_ann
    vrp = float(sig_q_ann**2 - sig_p_ann**2)

    densities = [
        DensitySeries(
            label="Kalshi (prediction market)",
            measure="risk_neutral",
            venue="kalshi",
            pdf=_round_list(q_kalshi),
            cdf=_round_list(_cdf(grid, q_kalshi)),
            mean=round(m_k, 2),
            std=round(s_k, 2),
            note=f"n={kalshi.n_strikes} strikes",
        ),
        DensitySeries(
            label="Options (risk-neutral density)",
            measure="risk_neutral",
            venue="options",
            pdf=_round_list(q_options),
            cdf=_round_list(_cdf(grid, q_options)),
            mean=round(m_o, 2),
            std=round(s_o, 2),
            note=f"{options.n_options} OTM strikes, ATM IV {options.atm_iv:.1%}",
        ),
        DensitySeries(
            label="Physical (GARCH)",
            measure="physical",
            venue="garch",
            pdf=_round_list(p_phys),
            cdf=_round_list(_cdf(grid, p_phys)),
            mean=round(m_p, 2),
            std=round(s_p, 2),
            note=f"σ_ann {physical.sigma_ann:.1%}"
            + ("" if physical.garch_converged else " (sample-σ fallback)"),
        ),
    ]

    # --- Executable opportunities (theoretical fair price vs the live quote) --
    opt_lo = float(options.forward * np.exp(float(options.smile_k.min())))
    opt_hi = float(options.forward * np.exp(float(options.smile_k.max())))
    opps, mkt_quality = scan_opportunities(
        family.entries,
        np.asarray(options.grid),
        np.asarray(options.cdf),
        np.asarray(physical.grid),
        np.asarray(physical.cdf),
        opt_strike_lo=opt_lo,
        opt_strike_hi=opt_hi,
        discount=disc,
    )
    opportunities = [
        OpportunityModel(
            slug=o.slug,
            kind=o.kind,
            lo=o.lo,
            hi=o.hi,
            fair_value=o.fair_value,
            physical_prob=o.physical_prob,
            kalshi_bid=o.kalshi_bid,
            kalshi_ask=o.kalshi_ask,
            spread=o.spread,
            volume=o.volume,
            open_interest=o.open_interest,
            action=o.action,
            edge=o.edge,
            confidence=o.confidence,
            note=o.note,
        )
        for o in opps
    ]
    market_quality = MarketQuality(**mkt_quality)
    fair_table = [
        FairRow(**row)
        for row in fair_value_rows(
            family.entries,
            np.asarray(options.grid),
            np.asarray(options.cdf),
            np.asarray(physical.grid),
            np.asarray(physical.cdf),
            discount=disc,
        )
    ]
    maturity_label = f"{expiry.date().isoformat()} · {t_years * 365.25:.0f}d"

    return PricingKernelResult(
        asset=asset,
        ladder_key=key,
        maturity_label=maturity_label,
        spot=round(spot, 2),
        forward=round(float(options.forward), 2),
        maturity_utc=expiry,
        time_to_maturity_years=round(t_years, 6),
        risk_free=risk_free,
        annual_drift=annual_drift,
        grid=_round_list(grid, dp=2),
        densities=densities,
        cross_venue=cross,
        opportunities=opportunities,
        fair_table=fair_table,
        market_quality=market_quality,
        pricing_kernel=kernel,
        implied_risk_aversion=rra,
        relative_risk_aversion=rel_rra,
        variance_risk_premium=round(vrp, 6),
        warnings=warnings,
    )


#: Terminal-distribution Kalshi series per asset (range buckets — NOT the
#: running-max/min yearly markets, whose law is the extremum, not the terminal).
_TERMINAL_SERIES: dict[str, list[str]] = {
    "SPX": ["KXINX", "KXINXY"],
    "NDX": ["KXNDX", "KXNDXY"],
}


def list_index_maturities(asset: str, kalshi_client: Any) -> list[dict[str, Any]]:
    """Enumerate the asset's Kalshi index expiries with a liquidity summary.

    Powers the UI date selector: each entry says which expiry it is and how
    tradeable it currently is (two-sided quotes, spread), so the daily
    dead-markets are visibly distinguished from the liquid year-end one.

    Returns:
        Maturities sorted by date, each a dict with ``ladder_key``,
        ``event_ticker``, ``maturity_date``, ``time_to_maturity_days``,
        ``n_markets``, ``n_two_sided``, ``median_spread``, ``liquid``.
    """
    from pfm.sources.kalshi import _event_sort_key, _yes_bid_ask

    out: list[dict[str, Any]] = []
    now = datetime.now(UTC)
    for series in _TERMINAL_SERIES.get(asset.upper(), []):
        try:
            events = kalshi_client.get_events(series_ticker=series, with_nested_markets=True)
        except Exception:  # series may not exist for this asset
            continue
        for ev in events or []:
            mks = ev.get("markets") or []
            if not mks:
                continue
            two, spreads = 0, []
            for m in mks:
                b, a = _yes_bid_ask(m)
                if b is not None and a is not None and b > 0 and a < 1:
                    two += 1
                    spreads.append(a - b)
            try:
                mat = _event_sort_key(ev)
            except Exception:
                continue
            ttm_days = max((mat - now).total_seconds() / 86_400.0, 0.0)
            out.append(
                {
                    "ladder_key": series,
                    "event_ticker": ev.get("event_ticker"),
                    "maturity_date": mat.date().isoformat(),
                    "time_to_maturity_days": round(ttm_days, 2),
                    "n_markets": len(mks),
                    "n_two_sided": two,
                    "median_spread": round(float(sorted(spreads)[len(spreads) // 2]), 4)
                    if spreads
                    else None,
                    "liquid": two >= max(6, len(mks) // 2),
                }
            )
    out.sort(key=lambda d: d["maturity_date"])
    return out


def _round_list(arr: np.ndarray, *, dp: int = 8, where: np.ndarray | None = None) -> list[float]:
    """Round a numpy array to a JSON-friendly list (NaN/inf → 0)."""
    a = np.asarray(arr, dtype=float)
    a = np.where(np.isfinite(a), a, 0.0)
    if where is not None:
        a = np.where(where, a, 0.0)
    return [round(float(x), dp) for x in a]
