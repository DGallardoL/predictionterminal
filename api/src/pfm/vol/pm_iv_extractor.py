"""Multi-asset Polymarket+Kalshi implied-volatility extractor.

The 5 direction-shapes in this project (``above``, ``below``, ``dip_to``,
``hit_high``, ``range_low/high``) all encode the same latent risk-neutral
distribution of an underlying at a future date. This module unifies them
into a single survival function S(K) = P(X_T > K), applies the
call-curve second-derivative differencing to get a PMF on the strike grid, fits a
log-normal to recover (μ, σ_T), and annualises σ_T to σ_annual.

We deliberately reuse — never re-implement — the helpers proven by the
single-asset ladder in :mod:`pfm.vol_surface_pm`:
``_enforce_monotone``, ``_empirical_moments``, ``_fit_lognormal``,
``_market_yes_prob``, ``_safe_float``.
"""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

import numpy as np
from pydantic import BaseModel, Field
from scipy.optimize import minimize
from scipy.stats import norm

from pfm.cache_utils import get_cache
from pfm.vol_surface_pm import (
    _empirical_moments,
    _enforce_monotone,
    _fit_lognormal,
    _market_yes_prob,
    _safe_float,
)

logger = logging.getLogger(__name__)

_IV_CACHE = get_cache("pm_iv_extractor", ttl=600)

Direction = Literal["above", "below", "dip_to", "hit_high", "range_low", "range_high"]
Venue = Literal["polymarket", "kalshi"]
AssetClass = Literal["equity_index", "crypto", "commodity_energy", "commodity_metal"]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


class LadderEntry(BaseModel):
    slug: str
    strike: float
    direction: Direction
    venue: Venue
    market_value: float | None = None


class LadderFamily(BaseModel):
    asset: str
    asset_class: AssetClass
    maturity_utc: datetime
    spot_at_lookup: float
    entries: list[LadderEntry]


class PMIVResult(BaseModel):
    asset: str
    maturity_utc: datetime
    time_to_maturity_years: float
    sigma_annual: float = Field(..., ge=0.0)
    sigma_method: Literal[
        "lognormal_fit",
        "lognormal_survival_fit",
        "empirical_pmf",
        "single_strike_inverse",
    ]
    sigma_ci_low: float | None
    sigma_ci_high: float | None
    n_strikes: int = Field(..., ge=1)
    fitted_mean: float
    fitted_std: float
    implied_skew: float
    implied_kurtosis: float
    raw_strikes: list[float]
    raw_probs: list[float]
    warnings: list[str]


# ---------------------------------------------------------------------------
# LADDER_REGISTRY — hardcoded multi-direction ladder families
# ---------------------------------------------------------------------------
# Stored slugs match KNOWN_LADDERS in vol_surface_pm.py for the "above"
# direction (so the SPX/BTC/ETH ladders work end-to-end with the same Gamma
# mocks). Barrier ladders (dip_to/hit_high) come from factors.yml exactly as
# discovered in recon — full slugs with their resolution-suffix included.

LadderSpec = tuple[str, float, Direction, Venue]
# (slug, strike, direction, venue)

LADDER_REGISTRY: dict[str, dict[str, Any]] = {
    # SPX: no live multi-strike ladder on Polymarket as of 2026-05-15. The
    # 2026-05-16 discovery sweep (2000 active markets, sorted by 24h CLOB
    # volume) found only an SPY hit-high / hit-low pair for May 2026 plus a
    # single annual best-performance comparison — none of which form a
    # >=3-strike same-direction ladder. Re-check periodically; if Polymarket
    # relists S&P 500 strike binaries, add a new entry here with verified
    # slugs.
    "BTC": {
        "asset_class": "crypto",
        "families": [
            # EOY-2026 (resolves 2027-01-01) above-ladder — discovered live on
            # 2026-05-15. Eleven strikes between 90k and 1M cover the
            # in-the-money zone (~0.65 for 90k) down to the deep-OTM tail
            # (~0.015 for 1M). The mid-strikes (100k–190k) were sourced from
            # factors.yml and re-verified live on 2026-05-15. This is the
            # highest-density above-ladder live on Polymarket for BTC EOY-2026.
            {
                "maturity_utc": datetime(2027, 1, 1, 0, 0, tzinfo=UTC),
                "entries": [
                    (
                        "will-bitcoin-reach-90000-by-december-31-2026-113-862-581",
                        90_000.0,
                        "above",
                        "polymarket",
                    ),
                    (
                        "will-bitcoin-reach-100000-by-december-31-2026-571-361-361",
                        100_000.0,
                        "above",
                        "polymarket",
                    ),
                    (
                        "will-bitcoin-reach-140000-by-december-31-2026-131-829-299",
                        140_000.0,
                        "above",
                        "polymarket",
                    ),
                    (
                        "will-bitcoin-reach-150000-by-december-31-2026-557-246-971",
                        150_000.0,
                        "above",
                        "polymarket",
                    ),
                    (
                        "will-bitcoin-reach-160000-by-december-31-2026-934-934-164",
                        160_000.0,
                        "above",
                        "polymarket",
                    ),
                    (
                        "will-bitcoin-reach-190000-by-december-31-2026-936-485-627",
                        190_000.0,
                        "above",
                        "polymarket",
                    ),
                    (
                        "will-bitcoin-reach-200000-by-december-31-2026-752-232-389",
                        200_000.0,
                        "above",
                        "polymarket",
                    ),
                    (
                        "will-bitcoin-reach-250000-by-december-31-2026-579-442",
                        250_000.0,
                        "above",
                        "polymarket",
                    ),
                    (
                        "will-bitcoin-reach-500000-by-december-31-2026-864",
                        500_000.0,
                        "above",
                        "polymarket",
                    ),
                    (
                        "will-bitcoin-reach-1000000-by-december-31-2026-946",
                        1_000_000.0,
                        "above",
                        "polymarket",
                    ),
                ],
            },
            # EOY-2026 (resolves 2027-01-01) dip_to ladder — already validated
            # in prior A4 work; all 5 slugs re-verified live on 2026-05-15.
            {
                "maturity_utc": datetime(2027, 1, 1, 0, 0, tzinfo=UTC),
                "entries": [
                    (
                        "will-bitcoin-dip-to-15000-by-december-31-2026-416-954-417-853-"
                        "885-363-335-458-585-275-615-269-479-379-516-218-918-374-598",
                        15_000.0,
                        "dip_to",
                        "polymarket",
                    ),
                    (
                        "will-bitcoin-dip-to-25000-by-december-31-2026-948-243-253-666-"
                        "115-787-981-282-573-719-186-417-762-754-486-851-278-145",
                        25_000.0,
                        "dip_to",
                        "polymarket",
                    ),
                    (
                        "will-bitcoin-dip-to-35000-by-december-31-2026-744-877-748-219-"
                        "467-465-646-211-122-947-537-552-555-361-972-954-635-887",
                        35_000.0,
                        "dip_to",
                        "polymarket",
                    ),
                    (
                        "will-bitcoin-dip-to-45000-by-december-31-2026-674-923-755-971-"
                        "998-525-926-245-316-517-544-589-965-923-986-841-815-224",
                        45_000.0,
                        "dip_to",
                        "polymarket",
                    ),
                    (
                        "will-bitcoin-dip-to-55000-by-december-31-2026-527-627-868-745-"
                        "188-361-314-673-612-946-821-624-855-557-684-381",
                        55_000.0,
                        "dip_to",
                        "polymarket",
                    ),
                ],
            },
        ],
    },
    "ETH": {
        "asset_class": "crypto",
        "families": [
            # Short-dated 2026-06-01 above-ladder — discovered live on
            # 2026-05-15. No EOY-2026 ETH above-ladder is live on Polymarket;
            # only the monthly "reach $X in May 2026" markets form a usable
            # >=3-strike ladder. T_to_maturity is ~16 days at discovery —
            # annualised σ extraction will be noisier than the BTC EOY case.
            {
                "maturity_utc": datetime(2026, 6, 1, 0, 0, tzinfo=UTC),
                "entries": [
                    ("will-ethereum-reach-2600-in-may-2026", 2_600.0, "above", "polymarket"),
                    ("will-ethereum-reach-2800-in-may-2026", 2_800.0, "above", "polymarket"),
                    ("will-ethereum-reach-3600-in-may-2026", 3_600.0, "above", "polymarket"),
                    ("will-ethereum-reach-4000-in-may-2026", 4_000.0, "above", "polymarket"),
                    ("will-ethereum-reach-5000-in-may-2026", 5_000.0, "above", "polymarket"),
                ],
            },
            # Short-dated 2026-06-01 dip_to ladder — companion to above.
            {
                "maturity_utc": datetime(2026, 6, 1, 0, 0, tzinfo=UTC),
                "entries": [
                    ("will-ethereum-dip-to-600-in-may-2026", 600.0, "dip_to", "polymarket"),
                    ("will-ethereum-dip-to-1600-in-may-2026", 1_600.0, "dip_to", "polymarket"),
                    ("will-ethereum-dip-to-1800-in-may-2026", 1_800.0, "dip_to", "polymarket"),
                    ("will-ethereum-dip-to-2000-in-may-2026", 2_000.0, "dip_to", "polymarket"),
                    ("will-ethereum-dip-to-2200-in-may-2026", 2_200.0, "dip_to", "polymarket"),
                ],
            },
        ],
    },
    "WTI": {
        "asset_class": "commodity_energy",
        "families": [
            {
                "maturity_utc": datetime(2026, 6, 30, 23, 59, tzinfo=UTC),
                "entries": [
                    ("cl-above-50-jun-2026", 50.0, "above", "polymarket"),
                    ("cl-above-75-jun-2026", 75.0, "above", "polymarket"),
                    ("cl-above-90-jun-2026", 90.0, "above", "polymarket"),
                ],
            },
            {
                "maturity_utc": datetime(2026, 6, 30, 23, 59, tzinfo=UTC),
                "entries": [
                    (
                        "will-crude-oil-cl-hit-high-115-by-end-of-june-217-913-468-473",
                        115.0,
                        "hit_high",
                        "polymarket",
                    ),
                    (
                        "will-crude-oil-cl-hit-high-140-by-end-of-june-828-295-574-155",
                        140.0,
                        "hit_high",
                        "polymarket",
                    ),
                    (
                        "will-crude-oil-cl-hit-high-150-by-end-of-june-788-691",
                        150.0,
                        "hit_high",
                        "polymarket",
                    ),
                    (
                        "will-crude-oil-cl-hit-high-175-by-end-of-june-456-295",
                        175.0,
                        "hit_high",
                        "polymarket",
                    ),
                    (
                        "will-crude-oil-cl-hit-high-200-by-end-of-june-677",
                        200.0,
                        "hit_high",
                        "polymarket",
                    ),
                ],
            },
        ],
    },
    "GOLD": {
        "asset_class": "commodity_metal",
        "families": [
            {
                "maturity_utc": datetime(2026, 6, 30, 23, 59, tzinfo=UTC),
                "entries": [
                    (
                        "gc-hit-5500-high-jun-2026-424-457-376-356",
                        5_500.0,
                        "hit_high",
                        "polymarket",
                    ),
                    (
                        "gc-hit-6000-high-jun-2026-148-914-853",
                        6_000.0,
                        "hit_high",
                        "polymarket",
                    ),
                    (
                        "gc-hit-7000-high-jun-2026-433-244-291",
                        7_000.0,
                        "hit_high",
                        "polymarket",
                    ),
                    (
                        "gc-hit-8000-high-jun-2026-342-647-753",
                        8_000.0,
                        "hit_high",
                        "polymarket",
                    ),
                    (
                        "will-gold-gc-hit-high-8500-by-end-of-june-241-524",
                        8_500.0,
                        "hit_high",
                        "polymarket",
                    ),
                    (
                        "will-gold-gc-hit-high-10000-by-end-of-june",
                        10_000.0,
                        "hit_high",
                        "polymarket",
                    ),
                ],
            },
        ],
    },
}

_ASSET_DEFAULT_SPOT: dict[str, float] = {
    # SPX intentionally omitted — no live SPX ladder on Polymarket (see
    # LADDER_REGISTRY comment above). Keep as a no-op map entry only if a
    # future ladder is re-added.
    "BTC": 110_000.0,
    "ETH": 2_300.0,  # ~current ETH spot at 2026-05-15 discovery
    "WTI": 75.0,
    "GOLD": 4_000.0,
}


# ---------------------------------------------------------------------------
# Core math — direction-shape unification
# ---------------------------------------------------------------------------


def _survival_from_above(strikes: list[float], probs: list[float]) -> list[float]:
    """``above`` already encodes P(X_T > K) directly; just clip into [0, 1]."""
    return [float(np.clip(p, 0.0, 1.0)) for p in probs]


def _survival_from_below(strikes: list[float], probs: list[float]) -> list[float]:
    """``below`` is P(X_T < K) — flip to S(K) = 1 - P(X_T < K)."""
    return [float(np.clip(1.0 - p, 0.0, 1.0)) for p in probs]


def _survival_from_dip_to(strikes: list[float], probs: list[float]) -> list[float]:
    """One-touch barrier inversion (central 0.5 factor).

    For GBM-like paths, P(min_t X_t <= K) ≈ 2·P(X_T < K). Central estimate:
    P(X_T < K) ≈ 0.5·P(touch). So S(K) = 1 - 0.5·P(touch).
    """
    return [float(np.clip(1.0 - 0.5 * p, 0.0, 1.0)) for p in probs]


def _survival_from_hit_high(strikes: list[float], probs: list[float]) -> list[float]:
    """Mirror of dip_to: P(max_t X_t >= K) ≈ 2·P(X_T > K) → S(K) ≈ 0.5·P(hit)."""
    return [float(np.clip(0.5 * p, 0.0, 1.0)) for p in probs]


def _survival_from_range(
    strikes: list[float],
    probs: list[float],
    direction: Direction,
) -> list[float]:
    """Convert overlapping/adjacent range buckets to a survival fn.

    Strike of a range bucket is the lower-edge for ``range_low`` and the
    upper-edge for ``range_high``. PMF mass at K is interpreted as
    P(K_i < X_T <= K_i+w) for some band width. We accumulate from the
    right: S(K_i) = sum_{j: K_j >= K_i} p_j.
    """
    sorted_idx = sorted(range(len(strikes)), key=lambda i: strikes[i])
    surv: list[float] = []
    for i in sorted_idx:
        # mass from this bucket and all to the right is interpreted as
        # P(X_T > strike[i] - epsilon).
        right_mass = sum(max(0.0, float(probs[j])) for j in sorted_idx if strikes[j] >= strikes[i])
        surv.append(float(np.clip(right_mass, 0.0, 1.0)))
    # Re-order to match strikes' original order, then drop ordering;
    # caller will sort. We return in sorted order.
    _ = direction  # both range_low and range_high handled identically
    return surv


def build_survival_function(family: LadderFamily) -> tuple[list[float], list[float]]:
    """Convert any direction-shape ladder into a monotone P(X_T > K) survival fn.

    Per-direction handling matches the docstring on the public API: identity
    for ``above``, flip for ``below``, central one-touch inversion for
    ``dip_to`` / ``hit_high``, and right-cumulative for range buckets. The
    result is force-monotonised via ``_enforce_monotone`` so downstream
    differencing yields non-negative PMF mass.
    """
    pairs: list[tuple[float, float, Direction]] = []
    for entry in family.entries:
        if entry.market_value is None:
            continue
        mv = _safe_float(entry.market_value)
        if mv is None:
            continue
        pairs.append((float(entry.strike), float(mv), entry.direction))
    if not pairs:
        return [], []

    # Range directions are handled atomically because they require all
    # buckets together to accumulate. Other directions are per-strike.
    range_entries = [(k, p) for k, p, d in pairs if d in ("range_low", "range_high")]
    other_entries = [(k, p, d) for k, p, d in pairs if d not in ("range_low", "range_high")]

    transformed: list[tuple[float, float]] = []
    for k, p, d in other_entries:
        if d == "above":
            s = _survival_from_above([k], [p])[0]
        elif d == "below":
            s = _survival_from_below([k], [p])[0]
        elif d == "dip_to":
            s = _survival_from_dip_to([k], [p])[0]
        elif d == "hit_high":
            s = _survival_from_hit_high([k], [p])[0]
        else:  # defensive — Literal guarantees coverage
            continue
        transformed.append((k, s))

    if range_entries:
        rk = [k for k, _ in range_entries]
        rp = [p for _, p in range_entries]
        rs = _survival_from_range(rk, rp, "range_low")
        rk_sorted = sorted(rk)
        for k, s in zip(rk_sorted, rs, strict=True):
            transformed.append((k, s))

    transformed.sort(key=lambda r: r[0])
    strikes = [k for k, _ in transformed]
    probs = [s for _, s in transformed]
    monotone = _enforce_monotone(strikes, probs)
    return strikes, monotone


# ---------------------------------------------------------------------------
# σ extraction
# ---------------------------------------------------------------------------


def _fit_lognormal_survival(
    strikes: list[float],
    survival_probs: list[float],
    weights: list[float] | None = None,
) -> tuple[float, float, float]:
    """Directly fit a lognormal survival function to observed (K, S(K)) pairs.

    Solves the weighted least-squares problem

        minimize over (μ, σ_T):
            Σ_i w_i · (S_i - (1 - Φ((ln K_i - μ) / σ_T)))^2

    over the strictly observed strikes — *without* extrapolating to lower /
    upper tails. This avoids the systematic over-statement of σ that the
    moment-match pipeline incurs on wide ladders (it treats all mass past
    K_max as a point at 1.25·K_max).

    Args:
        strikes: Strikes K_i (positive, strictly increasing recommended).
        survival_probs: Observed S_i = P(X_T > K_i) ∈ [0, 1].
        weights: Optional w_i. Default = 1 / max(S_i·(1-S_i), 0.01) — gives
            near-50% strikes the highest weight (lowest binomial variance on
            a noisy PM midpoint) and 1pp / 99pp strikes a floor.

    Returns:
        (mu, sigma_T, residual_norm). If optimisation fails (≤2 valid points,
        flat survival, divergent search, or non-positive σ), returns
        (NaN, NaN, NaN).
    """
    if len(strikes) != len(survival_probs):
        return float("nan"), float("nan"), float("nan")
    pts: list[tuple[float, float]] = []
    for k, s in zip(strikes, survival_probs, strict=True):
        if k is None or s is None or not math.isfinite(k) or not math.isfinite(s):
            continue
        if k <= 0:
            continue
        # Clip into (0, 1) for stability of the log-CDF.
        s_clipped = float(np.clip(s, 1e-6, 1.0 - 1e-6))
        pts.append((float(k), s_clipped))
    if len(pts) < 3:
        return float("nan"), float("nan"), float("nan")
    arr_k = np.asarray([k for k, _ in pts], dtype=float)
    arr_s = np.asarray([s for _, s in pts], dtype=float)

    # Reject totally flat survival (no information about σ).
    if float(np.std(arr_s)) < 1e-6:
        return float("nan"), float("nan"), float("nan")

    if weights is None:
        w = np.asarray(
            [1.0 / max(s * (1.0 - s), 0.01) for s in arr_s.tolist()],
            dtype=float,
        )
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape[0] != arr_k.shape[0]:
            w = np.ones_like(arr_k)
    w = w / float(w.sum() + 1e-12)

    log_k = np.log(arr_k)
    median_k = float(np.median(arr_k))
    mu_0 = math.log(max(median_k, 1e-9))
    sigma_0 = 0.3

    def objective(params: np.ndarray) -> float:
        mu, sigma_t = float(params[0]), float(params[1])
        if sigma_t <= 1e-6 or not math.isfinite(sigma_t) or not math.isfinite(mu):
            return 1e9
        z = (log_k - mu) / sigma_t
        # Survival of LogNormal: 1 - Φ(z)
        pred = 1.0 - norm.cdf(z)
        resid = arr_s - pred
        return float(np.sum(w * resid * resid))

    # Try L-BFGS-B with a positivity bound on σ_T; fall back to Nelder-Mead.
    best_mu = float("nan")
    best_sigma = float("nan")
    best_obj = float("inf")
    try:
        res_lbfgs = minimize(
            objective,
            x0=np.asarray([mu_0, sigma_0]),
            method="L-BFGS-B",
            bounds=[(mu_0 - 10.0, mu_0 + 10.0), (1e-4, 10.0)],
            options={"maxiter": 500, "ftol": 1e-12},
        )
        if res_lbfgs.success and math.isfinite(res_lbfgs.fun):
            best_obj = float(res_lbfgs.fun)
            best_mu = float(res_lbfgs.x[0])
            best_sigma = float(res_lbfgs.x[1])
    except Exception as exc:  # pragma: no cover - defensive
        logger.info("survival_fit L-BFGS-B failed: %s", exc)

    try:
        res_nm = minimize(
            objective,
            x0=np.asarray([mu_0, sigma_0]),
            method="Nelder-Mead",
            options={"maxiter": 2000, "xatol": 1e-7, "fatol": 1e-12},
        )
        if res_nm.success and math.isfinite(res_nm.fun) and float(res_nm.fun) < best_obj:
            best_obj = float(res_nm.fun)
            best_mu = float(res_nm.x[0])
            best_sigma = float(res_nm.x[1])
    except Exception as exc:  # pragma: no cover - defensive
        logger.info("survival_fit Nelder-Mead failed: %s", exc)

    if not math.isfinite(best_sigma) or best_sigma <= 1e-4 or best_sigma >= 10.0 - 1e-6:
        return float("nan"), float("nan"), float("nan")
    return best_mu, best_sigma, math.sqrt(max(best_obj, 0.0))


def _single_strike_inverse_sigma(
    strike: float,
    prob_above: float,
    spot: float,
    t_years: float,
) -> float:
    """Recover σ_annual from a single P(X_T > K) via Black-Scholes-like inversion.

    Under a log-normal GBM with no drift: P(X_T > K) = Φ((ln(spot/K))/(σ√T) + σ√T/2).
    Solve numerically for σ. Used as a last-resort when only one strike is usable.
    """
    if spot <= 0 or strike <= 0 or t_years <= 0:
        return 0.0
    p = float(np.clip(prob_above, 1e-6, 1.0 - 1e-6))
    log_m = math.log(spot / strike)
    # Bisect σ on (1e-4, 5.0).

    def f(sigma: float) -> float:
        sd = sigma * math.sqrt(t_years)
        if sd <= 1e-9:
            return 1.0 if log_m > 0 else 0.0
        return float(norm.cdf(log_m / sd + 0.5 * sd)) - p

    lo, hi = 1e-4, 5.0
    f_lo, f_hi = f(lo), f(hi)
    if f_lo * f_hi > 0:
        # Monotone in σ — pick whichever bound is closer to p
        return lo if abs(f_lo) < abs(f_hi) else hi
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        f_mid = f(mid)
        if abs(f_mid) < 1e-7:
            return mid
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)


def _bootstrap_sigma_ci(
    strikes: list[float],
    probs: list[float],
    t_years: float,
    n_boot: int = 1000,
    seed: int = 42,
    primary_method: Literal["lognormal_survival_fit", "lognormal_fit"] = ("lognormal_survival_fit"),
) -> tuple[float | None, float | None]:
    """Bootstrap a 95% CI on σ_annual by resampling (strike, prob) pairs.

    Each bootstrap sample re-perturbs probs with σ_p=0.02 noise (typical
    bid/ask half-spread on PM binary contracts) AND resamples the strike
    indices with replacement. Returns (low, high) at the 2.5/97.5 percentile,
    or (None, None) if fewer than 3 strikes (CI is uninformative).

    ``primary_method`` controls which σ extractor each bootstrap resample
    runs through. When the production path is the survival fit, the CI is
    bootstrapped through the same fit; on fallback it uses moment-match.
    """
    n = len(strikes)
    if n < 3 or t_years <= 0:
        return None, None
    rng = np.random.default_rng(seed)
    samples: list[float] = []
    arr_k = np.asarray(strikes, dtype=float)
    arr_p = np.asarray(probs, dtype=float)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        idx.sort()
        boot_k = arr_k[idx].tolist()
        noise = rng.normal(0.0, 0.02, size=n)
        boot_p = np.clip(arr_p[idx] + noise, 0.0, 1.0).tolist()
        # Enforce monotone after noise + dedup of identical strikes
        uniq_k: list[float] = []
        uniq_p: list[float] = []
        for k, p in zip(boot_k, boot_p, strict=True):
            if uniq_k and abs(k - uniq_k[-1]) < 1e-9:
                continue
            uniq_k.append(k)
            uniq_p.append(p)
        if len(uniq_k) < 2:
            continue
        uniq_p_m = _enforce_monotone(uniq_k, uniq_p)
        sigma_t = 0.0
        if primary_method == "lognormal_survival_fit" and len(uniq_k) >= 3:
            _, sigma_t_fit, _ = _fit_lognormal_survival(uniq_k, uniq_p_m)
            if math.isfinite(sigma_t_fit) and sigma_t_fit > 0:
                sigma_t = float(sigma_t_fit)
        if sigma_t <= 0:
            mom = _empirical_moments(uniq_k, uniq_p_m)
            _, sigma_t_mom = _fit_lognormal(mom["mean"], mom["std"])
            sigma_t = float(sigma_t_mom)
        if sigma_t > 0 and t_years > 0:
            samples.append(sigma_t / math.sqrt(t_years))
    if len(samples) < 20:
        return None, None
    lo = float(np.percentile(samples, 2.5))
    hi = float(np.percentile(samples, 97.5))
    return lo, hi


def fit_implied_sigma(family: LadderFamily) -> PMIVResult:
    """Ladder family → annualised σ + CIs + diagnostics.

    Pipeline:
        1. build_survival_function (handles all 5 direction shapes)
        2. Convert survival to empirical PMF on the strike grid
        3. _empirical_moments → mean, std, skew, kurtosis
        4. _fit_lognormal → (μ, σ_T)
        5. Annualise: σ_annual = σ_T / sqrt(T)
        6. Bootstrap CI(95) via _bootstrap_sigma_ci
        7. Emit warnings for short maturity / few strikes / monotonicity violations
    """
    warnings: list[str] = []
    now = datetime.now(tz=UTC)
    maturity = family.maturity_utc
    if maturity.tzinfo is None:
        maturity = maturity.replace(tzinfo=UTC)
    t_seconds = (maturity - now).total_seconds()
    t_years = max(t_seconds / (365.25 * 86_400.0), 1e-6)
    if t_years < 7.0 / 365.25:
        warnings.append("short_maturity")

    raw_pairs_pre = [
        (e.strike, e.market_value) for e in family.entries if e.market_value is not None
    ]
    raw_pairs_pre.sort(key=lambda r: r[0])
    raw_strikes_in = [k for k, _ in raw_pairs_pre]
    # Pre-monotone direct survival (only for "above" — used for monotonicity warning)
    direct_above = [
        (e.strike, e.market_value)
        for e in family.entries
        if e.market_value is not None and e.direction == "above"
    ]
    direct_above.sort(key=lambda r: r[0])
    if len(direct_above) >= 2:
        for i in range(len(direct_above) - 1):
            if (direct_above[i + 1][1] or 0.0) > (direct_above[i][1] or 0.0) + 1e-6:
                warnings.append("monotonicity_violated")
                break

    strikes, probs = build_survival_function(family)
    n = len(strikes)
    if n == 0:
        raise ValueError("no usable strikes after applying market_value filter")

    if n < 3:
        warnings.append("few_strikes")

    # --- σ extraction --------------------------------------------------------
    # Primary: direct lognormal survival-function least-squares fit on the
    # observed strikes. Avoids the tail-mass bias that inflates σ for wide
    # ladders (which treated everything past K_max as a point at 1.25·K_max
    # in the moment-match pipeline). Validated post-A4 on WTI/GOLD.
    sigma_method: Literal[
        "lognormal_fit",
        "lognormal_survival_fit",
        "empirical_pmf",
        "single_strike_inverse",
    ]
    sigma_t: float
    mom: dict[str, float]
    if n >= 3:
        mom = _empirical_moments(strikes, probs)
        _, sigma_t_surv, _ = _fit_lognormal_survival(strikes, probs)
        if math.isfinite(sigma_t_surv) and sigma_t_surv > 0:
            sigma_method = "lognormal_survival_fit"
            sigma_t = float(sigma_t_surv)
        else:
            _, sigma_t_mom = _fit_lognormal(mom["mean"], mom["std"])
            if sigma_t_mom > 0 and mom["std"] > 0:
                sigma_method = "lognormal_fit"
                sigma_t = float(sigma_t_mom)
            elif mom["mean"] > 0 and mom["std"] > 0:
                sigma_method = "empirical_pmf"
                sigma_t = float(mom["std"] / mom["mean"])
            else:
                sigma_method = "empirical_pmf"
                sigma_t = 0.0
    elif n == 2:
        mom = _empirical_moments(strikes, probs)
        _, sigma_t = _fit_lognormal(mom["mean"], mom["std"])
        if sigma_t > 0 and mom["std"] > 0:
            sigma_method = "lognormal_fit"
        else:
            sigma_method = "empirical_pmf"
            # Fall back to a heuristic: relative std of empirical PMF.
            if mom["mean"] > 0 and mom["std"] > 0:
                sigma_t = float(mom["std"] / mom["mean"])
            else:
                sigma_t = 0.0
    else:
        # Single strike → invert Black-Scholes-like CDF
        sigma_method = "single_strike_inverse"
        only_strike = strikes[0]
        only_prob = probs[0]
        spot = family.spot_at_lookup if family.spot_at_lookup > 0 else only_strike
        sigma_annual_direct = _single_strike_inverse_sigma(only_strike, only_prob, spot, t_years)
        mom = {
            "mean": spot,
            "std": sigma_annual_direct * spot * math.sqrt(t_years),
            "skew": 0.0,
            "kurtosis": 0.0,
        }
        sigma_t = sigma_annual_direct * math.sqrt(t_years)

    sigma_annual = float(sigma_t / math.sqrt(t_years)) if t_years > 0 else 0.0

    # --- bootstrap CI --------------------------------------------------------
    if sigma_method == "single_strike_inverse":
        ci_low, ci_high = None, None
    else:
        boot_primary: Literal["lognormal_survival_fit", "lognormal_fit"] = (
            "lognormal_survival_fit"
            if sigma_method == "lognormal_survival_fit"
            else "lognormal_fit"
        )
        ci_low, ci_high = _bootstrap_sigma_ci(strikes, probs, t_years, primary_method=boot_primary)

    return PMIVResult(
        asset=family.asset,
        maturity_utc=maturity,
        time_to_maturity_years=round(t_years, 6),
        sigma_annual=round(sigma_annual, 6),
        sigma_method=sigma_method,
        sigma_ci_low=round(ci_low, 6) if ci_low is not None else None,
        sigma_ci_high=round(ci_high, 6) if ci_high is not None else None,
        n_strikes=n,
        fitted_mean=round(mom["mean"], 6),
        fitted_std=round(mom["std"], 6),
        implied_skew=round(mom["skew"], 6),
        implied_kurtosis=round(mom["kurtosis"], 6),
        raw_strikes=raw_strikes_in,
        raw_probs=[float(p) for _, p in raw_pairs_pre],
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Discovery + network fetch
# ---------------------------------------------------------------------------


class _PolymarketClientLike(Protocol):
    """Minimal protocol so tests can stub without instantiating the real client."""

    def get_market_metadata(self, slug: str) -> Any: ...


def _fetch_midpoint_polymarket(
    client: _PolymarketClientLike,
    slug: str,
) -> float | None:
    """Best-effort midpoint fetch.

    PolymarketClient.get_market_metadata returns a MarketMetadata dataclass
    (no bestBid/bestAsk fields) so we also look for an optional ``market_dict``
    attribute (set by some test stubs) or treat the returned object itself as
    a dict-like fixture. If neither works, return None and let the warning
    pipeline handle it.
    """
    try:
        meta = client.get_market_metadata(slug)
    except Exception as exc:
        logger.info("pm_iv_extractor: metadata fetch failed for %s: %s", slug, exc)
        return None
    # Tests often stub a dict directly — handle that path first.
    if isinstance(meta, dict):
        return _market_yes_prob(meta)
    # Fallback: dataclass with optional .market_dict attribute (test convenience)
    market_dict = getattr(meta, "market_dict", None)
    if isinstance(market_dict, dict):
        return _market_yes_prob(market_dict)
    # Last resort: try bestBid/bestAsk on the object itself
    bb = _safe_float(getattr(meta, "bestBid", None))
    ba = _safe_float(getattr(meta, "bestAsk", None))
    if bb is not None and ba is not None:
        return (bb + ba) / 2.0
    return None


def discover_ladder_family(
    asset: str,
    *,
    polymarket_client: _PolymarketClientLike,
    kalshi_client: Any = None,
    maturity_filter: str | None = None,
) -> LadderFamily | None:
    """Look up a registered ladder family by asset, then fetch live midpoints.

    Returns ``None`` if the asset is unknown or no family matches the
    ``maturity_filter`` (ISO date prefix, e.g. ``"2026-12-31"``).
    """
    _ = kalshi_client  # reserved for future Kalshi-venue ladders
    asset_key = asset.upper().strip()
    if asset_key not in LADDER_REGISTRY:
        return None
    cfg = LADDER_REGISTRY[asset_key]
    families = cfg["families"]
    chosen: dict[str, Any] | None = None
    for fam in families:
        if maturity_filter is None:
            chosen = fam
            break
        mat = fam["maturity_utc"]
        if mat.isoformat().startswith(maturity_filter):
            chosen = fam
            break
    if chosen is None and maturity_filter is None and families:
        chosen = families[0]
    if chosen is None:
        return None

    entries: list[LadderEntry] = []
    for slug, strike, direction, venue in chosen["entries"]:
        mv: float | None = None
        if venue == "polymarket":
            mv = _fetch_midpoint_polymarket(polymarket_client, slug)
        entries.append(
            LadderEntry(
                slug=slug,
                strike=float(strike),
                direction=direction,
                venue=venue,
                market_value=mv,
            )
        )

    spot = _ASSET_DEFAULT_SPOT.get(asset_key, 0.0)
    return LadderFamily(
        asset=asset_key,
        asset_class=cfg["asset_class"],
        maturity_utc=chosen["maturity_utc"],
        spot_at_lookup=spot,
        entries=entries,
    )


__all__ = [
    "LADDER_REGISTRY",
    "LadderEntry",
    "LadderFamily",
    "PMIVResult",
    "build_survival_function",
    "discover_ladder_family",
    "fit_implied_sigma",
]
