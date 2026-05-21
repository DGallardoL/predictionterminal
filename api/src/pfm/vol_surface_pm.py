"""Volatility surface from Polymarket strike-ladder odds.

For underlyings with PM markets that form a strike ladder
(``btc-price-above-100k-eoy-2026``, ``...-above-150k-eoy-2026``, …), the
contracts collectively imply a *risk-neutral* survival function

    S(K) = P(price_T > K).

Differencing the survival function gives a discrete probability mass
function on bins ``(K_i, K_{i+1}]``. From the PMF we extract:

  - ``fitted_mean`` and ``fitted_std`` (raw moments of the empirical
    PMF, with mid-bin representatives),
  - ``implied_skew`` and ``implied_kurtosis`` (third / fourth standardised
    moments, computed empirically — the log-normal fit only sets the
    scale, not these higher moments),
  - a log-normal best-fit ``μ, σ`` recovered analytically from the
    mean/variance of the empirical PMF.

Compared to options IV smiles, the PM ladder is *coarse* (5–10 strikes,
typically) but it does not require dealing with American-vs-European
exercise, dividend assumptions, or microstructure quirks. It is also
denominated directly in the underlying — which is exactly what an
allocator wants for tail-risk calibration.

Endpoints (mounted via the module's ``router``):

  - ``GET /vol-surface/pm/{slug_pattern}``
  - ``GET /vol-surface/compare?ticker=BTC&pm_pattern=...``
"""

from __future__ import annotations

import logging
import re
from typing import Any, Literal

import httpx
import numpy as np
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from pfm.cache_utils import get_cache
from pfm.terminal import fetch_gamma_market

logger = logging.getLogger(__name__)

GAMMA_URL: str = "https://gamma-api.polymarket.com"

_VOL_CACHE = get_cache("vol_surface_pm", ttl=600)

# A handful of well-known ladders we resolve at import time.
# The pattern key is what users pass on the URL; the value is an ordered
# list of slug, strike pairs.
#
# 2026-05-15 audit: the original `btc-above-Xk-eoy-2026`, `eth-above-Xk-eoy-2026`
# and `spx-above-X-eoy-2026` slugs all return "no market found" on Gamma. They
# were replaced by Polymarket with the long-form "will-bitcoin-reach-…-by-
# december-31-2026" / "will-ethereum-reach-…" patterns (BTC) and short-dated
# monthly markets (ETH). SPX has no live multi-strike ladder at all.
KNOWN_LADDERS: dict[str, list[tuple[str, float]]] = {
    # BTC EOY-2026 above-ladder — replaces the dead `btc-above-Xk-eoy-2026`
    # set with verified live slugs (discovery 2026-05-15). Ten strikes
    # spanning 90k–1M; mirrors the BTC family in
    # ``pfm.vol.pm_iv_extractor.LADDER_REGISTRY``.
    "btc-price-eoy-2026": [
        ("will-bitcoin-reach-90000-by-december-31-2026-113-862-581", 90_000.0),
        ("will-bitcoin-reach-100000-by-december-31-2026-571-361-361", 100_000.0),
        ("will-bitcoin-reach-140000-by-december-31-2026-131-829-299", 140_000.0),
        ("will-bitcoin-reach-150000-by-december-31-2026-557-246-971", 150_000.0),
        ("will-bitcoin-reach-160000-by-december-31-2026-934-934-164", 160_000.0),
        ("will-bitcoin-reach-190000-by-december-31-2026-936-485-627", 190_000.0),
        ("will-bitcoin-reach-200000-by-december-31-2026-752-232-389", 200_000.0),
        ("will-bitcoin-reach-250000-by-december-31-2026-579-442", 250_000.0),
        ("will-bitcoin-reach-500000-by-december-31-2026-864", 500_000.0),
        ("will-bitcoin-reach-1000000-by-december-31-2026-946", 1_000_000.0),
    ],
    # ETH short-dated "reach $X in May 2026" — no EOY-2026 ETH above-ladder
    # is live on Polymarket as of 2026-05-15, so this is the only option.
    "eth-price-may-2026": [
        ("will-ethereum-reach-2600-in-may-2026", 2_600.0),
        ("will-ethereum-reach-2800-in-may-2026", 2_800.0),
        ("will-ethereum-reach-3600-in-may-2026", 3_600.0),
        ("will-ethereum-reach-4000-in-may-2026", 4_000.0),
        ("will-ethereum-reach-5000-in-may-2026", 5_000.0),
    ],
    # SPX: no live multi-strike same-direction ladder on Polymarket as of
    # 2026-05-15. Re-check periodically.
}

_STRIKE_RE = re.compile(r"above-(\d+(?:\.\d+)?)(k|m)?", re.I)


def _safe_float(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return f


def _market_yes_prob(market: dict[str, Any]) -> float | None:
    bb = _safe_float(market.get("bestBid"))
    ba = _safe_float(market.get("bestAsk"))
    if bb is not None and ba is not None:
        return (bb + ba) / 2.0
    return _safe_float(market.get("lastTradePrice"))


def _parse_strike_from_slug(slug: str) -> float | None:
    """Heuristic: extract numeric strike from ``...above-100k-...`` style slugs."""
    m = _STRIKE_RE.search(slug)
    if not m:
        return None
    val = float(m.group(1))
    suffix = (m.group(2) or "").lower()
    if suffix == "k":
        val *= 1_000.0
    elif suffix == "m":
        val *= 1_000_000.0
    return val


def _resolve_ladder(slug_pattern: str) -> list[tuple[str, float]]:
    """Return an ordered ladder of ``(slug, strike)`` for a pattern."""
    if slug_pattern in KNOWN_LADDERS:
        return list(KNOWN_LADDERS[slug_pattern])
    raise KeyError(f"unknown ladder pattern {slug_pattern!r}")


def _enforce_monotone(strikes: list[float], probs: list[float]) -> list[float]:
    """Force P(price > K) to be non-increasing in K — a coherent survival fn."""
    cleaned: list[float] = []
    last = 1.0
    for p in probs:
        p = min(max(0.0, p), last)
        cleaned.append(p)
        last = p
    return cleaned


def _empirical_moments(strikes: list[float], probs: list[float]) -> dict[str, float]:
    """Compute raw, std, skew, kurtosis of the empirical PMF derived from the survival fn.

    PMF mass at bin (K_i, K_{i+1}] = probs[i] - probs[i+1]. We add an
    open lower tail (mass = 1 - probs[0]) centred at strikes[0] / 2 and
    an open upper tail (mass = probs[-1]) centred at strikes[-1] * 1.25.
    """
    if not strikes:
        return {"mean": 0.0, "std": 0.0, "skew": 0.0, "kurtosis": 0.0}

    bins: list[tuple[float, float]] = []  # (centre, mass)
    # Lower tail.
    bins.append((strikes[0] * 0.5, max(0.0, 1.0 - probs[0])))
    for i in range(len(strikes) - 1):
        centre = 0.5 * (strikes[i] + strikes[i + 1])
        mass = max(0.0, probs[i] - probs[i + 1])
        bins.append((centre, mass))
    # Upper tail.
    bins.append((strikes[-1] * 1.25, max(0.0, probs[-1])))

    centres = np.array([c for c, _ in bins], dtype=float)
    masses = np.array([m for _, m in bins], dtype=float)
    total = masses.sum()
    if total <= 0:
        return {"mean": 0.0, "std": 0.0, "skew": 0.0, "kurtosis": 0.0}
    masses = masses / total

    mean = float(np.sum(centres * masses))
    var = float(np.sum(masses * (centres - mean) ** 2))
    std = float(np.sqrt(max(var, 0.0)))
    if std < 1e-9:
        return {"mean": mean, "std": 0.0, "skew": 0.0, "kurtosis": 0.0}
    skew = float(np.sum(masses * ((centres - mean) / std) ** 3))
    kurt = float(np.sum(masses * ((centres - mean) / std) ** 4) - 3.0)
    return {"mean": mean, "std": std, "skew": skew, "kurtosis": kurt}


def _fit_lognormal(mean: float, std: float) -> tuple[float, float]:
    """Return ``(mu, sigma)`` of a log-normal whose first two moments match.

    For X ~ LogNormal(μ, σ²):
        E[X]   = exp(μ + σ²/2)
        Var[X] = (exp(σ²) - 1) · exp(2μ + σ²)

    ⇒ σ² = ln(1 + Var/E²), μ = ln(E) - σ²/2.
    """
    if mean <= 0 or std <= 0:
        return 0.0, 0.0
    var = std * std
    sigma_sq = float(np.log1p(var / (mean * mean)))
    sigma = float(np.sqrt(max(sigma_sq, 0.0)))
    mu = float(np.log(mean) - 0.5 * sigma_sq)
    return mu, sigma


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_implied_distribution(
    slug_pattern: str,
    market_value: float | None = None,
    *,
    http: httpx.Client | None = None,
    overrides: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a coherent risk-neutral PMF from a PM strike ladder.

    Args:
        slug_pattern: Key into :data:`KNOWN_LADDERS` (e.g. ``btc-price-eoy-2026``).
        market_value: Current spot. Used only to compute moneyness in the
            response — it does NOT enter the PMF estimation.
        http: Injectable httpx client (tests).
        overrides: ``slug -> market_dict`` map (tests).

    Returns:
        dict with ``strikes, implied_probs, fitted_mean, fitted_std,
        implied_skew, implied_kurtosis, fitted_distribution_type, lognormal_mu,
        lognormal_sigma, n_strikes, market_value``.
    """
    ladder = _resolve_ladder(slug_pattern)
    own_http = http is None
    http = http or httpx.Client(timeout=8.0)
    raw: list[tuple[float, float]] = []
    try:
        for slug, strike in ladder:
            m: dict[str, Any] | None = None
            if overrides is not None and slug in overrides:
                m = overrides[slug]
            else:
                try:
                    m = fetch_gamma_market(http, GAMMA_URL, slug)
                except (LookupError, httpx.HTTPError) as exc:
                    logger.info("vol_surface_pm: skipping %s: %s", slug, exc)
                    continue
            p = _market_yes_prob(m) if m is not None else None
            if p is None:
                continue
            raw.append((float(strike), float(np.clip(p, 0.0, 1.0))))
    finally:
        if own_http:
            http.close()

    if len(raw) < 2:
        raise ValueError(f"need at least 2 ladder rungs for a distribution; got {len(raw)}")

    raw.sort(key=lambda r: r[0])
    strikes = [s for s, _ in raw]
    probs = _enforce_monotone(strikes, [p for _, p in raw])

    mom = _empirical_moments(strikes, probs)
    mu, sigma = _fit_lognormal(mom["mean"], mom["std"])

    return {
        "slug_pattern": slug_pattern,
        "strikes": strikes,
        "implied_probs": [round(p, 6) for p in probs],
        "fitted_mean": round(mom["mean"], 4),
        "fitted_std": round(mom["std"], 4),
        "implied_skew": round(mom["skew"], 4),
        "implied_kurtosis": round(mom["kurtosis"], 4),
        "fitted_distribution_type": "lognormal",
        "lognormal_mu": round(mu, 6),
        "lognormal_sigma": round(sigma, 6),
        "n_strikes": len(strikes),
        "market_value": market_value,
    }


def compare_pm_vs_options_iv(
    ticker: str,
    current_price: float,
    pm_strikes_data: dict[str, Any],
    options_iv_annual: float | None = None,
) -> dict[str, Any]:
    """Compare PM-implied vol to a (possibly placeholder) listed-options IV.

    The PM "implied vol" is taken as ``lognormal_sigma`` from the fit;
    that is already the right-shape diffusion vol for the horizon
    embedded in the ladder. Without knowing the exact horizon (we only
    know "EOY 2026") we don't annualise; the spread is reported in raw σ
    units and the caller can divide by sqrt(T) themselves.

    ``options_iv_annual`` falls back to ``ticker``-specific RV*1.2 if not
    provided, keeping the demo self-contained.
    """
    pm_sigma = float(pm_strikes_data.get("lognormal_sigma") or 0.0)
    if options_iv_annual is None:
        # Same fallback table as earnings_whisper — kept here to avoid a
        # cross-module import cycle.
        rv = {
            "BTC": 0.65,
            "ETH": 0.78,
            "SPX": 0.18,
            "NVDA": 0.55,
            "TSLA": 0.65,
        }.get(ticker.upper(), 0.40)
        options_iv_annual = rv * 1.2

    spread = pm_sigma - float(options_iv_annual)
    direction: Literal["pm_richer", "options_richer", "flat"]
    if spread > 0.01:
        direction = "pm_richer"
    elif spread < -0.01:
        direction = "options_richer"
    else:
        direction = "flat"

    return {
        "ticker": ticker.upper(),
        "current_price": float(current_price),
        "pm_lognormal_sigma": round(pm_sigma, 6),
        "options_iv_annual": round(float(options_iv_annual), 6),
        "spread_sigma": round(spread, 6),
        "direction": direction,
        "pm_strikes": pm_strikes_data.get("strikes", []),
        "pm_implied_probs": pm_strikes_data.get("implied_probs", []),
    }


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class ImpliedDistribution(BaseModel):
    slug_pattern: str
    strikes: list[float]
    implied_probs: list[float]
    fitted_mean: float
    fitted_std: float = Field(..., ge=0.0)
    implied_skew: float
    implied_kurtosis: float
    fitted_distribution_type: Literal["lognormal", "empirical"]
    lognormal_mu: float
    lognormal_sigma: float = Field(..., ge=0.0)
    n_strikes: int = Field(..., ge=2)
    market_value: float | None = None


class CompareResponse(BaseModel):
    ticker: str
    current_price: float
    pm_lognormal_sigma: float = Field(..., ge=0.0)
    options_iv_annual: float = Field(..., ge=0.0)
    spread_sigma: float
    direction: Literal["pm_richer", "options_richer", "flat"]
    pm_strikes: list[float]
    pm_implied_probs: list[float]


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


router = APIRouter(prefix="/vol-surface", tags=["vol-surface"])


@router.get("/pm/{slug_pattern}", response_model=ImpliedDistribution)
def get_pm_distribution(
    slug_pattern: str,
    market_value: float | None = Query(default=None, ge=0.0),
) -> ImpliedDistribution:
    cache_key = ("dist", slug_pattern, market_value)
    cached = _VOL_CACHE.get(cache_key)
    if cached is not None:
        return ImpliedDistribution(**cached)

    try:
        payload = extract_implied_distribution(slug_pattern, market_value)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    _VOL_CACHE.set(cache_key, payload, ttl=600)
    return ImpliedDistribution(**payload)


@router.get("/compare", response_model=CompareResponse)
def compare(
    ticker: str = Query(..., min_length=1, max_length=12),
    pm_pattern: str = Query(...),
    current_price: float = Query(default=0.0, ge=0.0),
    options_iv_annual: float | None = Query(default=None, ge=0.0),
) -> CompareResponse:
    cache_key = ("compare", ticker.upper(), pm_pattern, current_price, options_iv_annual)
    cached = _VOL_CACHE.get(cache_key)
    if cached is not None:
        return CompareResponse(**cached)

    try:
        dist = extract_implied_distribution(pm_pattern, current_price or None)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    spot = current_price or float(dist.get("fitted_mean") or 0.0)
    payload = compare_pm_vs_options_iv(ticker, spot, dist, options_iv_annual)
    _VOL_CACHE.set(cache_key, payload, ttl=600)
    return CompareResponse(**payload)


__all__ = [
    "KNOWN_LADDERS",
    "CompareResponse",
    "ImpliedDistribution",
    "compare_pm_vs_options_iv",
    "extract_implied_distribution",
    "router",
]
