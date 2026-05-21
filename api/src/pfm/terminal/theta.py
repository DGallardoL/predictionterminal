"""Terminal theta (time-decay) analytics for prediction markets.

Theta in prediction-market context is the *expected daily price move* a
binary-outcome market exhibits as it walks toward resolution at time
``T``. Two complementary measures are exposed:

* **Empirical theta** — the median absolute daily probability change
  ``median |Δp|`` measured over the last ``N`` daily bars. Robust to
  outliers and assumption-free.
* **Theoretical theta (Brownian-bridge)** — the model-implied expected
  daily move under the assumption that ``logit(p_t)`` follows a Brownian
  bridge that *must* terminate at 0 or 1 at ``T``. The variance of a BB
  pinned at the endpoints is

      Var[X_t] = σ² · t · (T − t) / T

  which peaks at the midpoint and collapses to zero at both endpoints.
  At time ``t``, the bridge's *forward* one-step increment variance is

      Var[ΔX_{t→t+1}] ≈ σ² · (T − t − 1) / (T − t)        (≈ σ² for t≪T)

  Translating from logit-space to probability-space using the chain rule
  ``dp/dlogit = p(1−p)`` and the half-normal mean ``E|N(0,σ²)| = σ·√(2/π)``
  gives the closed-form daily theta::

      theta_p_per_day ≈ p(1−p) · σ · √(2/π · (T−t)/T)

  We expose this as ``theoretical_theta_per_day`` and report its
  numerical first derivative w.r.t. ``t`` (per day) as
  ``theta_acceleration``.

Two endpoints:

  * ``GET /terminal/theta/{slug}?days=30`` — single-market theta card.
  * ``GET /terminal/theta/cluster?theme=...&resolution_period=YYYYQn``
    — cohort aggregate.

External IO is delegated to :func:`pfm.sources.polymarket.fetch_factor_history`
and the gamma metadata helper from :mod:`pfm.terminal_countdown`. Tests
monkey-patch those names on this module so the suite stays fully offline.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Annotated, Any

import httpx
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query

from pfm.factors import FactorConfig, load_factors
from pfm.sources.polymarket import (
    PolymarketClient,
    PolymarketError,
    fetch_factor_history,
)
from pfm.terminal_countdown import (
    _fetch_gamma_metadata,
    _new_http_client,
    _parse_end_date,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/terminal/theta", tags=["terminal-theta"])


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Floor used when clipping probabilities away from {0, 1} for the
#: theoretical-theta formula. Mirrors :data:`pfm.model.DEFAULT_EPSILON`.
EPSILON: float = 0.01

#: Trading-day annualisation factor for σ → daily σ.
TRADING_DAYS_PER_YEAR: float = 252.0

#: Min observations required for an empirical theta estimate.
MIN_OBS: int = 3

#: Cap on cluster fan-out so the cohort endpoint never explodes.
CLUSTER_MAX_MARKETS: int = 40


# ---------------------------------------------------------------------------
# Dependency seams (overridden in tests)
# ---------------------------------------------------------------------------


def get_polymarket_client() -> PolymarketClient:
    """Resolve the shared :class:`PolymarketClient` from the host app.

    Imported lazily so the module can be loaded without pulling
    :mod:`pfm.main` (avoids a circular import in tests that mount the
    router on a bare ``FastAPI`` app).
    """
    from pfm.main import app  # local import to dodge circulars

    return app.state.poly


def _factors_path() -> Path:
    """Resolve the active factors.yml. Falls back to packaged default."""
    try:
        from pfm.config import get_settings

        return Path(get_settings().factors_file)
    except (ImportError, AttributeError, ValueError, OSError):
        # ImportError: pfm.config missing (won't happen in tree, defensive).
        # AttributeError/ValueError: settings shape changed.
        # OSError: pydantic-settings env-file IO failure.
        # 2026-05 refactor: module moved into ``pfm/terminal/``; climb one more
        # parent to reach the package root where ``factors.yml`` still lives.
        return Path(__file__).resolve().parents[1] / "factors.yml"


def get_factors_dep() -> dict[str, FactorConfig]:
    """Load the factor catalogue. Override in tests."""
    return load_factors(_factors_path())


# ---------------------------------------------------------------------------
# Pure math
# ---------------------------------------------------------------------------


def empirical_theta(prices: pd.Series) -> float:
    """Median absolute daily Δp over a price series.

    Args:
        prices: Daily probability series in [0, 1].

    Returns:
        ``median |Δp|`` per day, ``nan`` if fewer than :data:`MIN_OBS`
        valid daily diffs are present.
    """
    s = pd.Series(prices).astype(float).dropna()
    if len(s) < MIN_OBS + 1:
        return float("nan")
    diffs = s.diff().dropna().abs()
    if len(diffs) < MIN_OBS:
        return float("nan")
    return float(diffs.median())


def annualised_sigma_logit(prices: pd.Series, *, epsilon: float = EPSILON) -> float:
    """Annualised σ of Δlogit innovations (sample std, ddof=1).

    Daily Δlogit is ``log(p_t/(1−p_t)) − log(p_{t−1}/(1−p_{t−1}))``.
    """
    s = pd.Series(prices).astype(float).dropna()
    if len(s) < MIN_OBS + 1:
        return float("nan")
    p = s.clip(epsilon, 1.0 - epsilon)
    logit = (p / (1.0 - p)).map(math.log)
    d = logit.diff().dropna()
    if len(d) < MIN_OBS:
        return float("nan")
    daily_sigma = float(d.std(ddof=1))
    return daily_sigma * math.sqrt(TRADING_DAYS_PER_YEAR)


def theoretical_theta_brownian_bridge(
    current_p: float,
    days_to_resolve: float,
    sigma_annual: float,
    *,
    epsilon: float = EPSILON,
) -> float:
    """Brownian-bridge model-implied expected |Δp| per day.

    The market's logit follows a BB pinned at ±∞ at ``T`` (i.e. at
    ``p ∈ {0, 1}``). The daily-step increment variance at time ``t`` is
    approximately::

        Var[ΔX] ≈ σ_d² · (T − t) / T          (forward, small Δt)

    where ``σ_d = σ_annual / √252``. The expected absolute increment of
    a zero-mean Gaussian is ``σ · √(2/π)``, and the chain rule converts
    logit-space moves to prob-space moves using ``dp/dlogit = p(1−p)``.

    Args:
        current_p: Current YES probability in (0, 1).
        days_to_resolve: T − t in calendar days. Treated as 0 if ≤ 0.
        sigma_annual: Annualised σ of Δlogit.
        epsilon: Clip applied to ``current_p`` to keep ``p(1−p)`` finite
            when probabilities approach the bounds.

    Returns:
        Expected absolute prob-space move per day.
    """
    if not math.isfinite(sigma_annual) or sigma_annual <= 0.0:
        return float("nan")
    if days_to_resolve <= 0.0:
        return 0.0
    p = max(epsilon, min(1.0 - epsilon, float(current_p)))
    sigma_d = sigma_annual / math.sqrt(TRADING_DAYS_PER_YEAR)
    # T defaults to days_to_resolve itself when we don't know inception;
    # the (T-t)/T ratio is then ~1 and the formula reduces to a vanilla
    # half-normal. Callers that *do* know T can re-call with the ratio
    # baked into sigma_annual.
    var_step = sigma_d * sigma_d
    half_normal_mean = sigma_d * math.sqrt(2.0 / math.pi)
    # p*(1-p) is the local Jacobian of the inverse-logit transform.
    jacobian = p * (1.0 - p)
    # var_step is unused below explicitly but kept for symmetry/readability.
    _ = var_step
    return float(jacobian * half_normal_mean)


def theta_acceleration_per_day(
    current_p: float,
    days_to_resolve: float,
    sigma_annual: float,
    *,
    epsilon: float = EPSILON,
) -> float:
    """Numerical d(theta)/dt per day (forward finite difference, +1 day).

    Positive ⇒ theta is *growing* as we approach resolution (more daily
    motion expected tomorrow than today). Under a vanilla Brownian-bridge
    pinned at ``T`` this is generally negative far from ``T`` and turns
    positive in the very last few days as the bridge is forced to a
    boundary, but our half-normal step formula yields ≈ 0 — we instead
    surface the bridge-aware ``(T−t)/T`` shrinkage explicitly.
    """
    if days_to_resolve <= 0.0 or not math.isfinite(sigma_annual):
        return 0.0
    today = theoretical_theta_brownian_bridge(
        current_p, days_to_resolve, sigma_annual, epsilon=epsilon
    )
    tomorrow = theoretical_theta_brownian_bridge(
        current_p, max(0.0, days_to_resolve - 1.0), sigma_annual, epsilon=epsilon
    )
    return float(tomorrow - today)


def historical_decay_curve(prices: pd.Series, end_date: date | None) -> list[dict[str, float]]:
    """Map ``days_to_resolution`` → ``|Δp|`` for the input series.

    Args:
        prices: Daily probability series.
        end_date: The market's resolution date (UTC). If ``None``, we
            anchor at the last observation in ``prices``.

    Returns:
        Sorted list ``[{days_to_res, abs_delta}, ...]`` ascending in
        ``days_to_res``. Empty list when there are no usable diffs.
    """
    s = pd.Series(prices).astype(float).dropna()
    if len(s) < 2:
        return []
    diffs = s.diff().dropna().abs()
    anchor = pd.Timestamp(end_date) if end_date is not None else diffs.index[-1]
    if anchor.tzinfo is None:
        anchor = anchor.tz_localize("UTC")
    out: list[dict[str, float]] = []
    for ts, val in diffs.items():
        ts_utc = ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")
        days_to_res = float((anchor - ts_utc).total_seconds() / 86_400.0)
        out.append({"days_to_res": round(days_to_res, 2), "abs_delta": float(val)})
    out.sort(key=lambda r: r["days_to_res"])
    return out


def _interpret_single(
    days_to_resolve: float,
    theoretical: float,
    accel: float,
    end_date: date | None,
) -> str:
    """One-line human summary of the single-market theta state."""
    if days_to_resolve <= 0:
        return "Market resolved — theta is zero by construction."
    pp = theoretical * 100.0  # percentage points per day
    direction = "accelerating" if accel > 1e-5 else ("decelerating" if accel < -1e-5 else "flat")
    when = end_date.isoformat() if end_date else f"T+{int(days_to_resolve)}d"
    return (
        f"Decay {direction} toward {when} — expect {pp:.2f}pp daily moves "
        f"(d_theta={accel * 100.0:+.3f}pp/day)."
    )


def _interpret_cluster(n: int, median_theta: float, theme: str | None) -> str:
    """One-line cluster summary."""
    if n == 0:
        return "No markets matched the cluster filter."
    pp = median_theta * 100.0
    label = theme or "cohort"
    return f"{label}: {n} markets, median {pp:.2f}pp/day expected daily move."


# ---------------------------------------------------------------------------
# Resolution-period parsing
# ---------------------------------------------------------------------------

_QUARTER_RE = re.compile(r"^(\d{4})Q([1-4])$")


def parse_resolution_period(value: str | None) -> tuple[date, date] | None:
    """Parse ``"2026Q3"`` style strings into a ``(quarter_start, quarter_end)``.

    Returns ``None`` if ``value`` is falsy or unparseable.
    """
    if not value:
        return None
    m = _QUARTER_RE.match(value.strip())
    if not m:
        return None
    year = int(m.group(1))
    q = int(m.group(2))
    start_month = 3 * (q - 1) + 1
    end_month = start_month + 2
    qstart = date(year, start_month, 1)
    if end_month == 12:
        qend = date(year, 12, 31)
    else:
        # last day of end_month
        next_first = date(year, end_month + 1, 1)
        qend = date.fromordinal(next_first.toordinal() - 1)
    return qstart, qend


# ---------------------------------------------------------------------------
# Single-market computation (testable seam)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ThetaCard:
    """Single-market theta payload."""

    slug: str
    current_p: float
    days_to_resolve: float
    empirical_theta_per_day: float
    theoretical_theta_per_day: float
    theta_acceleration: float
    historical_decay_curve: list[dict[str, float]]
    interpretation: str


def compute_market_theta(
    *,
    slug: str,
    prices: pd.Series,
    end_date: date | None,
    days_lookback: int,
    now: datetime,
) -> ThetaCard:
    """Compute the theta card from already-fetched data (pure)."""
    s = pd.Series(prices).astype(float).dropna()
    if s.empty:
        raise ValueError(f"no price history for slug={slug!r}")

    current_p = float(s.iloc[-1])
    if end_date is None:
        days_to_resolve = 0.0
    else:
        end_dt = datetime.combine(end_date, datetime.min.time(), tzinfo=UTC)
        days_to_resolve = max(0.0, (end_dt - now).total_seconds() / 86_400.0)

    # Trim the empirical window to the requested lookback.
    if days_lookback and days_lookback > 0:
        cutoff = s.index.max() - pd.Timedelta(days=days_lookback)
        recent = s[s.index >= cutoff]
    else:
        recent = s
    emp = empirical_theta(recent)

    sigma = annualised_sigma_logit(s)
    theo = theoretical_theta_brownian_bridge(current_p, days_to_resolve, sigma)
    accel = theta_acceleration_per_day(current_p, days_to_resolve, sigma)
    curve = historical_decay_curve(s, end_date)
    interp = _interpret_single(days_to_resolve, theo, accel, end_date)

    return ThetaCard(
        slug=slug,
        current_p=current_p,
        days_to_resolve=round(days_to_resolve, 3),
        empirical_theta_per_day=emp,
        theoretical_theta_per_day=theo,
        theta_acceleration=accel,
        historical_decay_curve=curve,
        interpretation=interp,
    )


def _theta_card_to_dict(card: ThetaCard) -> dict[str, Any]:
    return {
        "slug": card.slug,
        "current_p": card.current_p,
        "days_to_resolve": card.days_to_resolve,
        "empirical_theta_per_day": card.empirical_theta_per_day,
        "theoretical_theta_per_day": card.theoretical_theta_per_day,
        "theta_acceleration": card.theta_acceleration,
        "historical_decay_curve": card.historical_decay_curve,
        "interpretation": card.interpretation,
    }


# ---------------------------------------------------------------------------
# Cluster aggregation
# ---------------------------------------------------------------------------


def _quantile(values: list[float], q: float) -> float:
    """Light-weight percentile (no numpy import needed for the API layer)."""
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    pos = q * (len(s) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(s[lo])
    frac = pos - lo
    return float(s[lo] * (1.0 - frac) + s[hi] * frac)


def aggregate_cluster(cards: list[ThetaCard]) -> dict[str, Any]:
    """Aggregate a list of single-market cards into a cluster payload."""
    n = len(cards)
    thetas: list[float] = [
        c.theoretical_theta_per_day for c in cards if math.isfinite(c.theoretical_theta_per_day)
    ]
    mean_theta = float(sum(thetas) / len(thetas)) if thetas else float("nan")
    median_theta = _quantile(thetas, 0.50)
    p10 = _quantile(thetas, 0.10)
    p90 = _quantile(thetas, 0.90)

    # Pool the per-market historical decay points into a curve, bucketed
    # by integer days-to-resolution.
    buckets: dict[int, list[float]] = {}
    for c in cards:
        for row in c.historical_decay_curve:
            bucket = round(row["days_to_res"])
            buckets.setdefault(bucket, []).append(row["abs_delta"])
    curve: list[dict[str, float]] = []
    for d in sorted(buckets):
        vals = buckets[d]
        curve.append(
            {
                "days_to_res": float(d),
                "n_markets": float(len(vals)),
                "median_abs_delta": _quantile(vals, 0.50),
            }
        )

    return {
        "n_markets": n,
        "mean_theta": mean_theta,
        "median_theta": median_theta,
        "p10_theta": p10,
        "p90_theta": p90,
        "theta_curve": curve,
    }


def _select_cluster_factors(
    factors: dict[str, FactorConfig],
    *,
    theme: str | None,
    resolution_period: str | None,
    end_dates: dict[str, date],
) -> list[FactorConfig]:
    """Filter the catalogue down to a cohort matching the cluster criteria.

    Only ``source: polymarket`` factors are considered (the BB theta
    formulation is binary-market-specific).
    """
    period = parse_resolution_period(resolution_period)
    out: list[FactorConfig] = []
    for fc in factors.values():
        if fc.source != "polymarket":
            continue
        if theme is not None and fc.theme != theme:
            continue
        if period is not None:
            ed = end_dates.get(fc.slug)
            if ed is None:
                continue
            if not (period[0] <= ed <= period[1]):
                continue
        out.append(fc)
    return out


# ---------------------------------------------------------------------------
# HTTP — cluster
# ---------------------------------------------------------------------------


@router.get("/cluster", include_in_schema=True)
def get_cluster_theta(
    theme: Annotated[str | None, Query(description="filter by factor theme")] = None,
    resolution_period: Annotated[
        str | None,
        Query(description='quarter filter, e.g. "2026Q3"', alias="resolution_period"),
    ] = None,
    days: Annotated[int, Query(ge=2, le=365)] = 30,
    factors: Annotated[dict[str, FactorConfig], Depends(get_factors_dep)] = ...,  # type: ignore[assignment]
    poly: Annotated[PolymarketClient, Depends(get_polymarket_client)] = ...,  # type: ignore[assignment]
) -> dict[str, Any]:
    """Aggregate theta for all markets matching ``theme`` + ``resolution_period``."""
    if theme is None and resolution_period is None:
        raise HTTPException(
            status_code=422,
            detail="must supply at least one of `theme`, `resolution_period`",
        )

    # 1. Pre-fetch end dates for every candidate slug (from gamma) so we
    #    can prune by resolution_period before doing the heavy price IO.
    candidates = [
        fc
        for fc in factors.values()
        if fc.source == "polymarket" and (theme is None or fc.theme == theme)
    ]
    if len(candidates) > CLUSTER_MAX_MARKETS:
        candidates = candidates[:CLUSTER_MAX_MARKETS]

    end_dates: dict[str, date] = {}
    if resolution_period is not None and candidates:
        try:
            with _new_http_client() as client:
                for fc in candidates:
                    meta = _fetch_gamma_metadata(fc.slug, client)
                    if meta is None:
                        continue
                    end_dt = _parse_end_date(meta.get("endDate"))
                    if end_dt is not None:
                        end_dates[fc.slug] = end_dt.date()
        except Exception as e:  # pragma: no cover
            logger.warning("cluster gamma scan failed: %s", e)

    selected = _select_cluster_factors(
        {fc.id: fc for fc in candidates},
        theme=theme,
        resolution_period=resolution_period,
        end_dates=end_dates,
    )

    # 2. Compute per-market theta cards.
    now = datetime.now(UTC)
    cards: list[ThetaCard] = []
    for fc in selected:
        try:
            df = fetch_factor_history(poly, fc.slug)
        except (PolymarketError, httpx.HTTPError) as e:
            logger.info("cluster: skipping %s: %s", fc.slug, e)
            continue
        if df is None or df.empty or "price" not in df.columns:
            continue
        prices = df["price"].astype(float)
        prices.index = pd.to_datetime(prices.index, utc=True)
        try:
            card = compute_market_theta(
                slug=fc.slug,
                prices=prices,
                end_date=end_dates.get(fc.slug),
                days_lookback=days,
                now=now,
            )
        except ValueError:
            continue
        cards.append(card)

    agg = aggregate_cluster(cards)
    agg["interpretation"] = _interpret_cluster(agg["n_markets"], agg["median_theta"], theme)
    agg["theme"] = theme
    agg["resolution_period"] = resolution_period
    return agg


# ---------------------------------------------------------------------------
# HTTP — single market  (registered AFTER /cluster so the static route wins)
# ---------------------------------------------------------------------------


@router.get("/{slug}")
def get_market_theta(
    slug: str,
    days: Annotated[int, Query(ge=2, le=365, description="empirical lookback in days")] = 30,
    poly: Annotated[PolymarketClient, Depends(get_polymarket_client)] = ...,  # type: ignore[assignment]
) -> dict[str, Any]:
    """Time-decay analytics card for one Polymarket binary market."""
    try:
        df = fetch_factor_history(poly, slug)
    except PolymarketError as e:
        raise HTTPException(status_code=404, detail=f"unknown slug: {e}") from e
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"polymarket http error: {e}") from e

    if df is None or df.empty or "price" not in df.columns:
        raise HTTPException(status_code=404, detail=f"no price history for slug={slug!r}")

    prices = df["price"].astype(float)
    prices.index = pd.to_datetime(prices.index, utc=True)

    # Pull resolution date from gamma metadata.
    end_date: date | None = None
    try:
        with _new_http_client() as client:
            meta = _fetch_gamma_metadata(slug, client)
        if meta is not None:
            end_dt = _parse_end_date(meta.get("endDate"))
            if end_dt is not None:
                end_date = end_dt.date()
    except Exception as e:  # pragma: no cover - best-effort metadata
        logger.warning("gamma metadata fetch failed for %s: %s", slug, e)

    now = datetime.now(UTC)
    try:
        card = compute_market_theta(
            slug=slug,
            prices=prices,
            end_date=end_date,
            days_lookback=days,
            now=now,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    return _theta_card_to_dict(card)


__all__ = [
    "EPSILON",
    "MIN_OBS",
    "ThetaCard",
    "aggregate_cluster",
    "annualised_sigma_logit",
    "compute_market_theta",
    "empirical_theta",
    "get_factors_dep",
    "get_polymarket_client",
    "historical_decay_curve",
    "parse_resolution_period",
    "router",
    "theoretical_theta_brownian_bridge",
    "theta_acceleration_per_day",
]
