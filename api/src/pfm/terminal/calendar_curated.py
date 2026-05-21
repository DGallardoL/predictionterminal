"""Curated calendar-spread surface — the *intuitive* subset.

The mechanical companion endpoint (``/terminal/calendar-pair/{slug}``)
emits ANY two markets that share an event-token at different deadlines.
Most of those pairs are **noise**: sports games at different dates,
election-day markets with two decision deadlines, gold/SPX strike
ladders that look like calendars but really are different strikes.

This module hard-curates the small set of clusters where a constant-
hazard model genuinely applies, i.e. where the *same* underlying event
can occur at any time and the contracts only differ in how much time
they give the event to happen:

  1. Fed-decision ladders (cumulative *or* per-meeting Kalshi)
  2. Political tenure (Powell, Trump, Putin, Xi)
  3. Conflict / regime change (Russia-Ukraine, Iran)
  4. Crypto first-ATH calendars (BTC, ETH)

For each cluster the endpoint computes the implied hazard λ per leg::

    λ = -ln(1 - p) / T

and reports the dispersion of λ across deadlines plus a 3-state trade
signal:

  * ``FLATTEN_CURVE``  — front λ ≫ back λ → sell front, buy back
  * ``STEEPEN_CURVE``  — front λ ≪ back λ → buy front, sell back
  * ``HOLD``           — λ-curve is roughly flat (no trade)

Routing note: this module owns its :class:`fastapi.APIRouter` mirroring
``terminal_calendar_pair``, ``terminal_equity`` etc. ``main.py`` only
needs::

    from pfm.terminal_calendar_curated import router as terminal_calendar_curated_router
    app.include_router(terminal_calendar_curated_router)
"""

from __future__ import annotations

import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date, datetime
from typing import Annotated

import httpx
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from fastapi import Path as FPath
from pydantic import BaseModel, Field

from pfm.config import Settings, get_settings
from pfm.factors import FactorConfig, load_factors
from pfm.sources.polymarket import (
    PolymarketClient,
    PolymarketError,
    fetch_factor_history,
)

logger = logging.getLogger(__name__)


# --- constants --------------------------------------------------------------

# Reference "today" for deadline arithmetic. Kept aligned with
# ``terminal_calendar_pair`` so backtest replay and live behaviour agree.
DEFAULT_TODAY: str = "2026-05-02"

# λ-ratio thresholds that trigger a trade signal. Calibrated from the
# Strategy-24 backtest (``/tmp/strat28_calendar_revalid.json``):
#   * |log(λ_far/λ_near)| ≥ 0.5 was the in-sample profitable threshold;
#   * 1.5 / 0.67 ≈ exp(±0.405) — slightly wider so we don't fire on every
#     small kink. Equivalent to "front leg's hazard is at least 50% higher
#     (or lower) than the back leg's".
LAMBDA_RATIO_RICH: float = 1.5  # front rich
LAMBDA_RATIO_CHEAP: float = 1.0 / LAMBDA_RATIO_RICH  # = 0.6667

# Length of the historical λ-ratio time-series returned by the detail
# endpoint. 90 days is roughly one quarter — long enough to spot a regime
# shift without forcing every leg to have ≥ 90 bars of history.
HIST_DAYS: int = 90


# --- curated mapping --------------------------------------------------------
# Each cluster lists factors.yml *ids* (not slugs) in **chronological
# deadline order**. The ``deadline`` is the resolution date that drives
# the days-to-resolve denominator in the λ formula.
#
# IMPORTANT: only ids that exist in factors.yml may appear here. The
# audit at module-load time logs any missing entries (so a future YAML
# prune doesn't break the endpoint silently) and silently drops them.


@dataclass(frozen=True)
class _CuratedLeg:
    factor_id: str
    deadline: date


@dataclass(frozen=True)
class _CuratedCluster:
    cluster_id: str
    title: str
    theory: str
    legs: tuple[_CuratedLeg, ...]


_CURATED_CLUSTERS: tuple[_CuratedCluster, ...] = (
    # 1. Fed cuts — Polymarket cumulative-by-date ladder.
    _CuratedCluster(
        cluster_id="fed_cuts_meetings",
        title="Fed rate cuts cumulative ladder · 2026",
        theory=(
            "Constant-hazard arrival of the first FOMC cut; under a flat λ "
            "prior the mid- and end-year probabilities should align."
        ),
        legs=(
            _CuratedLeg("fed_rate_cut_by_june", date(2026, 6, 30)),
            _CuratedLeg("fed_rate_cut_by_december", date(2026, 12, 31)),
        ),
    ),
    # 2. Fed cuts — per-meeting 25bp Kalshi ladder. Each market resolves
    #    at its own FOMC date, not cumulative.
    _CuratedCluster(
        cluster_id="fed_25bp_per_meeting",
        title="Fed 25bp cut per FOMC meeting (Kalshi)",
        theory=(
            "If the market thinks 25bp is the modal action, the per-meeting "
            "probability should be roughly constant; spreads reveal which "
            "meeting the curve is over-/under-pricing."
        ),
        legs=(
            _CuratedLeg("k_fed_jun_cut25", date(2026, 6, 17)),
            _CuratedLeg("k_fed_jul_cut25", date(2026, 7, 29)),
            _CuratedLeg("k_fed_sep_cut25", date(2026, 9, 16)),
            _CuratedLeg("k_fed_dec_cut25", date(2026, 12, 16)),
        ),
    ),
    # 3. Powell tenure (Fed Chair / Fed Board). Same person, three
    #    deadlines — a textbook constant-hazard ladder.
    _CuratedCluster(
        cluster_id="powell_tenure",
        title="Powell out as Fed Chair / Board",
        theory=(
            "Hazard of Powell's exit should be ~constant during normal "
            "times; it should spike only around renomination / pressure "
            "windows."
        ),
        legs=(
            _CuratedLeg("powell_out_may", date(2026, 5, 14)),
            _CuratedLeg("jerome_powell_out_from_fed", date(2026, 5, 30)),
            _CuratedLeg("jerome_powell_out_from_fed_2", date(2026, 12, 31)),
        ),
    ),
    # 4. Trump tenure.
    _CuratedCluster(
        cluster_id="trump_tenure",
        title="Trump out as President",
        theory=(
            "Removal-of-sitting-president hazard. Constant-λ prior is "
            "appropriate; large dispersion would reflect concentrated "
            "near-term political risk (impeachment, health) priced into "
            "the front leg."
        ),
        legs=(
            _CuratedLeg("trump_out_jun30", date(2026, 6, 30)),
            _CuratedLeg("trump_out_2027", date(2026, 12, 31)),
        ),
    ),
    # 5. Putin tenure.
    _CuratedCluster(
        cluster_id="putin_tenure",
        title="Putin out as Russian president",
        theory=(
            "Russian leadership-exit hazard. Pairs naturally with the "
            "Russia-Ukraine ceasefire cluster: a coup / death / step-down "
            "shifts both."
        ),
        legs=(
            _CuratedLeg("putin_out_jun", date(2026, 6, 30)),
            _CuratedLeg("putin_out_2027", date(2026, 12, 31)),
        ),
    ),
    # 6. Xi tenure.
    _CuratedCluster(
        cluster_id="xi_tenure",
        title="Xi Jinping out",
        theory=(
            "China leadership tail-risk. Hazard should be very close to "
            "zero; any front-end rich-ness is a flag."
        ),
        legs=(
            _CuratedLeg("xi_jinping_out_by", date(2026, 6, 30)),
            _CuratedLeg("xi_out_2027", date(2026, 12, 31)),
        ),
    ),
    # 7. Russia-Ukraine ceasefire — cluster retired 2026-05-13: both legs
    #    (russia_x_ukraine_ceasefire_by, russia_ukraine_ceasefire) were
    #    delisted on Polymarket and pruned from factors.yml by the cleanup
    #    pass. Restore here when fresh ceasefire-window markets list.
    # 8. Iran regime change.
    _CuratedCluster(
        cluster_id="iran_regime",
        title="Iranian regime change",
        theory=(
            "Regime-fall hazard. The two deadlines bracket the same risk "
            "process; cross-deadline λ tells whether the front month is "
            "pricing a near-term escalation that the long leg ignores."
        ),
        legs=(
            _CuratedLeg("iran_regime_jun", date(2026, 6, 30)),
            _CuratedLeg("iran_regime_eoy", date(2026, 12, 31)),
        ),
    ),
    # 9. BTC first-ATH calendar.
    _CuratedCluster(
        cluster_id="btc_ath_calendar",
        title="Bitcoin all-time-high · 2026 ladder",
        theory=(
            "First-ATH arrival follows a constant-hazard prior in a "
            "trending bull regime. λ-dispersion is a clean read on "
            "risk-on conviction across the year."
        ),
        legs=(
            _CuratedLeg("btc_ath_jun", date(2026, 6, 30)),
            _CuratedLeg("bitcoin_all_time_high_by", date(2026, 9, 30)),
            _CuratedLeg("bitcoin_all_time_high_by_2", date(2026, 12, 31)),
        ),
    ),
    # 10. ETH first-ATH calendar.
    _CuratedCluster(
        cluster_id="eth_ath_calendar",
        title="Ethereum all-time-high · 2026 ladder",
        theory=(
            "ETH first-ATH twin of the BTC calendar. Useful as a cross-"
            "asset crypto-regime sanity-check."
        ),
        legs=(
            _CuratedLeg("ethereum_all_time_high_by", date(2026, 9, 30)),
            _CuratedLeg("eth_ath_eoy", date(2026, 12, 31)),
        ),
    ),
)


# --- schemas ----------------------------------------------------------------


class CuratedLeg(BaseModel):
    """One contract on a curated calendar."""

    factor_id: str
    slug: str
    name: str
    source: str
    deadline: str = Field(..., description="ISO-8601 resolution date.")
    days_to_resolve: int = Field(..., ge=0)
    current_p: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Latest mid; ``null`` when the upstream API failed.",
    )
    implied_lambda: float | None = Field(
        None,
        description="Constant-hazard rate λ = -ln(1 - p) / T per day.",
    )


class CuratedClusterSummary(BaseModel):
    """Summary row returned by ``GET /terminal/calendar-curated/clusters``."""

    cluster_id: str
    title: str
    theory: str
    legs: list[CuratedLeg]
    n_legs: int
    lambda_range: tuple[float | None, float | None] = Field(
        ..., description="(min λ, max λ) across legs. ``null`` if no leg priced."
    )
    lambda_std: float | None = Field(
        None, description="Sample std of λ across legs; ``null`` for n<2."
    )
    lambda_ratio_front_back: float | None = Field(
        None,
        description="λ_front / λ_back — > 1 means front-leg hazard exceeds back.",
    )
    trade_signal: str = Field(
        ...,
        description=("One of FLATTEN_CURVE | STEEPEN_CURVE | HOLD | INSUFFICIENT_DATA."),
    )
    intuition: str = Field(..., description="Plain-English read on the signal.")


class HistoricalRatioPoint(BaseModel):
    """One day on the historical λ-ratio time-series."""

    date: str
    lambda_front: float | None
    lambda_back: float | None
    ratio: float | None


class CuratedClusterDetail(CuratedClusterSummary):
    """Detail payload — adds the 90-day historical λ-ratio series."""

    historical_ratio: list[HistoricalRatioPoint]


# --- helpers ----------------------------------------------------------------


def _today() -> date:
    return datetime.fromisoformat(DEFAULT_TODAY).date()


def _implied_lambda(p: float | None, days: int) -> float | None:
    """Closed-form constant-hazard rate. Returns ``None`` for degenerate input.

    A ``None`` (rather than ``0.0``) sentinel keeps the trade-signal logic
    honest: a leg that we couldn't price must NOT be confused with a leg
    where the market priced p = 0 exactly.
    """
    if p is None or days <= 0 or p <= 0.0:
        return None
    p_clipped = min(p, 0.999_999)
    return -math.log(1.0 - p_clipped) / float(days)


def _classify(
    lambdas: list[float | None],
) -> tuple[str, str, float | None, float | None, float | None, float | None]:
    """Compute (signal, intuition, ratio_front_back, lam_min, lam_max, std).

    ``lambdas`` is in chronological-deadline order, so index 0 = front,
    index -1 = back.
    """
    priced = [x for x in lambdas if x is not None]
    if len(priced) < 2:
        return (
            "INSUFFICIENT_DATA",
            "Not enough priced legs to fit a hazard curve.",
            None,
            priced[0] if priced else None,
            priced[0] if priced else None,
            None,
        )

    lam_front = lambdas[0]
    lam_back = lambdas[-1]
    if lam_front is None or lam_back is None or lam_back == 0.0:
        # Front or back leg unpriced — fall back to range/std but flag HOLD.
        ratio = None
    else:
        ratio = lam_front / lam_back

    lam_min = min(priced)
    lam_max = max(priced)
    # Sample std (ddof=1) of the priced legs.
    n = len(priced)
    mean = sum(priced) / n
    var = sum((x - mean) ** 2 for x in priced) / (n - 1) if n >= 2 else 0.0
    std = math.sqrt(var)

    if ratio is None:
        signal = "HOLD"
        intuition = "Curve incomplete — holding pending fresh prints."
    elif ratio >= LAMBDA_RATIO_RICH:
        signal = "FLATTEN_CURVE"
        intuition = (
            "Front-month hazard is materially richer than the back leg — "
            "sell the front, buy the back to flatten the λ-curve."
        )
    elif ratio <= LAMBDA_RATIO_CHEAP:
        signal = "STEEPEN_CURVE"
        intuition = (
            "Front-month hazard is materially cheaper than the back leg — "
            "buy the front, sell the back to steepen the λ-curve."
        )
    else:
        signal = "HOLD"
        intuition = (
            "λ-curve is broadly flat across the surface; no calendar "
            "edge versus the constant-hazard prior."
        )

    return signal, intuition, ratio, lam_min, lam_max, std


# --- factor lookup ----------------------------------------------------------


def _load_factor_index() -> dict[str, FactorConfig]:
    """Read factors.yml and return id → FactorConfig (or empty on failure)."""
    settings = get_settings()
    try:
        return load_factors(settings.factors_file)
    except FileNotFoundError:
        logger.warning("factors.yml not found at %s", settings.factors_file)
        return {}
    except Exception as e:  # pragma: no cover — surfaced at import time
        logger.warning("failed to parse factors.yml: %s", e)
        return {}


def _resolve_legs(
    cluster: _CuratedCluster, factors: dict[str, FactorConfig]
) -> list[tuple[_CuratedLeg, FactorConfig]]:
    """Drop legs whose factor_id is missing from factors.yml."""
    out: list[tuple[_CuratedLeg, FactorConfig]] = []
    for leg in cluster.legs:
        cfg = factors.get(leg.factor_id)
        if cfg is None:
            logger.info(
                "curated cluster %s: factor %r missing from factors.yml; skipping",
                cluster.cluster_id,
                leg.factor_id,
            )
            continue
        out.append((leg, cfg))
    return out


# --- price fetchers ---------------------------------------------------------


def _fetch_history(
    poly: PolymarketClient,
    factor: FactorConfig,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Series:
    """Best-effort history fetch — empty Series on any failure.

    We deliberately swallow upstream errors here: a single dead factor
    must not 502 the whole curated-clusters endpoint. The caller looks
    at ``len(series) == 0`` to decide whether to treat the leg as
    "unpriced".
    """
    if factor.source != "polymarket":
        # Kalshi / chain factors aren't fetched in the POC curated path:
        # the constant-hazard math only needs *one* price per leg per day,
        # and the Polymarket fetch is already battle-tested. Kalshi legs
        # surface as ``current_p = None`` until a Kalshi shim is added.
        return pd.Series(dtype=float)
    try:
        df = fetch_factor_history(poly, factor.slug, start=start, end=end)
    except (PolymarketError, httpx.HTTPError) as e:
        logger.info("polymarket fetch failed for %s: %s", factor.slug, e)
        return pd.Series(dtype=float)
    if df is None or df.empty or "price" not in df.columns:
        return pd.Series(dtype=float)
    s = df["price"].astype(float)
    s.index = pd.to_datetime(s.index, utc=True).normalize()
    return s


def _latest_price(series: pd.Series) -> float | None:
    """Last non-NaN observation, or ``None``."""
    if series.empty:
        return None
    s = series.dropna()
    if s.empty:
        return None
    return float(s.iloc[-1])


def _build_summary_from_prices(
    cluster: _CuratedCluster,
    legs: list[tuple[_CuratedLeg, FactorConfig]],
    today: date,
    leg_prices: list[float | None],
) -> CuratedClusterSummary:
    """Assemble a summary payload given pre-fetched per-leg prices."""
    leg_models: list[CuratedLeg] = []
    lambdas: list[float | None] = []
    for (leg, cfg), price in zip(legs, leg_prices, strict=True):
        days = max(0, (leg.deadline - today).days)
        lam = _implied_lambda(price, days)
        lambdas.append(lam)
        leg_models.append(
            CuratedLeg(
                factor_id=leg.factor_id,
                slug=cfg.slug,
                name=cfg.name,
                source=cfg.source,
                deadline=leg.deadline.isoformat(),
                days_to_resolve=days,
                current_p=price,
                implied_lambda=lam,
            )
        )

    signal, intuition, ratio, lam_min, lam_max, std = _classify(lambdas)
    return CuratedClusterSummary(
        cluster_id=cluster.cluster_id,
        title=cluster.title,
        theory=cluster.theory,
        legs=leg_models,
        n_legs=len(leg_models),
        lambda_range=(lam_min, lam_max),
        lambda_std=std,
        lambda_ratio_front_back=ratio,
        trade_signal=signal,
        intuition=intuition,
    )


def _historical_ratio(
    legs: list[tuple[_CuratedLeg, FactorConfig]],
    series_by_id: dict[str, pd.Series],
    today: date,
) -> list[HistoricalRatioPoint]:
    """Daily λ_front / λ_back over the last :data:`HIST_DAYS` calendar days.

    Front leg = first leg in chronological order; back leg = last leg.
    For days where either leg is missing a print, the ratio is ``None``
    (the per-leg λ is still emitted so the frontend can render the two
    independent hazard time-series).
    """
    if len(legs) < 2:
        return []
    front_leg, _front_cfg = legs[0]
    back_leg, _back_cfg = legs[-1]
    front_series = series_by_id.get(front_leg.factor_id, pd.Series(dtype=float))
    back_series = series_by_id.get(back_leg.factor_id, pd.Series(dtype=float))

    # Build the daily index over the last HIST_DAYS calendar days
    # (including today). Use UTC-normalised timestamps to match the
    # series indexes.
    end_ts = pd.Timestamp(today, tz="UTC").normalize()
    start_ts = end_ts - pd.Timedelta(days=HIST_DAYS - 1)
    idx = pd.date_range(start_ts, end_ts, freq="D")

    # Forward-fill so a missing day uses the most recent observation
    # within the window — typical for low-volume markets.
    fs = front_series.reindex(idx).ffill()
    bs = back_series.reindex(idx).ffill()

    out: list[HistoricalRatioPoint] = []
    for ts in idx:
        d = ts.date()
        days_front = max(0, (front_leg.deadline - d).days)
        days_back = max(0, (back_leg.deadline - d).days)
        p_front = float(fs.loc[ts]) if pd.notna(fs.loc[ts]) else None
        p_back = float(bs.loc[ts]) if pd.notna(bs.loc[ts]) else None
        lam_front = _implied_lambda(p_front, days_front)
        lam_back = _implied_lambda(p_back, days_back)
        if lam_front is None or lam_back is None or lam_back == 0.0:
            ratio: float | None = None
        else:
            ratio = lam_front / lam_back
        out.append(
            HistoricalRatioPoint(
                date=d.isoformat(),
                lambda_front=lam_front,
                lambda_back=lam_back,
                ratio=ratio,
            )
        )
    return out


# --- module-level response cache --------------------------------------------
# /clusters fans out one Polymarket history fetch per leg across ~10 clusters
# (≈25 sequential HTTP calls). Both the upstream API and the curated cluster
# definitions are static within a 5-min horizon, so a short TTL cache turns
# warm latency from ~1s into <5ms. The detail endpoint reuses _fetch_history
# results indirectly via the same Polymarket-side fetch-history cache and
# Pandas series cache so per-cluster detail is already cheap.

_CLUSTERS_CACHE: dict[str, tuple[float, list[CuratedClusterSummary]]] = {}
_CLUSTERS_CACHE_TTL_SECONDS: float = 300.0  # 5 min
_CLUSTERS_MAX_WORKERS: int = 8


# --- router -----------------------------------------------------------------

router = APIRouter(prefix="/terminal", tags=["terminal-calendar-curated"])


def get_polymarket_client() -> PolymarketClient:
    """Resolve the shared :class:`PolymarketClient` from app state."""
    from pfm.main import app  # local import — avoids circular at module load

    return app.state.poly


@router.get("/calendar-curated/clusters")
def list_clusters(
    *,
    settings: Annotated[Settings, Depends(get_settings)],
    poly: Annotated[PolymarketClient, Depends(get_polymarket_client)],
) -> list[CuratedClusterSummary]:
    """Return one summary per curated cluster.

    Each cluster is reported even if some of its legs are unpriced
    (``current_p = null``); the trade signal degrades gracefully to
    ``HOLD`` / ``INSUFFICIENT_DATA``.

    Performance: per-leg Polymarket history fetches run in a bounded thread
    pool (was serial — ~25 sequential RTTs). A 5-minute response cache keyed
    on the day of computation makes warm hits ~constant time.
    """
    # --- fast path: cached response ---------------------------------------
    now_ts = time.time()
    cache_key = _today().isoformat()
    cached = _CLUSTERS_CACHE.get(cache_key)
    if cached is not None and (now_ts - cached[0]) < _CLUSTERS_CACHE_TTL_SECONDS:
        return cached[1]

    factors = _load_factor_index()
    today = _today()
    end_ts = pd.Timestamp(today, tz="UTC").normalize()
    # We only need the current mid for the summary endpoint, but reuse
    # the same window the detail endpoint uses so caches line up.
    start_ts = end_ts - pd.Timedelta(days=HIST_DAYS)

    # Resolve every leg up-front so we can fan-out fetches across all
    # clusters at once (rather than nesting cluster→leg loops).
    resolved: list[tuple[_CuratedCluster, list[tuple[_CuratedLeg, FactorConfig]]]] = []
    fetch_tasks: list[tuple[int, int, FactorConfig]] = []  # (cluster_idx, leg_idx, cfg)
    for cluster in _CURATED_CLUSTERS:
        legs = _resolve_legs(cluster, factors)
        if len(legs) < 2:
            continue
        cluster_idx = len(resolved)
        resolved.append((cluster, legs))
        for leg_idx, (_leg, cfg) in enumerate(legs):
            fetch_tasks.append((cluster_idx, leg_idx, cfg))

    # Bounded parallel fetch. Per-fetch failures already degrade to empty
    # series inside _fetch_history, so the worker never raises.
    series_grid: dict[tuple[int, int], pd.Series] = {}
    if fetch_tasks:
        max_workers = min(len(fetch_tasks), _CLUSTERS_MAX_WORKERS)
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="cal-curated") as ex:
            futures = {
                ex.submit(_fetch_history, poly, cfg, start_ts, end_ts): (ci, li)
                for ci, li, cfg in fetch_tasks
            }
            for fut, (ci, li) in futures.items():
                try:
                    series_grid[(ci, li)] = fut.result()
                except Exception as e:
                    # _fetch_history already swallows expected errors; this is
                    # purely defensive so a single bad future can't 500 the page.
                    logger.warning("calendar-curated fetch worker raised: %s", e)
                    series_grid[(ci, li)] = pd.Series(dtype=float)

    out: list[CuratedClusterSummary] = []
    for ci, (cluster, legs) in enumerate(resolved):
        prices: list[float | None] = []
        for li, (_leg, _cfg) in enumerate(legs):
            series = series_grid.get((ci, li), pd.Series(dtype=float))
            prices.append(_latest_price(series))
        out.append(_build_summary_from_prices(cluster, legs, today, prices))

    _CLUSTERS_CACHE[cache_key] = (now_ts, out)
    return out


@router.get("/calendar-curated/{cluster_id}")
def get_cluster(
    cluster_id: Annotated[str, FPath(min_length=1, max_length=80)],
    *,
    settings: Annotated[Settings, Depends(get_settings)],
    poly: Annotated[PolymarketClient, Depends(get_polymarket_client)],
) -> CuratedClusterDetail:
    """Detail payload for a single cluster, including 90-day λ-ratio.

    Raises HTTP 404 when ``cluster_id`` is not curated, and HTTP 422
    when the cluster lost too many legs to the YAML pruner to be useful.
    """
    cluster = next((c for c in _CURATED_CLUSTERS if c.cluster_id == cluster_id), None)
    if cluster is None:
        raise HTTPException(status_code=404, detail=f"unknown curated cluster {cluster_id!r}")

    factors = _load_factor_index()
    legs = _resolve_legs(cluster, factors)
    if len(legs) < 2:
        raise HTTPException(
            status_code=422,
            detail=(
                f"cluster {cluster_id!r} has fewer than 2 surviving legs in "
                "factors.yml — nothing to spread."
            ),
        )

    today = _today()
    end_ts = pd.Timestamp(today, tz="UTC").normalize()
    start_ts = end_ts - pd.Timedelta(days=HIST_DAYS)

    series_by_id: dict[str, pd.Series] = {}
    prices: list[float | None] = []
    for leg, cfg in legs:
        s = _fetch_history(poly, cfg, start_ts, end_ts)
        series_by_id[leg.factor_id] = s
        prices.append(_latest_price(s))

    summary = _build_summary_from_prices(cluster, legs, today, prices)
    history = _historical_ratio(legs, series_by_id, today)

    return CuratedClusterDetail(
        **summary.model_dump(),
        historical_ratio=history,
    )


# --- audit (run at import) --------------------------------------------------


def _audit() -> dict[str, list[str]]:
    """Cross-check curated factor_ids against factors.yml — diagnostic only.

    Returns a mapping ``{cluster_id → [missing_ids]}``. Logged at INFO.
    Tests use this to assert the curated table stays in sync with YAML.
    """
    factors = _load_factor_index()
    audit: dict[str, list[str]] = {}
    for cluster in _CURATED_CLUSTERS:
        missing = [leg.factor_id for leg in cluster.legs if leg.factor_id not in factors]
        audit[cluster.cluster_id] = missing
        if missing:
            logger.warning(
                "curated cluster %s missing ids in factors.yml: %s",
                cluster.cluster_id,
                missing,
            )
    return audit


def curated_factor_ids() -> dict[str, list[str]]:
    """Return ``{cluster_id → [factor_id, ...]}`` for tests / docs."""
    return {c.cluster_id: [leg.factor_id for leg in c.legs] for c in _CURATED_CLUSTERS}


# Run the audit once at import so YAML drift surfaces in the logs the
# moment uvicorn starts. Soft-fail: never raise from import.
try:
    _audit()
except Exception as e:  # pragma: no cover - defensive
    logger.warning("curated calendar audit raised: %s", e)


__all__ = [
    "CuratedClusterDetail",
    "CuratedClusterSummary",
    "CuratedLeg",
    "HistoricalRatioPoint",
    "curated_factor_ids",
    "router",
]
