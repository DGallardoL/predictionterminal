"""Real-time calendar-arbitrage scanner.

This module surfaces *currently-actionable* calendar-spread opportunities
across the curated calendar clusters that survived the wave-5 stress
tests. It is the live alpha-deployment surface for the only structural
strategy with a positive net Sharpe at the |log λ-ratio| ≥ 0.75
threshold (Sharpe net = +1.78 on the 51-pair revalidation set —
``/tmp/strat28_calendar_revalid.json``).

The math mirrors :mod:`pfm.terminal_calendar_pair`:

* Each leg implies a constant-hazard rate ``λ = -ln(1 - p) / T``.
* For two legs of the same event the *log λ-ratio* ``ln(λ_far / λ_near)``
  measures term-structure dispersion under the constant-hazard prior.
* When |log λ-ratio| ≥ 0.75 the cluster is **actionable**: the trader
  longs the leg with the *lower* λ (overpriced in hazard terms) and
  shorts the leg with the *higher* λ.

Trade direction
---------------

* ``log_ratio = ln(λ_far / λ_near) > 0``  ⇒ far is hot, near is cold
  ⇒ **STEEPEN_CURVE**: long near, short far (curve will steepen back).
* ``log_ratio < 0``                       ⇒ near is hot, far is cold
  ⇒ **FLATTEN_CURVE**: long far, short near (curve will flatten back).

Endpoints
---------

* ``GET /terminal/calendar-scanner/active`` — list of currently-actionable
  signals across all curated clusters.
* ``GET /terminal/calendar-scanner/historical?cluster_id=...`` — 90-day
  PnL backtest for a single cluster's pairwise signal.

Routing
-------

The router is owned by this module. ``main.py`` only needs to::

    from pfm.terminal_calendar_scanner import router as terminal_calendar_scanner_router
    app.include_router(terminal_calendar_scanner_router)

main.py and the frontend are intentionally **not** modified by this
module — both will be wired in a follow-up surfacing pass.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# --- constants --------------------------------------------------------------

#: Strategy-28 actionability threshold on |log λ-ratio|.
ACTIONABLE_THRESHOLD: float = 0.75
#: Conviction tier ceiling for "high".
HIGH_CONVICTION_THRESHOLD: float = 1.0
#: Lower bound for "low" conviction (anything below this is dropped).
LOW_CONVICTION_FLOOR: float = 0.50
#: Round-trip taker cost per trade (2 legs × 1.8% maker/taker).
RT_COST: float = 0.036
#: Sharpe-to-EV translation slope; calibrated so log-ratio ≈ 1 ⇒ ~4 % gross.
GROSS_EV_PER_LOG_UNIT: float = 0.04
#: Clip ε for the implied-hazard helper (mirrors terminal_calendar_pair).
P_CLIP: float = 0.999_999

# Hold window default — the curated clusters resolve over months but the
# strategy holds individual signals for a fraction of that.
DEFAULT_HOLD_WINDOW_DAYS: int = 110
#: Mean-reversion exit threshold on |log λ-ratio|.
EXIT_THRESHOLD: float = 0.30
#: Backtest hold period for the historical replay.
BACKTEST_HOLD_DAYS: int = 5
#: Backtest lookback in calendar days.
BACKTEST_LOOKBACK_DAYS: int = 90

# Fallback strat-28 file (used only when the curated module is missing).
_STRAT28_PATH: Path = Path("/tmp/strat28_calendar_revalid.json")


# --- schemas ----------------------------------------------------------------


class ScannerLeg(BaseModel):
    """One side of a calendar-arb trade."""

    slug: str = Field(..., description="Polymarket slug or synthetic id-slug.")
    name: str = Field(..., description="Human-readable contract title.")
    current_p: float = Field(..., ge=0.0, le=1.0)
    implied_lambda: float = Field(
        ..., description="Constant-hazard rate λ = -ln(1 - p) / T (per day)."
    )


class ActionableSignal(BaseModel):
    """A currently-executable calendar-arb opportunity."""

    cluster_id: str
    title: str
    trade_type: Literal["FLATTEN_CURVE", "STEEPEN_CURVE"]
    long_leg: ScannerLeg
    short_leg: ScannerLeg
    log_lambda_ratio: float = Field(
        ...,
        description="ln(λ_far / λ_near). |·| ≥ 0.75 to be actionable.",
    )
    expected_ev_pct: float = Field(..., description="Estimated net edge (%) after 3.6% RT cost.")
    hold_window_days: int = Field(..., ge=1)
    conviction: Literal["high", "medium", "low"]
    entry_signal: str
    exit_rule: str


class BacktestPoint(BaseModel):
    """One day on the cluster-level historical PnL curve."""

    date: str
    log_lambda_ratio: float
    in_trade: bool
    pnl_today: float
    cum_pnl: float


class HistoricalBacktest(BaseModel):
    """90-day cluster-level backtest."""

    cluster_id: str
    n_days: int
    n_trades: int
    cum_pnl: float
    sharpe: float
    points: list[BacktestPoint]


# --- math helpers -----------------------------------------------------------


def _implied_lambda(p: float, days: int) -> float:
    """Constant-hazard rate λ such that ``1 - exp(-λ T) = p``.

    Mirrors :func:`pfm.terminal_calendar_pair._implied_lambda` so the two
    surfaces stay numerically consistent.
    """
    if days <= 0 or p <= 0.0:
        return 0.0
    p_clipped = min(p, P_CLIP)
    return -math.log(1.0 - p_clipped) / float(days)


def _log_lambda_ratio(lam_near: float, lam_far: float) -> float:
    """``ln(λ_far / λ_near)`` with degenerate-input handling."""
    if lam_near <= 0.0 or lam_far <= 0.0:
        return 0.0
    return math.log(lam_far / lam_near)


def _classify_conviction(abs_log_ratio: float) -> Literal["high", "medium", "low"]:
    """Map |log λ-ratio| onto the conviction tiers."""
    if abs_log_ratio >= HIGH_CONVICTION_THRESHOLD:
        return "high"
    if abs_log_ratio >= ACTIONABLE_THRESHOLD:
        return "medium"
    return "low"


def _expected_ev_pct(abs_log_ratio: float) -> float:
    """Net EV estimate (in %) after the round-trip taker cost.

    ``gross = abs_log_ratio × 0.04`` is a deliberately conservative
    Sharpe→EV translation: at ``|log λ-ratio| ≈ 1`` the calibrated
    Strategy-28 cell shows ~4 % gross mean PnL per pair-month.
    """
    gross = abs_log_ratio * GROSS_EV_PER_LOG_UNIT
    return round((gross - RT_COST) * 100.0, 2)


# --- cluster ingestion ------------------------------------------------------


def _load_curated_clusters() -> list[dict[str, Any]]:
    """Pull curated calendar clusters.

    Tries the parallel-built :mod:`pfm.terminal_calendar_curated` module
    first (the production source of truth). Falls back to the on-disk
    strat-28 backtest fixture so the scanner remains useful while the
    curated module is still being authored.

    The contract for either source is::

        [
          {
            "cluster_id": str,
            "title": str,
            "legs": [
              {"slug": str, "name": str, "current_p": float, "dtr": int},
              ...
            ],
          },
          ...
        ]
    """
    try:
        from pfm import terminal_calendar_curated
    except Exception:  # pragma: no cover - module may genuinely be absent
        terminal_calendar_curated = None  # type: ignore[assignment]

    if terminal_calendar_curated is not None and hasattr(terminal_calendar_curated, "get_clusters"):
        try:
            clusters = terminal_calendar_curated.get_clusters()
            return [_normalise_cluster(c) for c in clusters if c]
        except Exception as e:  # pragma: no cover
            logger.warning(
                "terminal_calendar_curated.get_clusters() failed (%s) — "
                "falling back to strat28 fixture",
                e,
            )

    return _clusters_from_strat28()


def _normalise_cluster(raw: Any) -> dict[str, Any]:
    """Coerce a curated-cluster object to the scanner's internal dict form."""
    if isinstance(raw, dict):
        cluster_id = str(raw.get("cluster_id") or raw.get("id") or "")
        title = str(raw.get("title") or raw.get("event") or cluster_id)
        legs_raw = raw.get("legs") or raw.get("members") or []
    else:
        cluster_id = str(getattr(raw, "cluster_id", getattr(raw, "id", "")))
        title = str(getattr(raw, "title", getattr(raw, "event", cluster_id)))
        legs_raw = getattr(raw, "legs", []) or getattr(raw, "members", [])

    legs: list[dict[str, Any]] = []
    for leg in legs_raw:
        if isinstance(leg, dict):
            legs.append(
                {
                    "slug": str(leg.get("slug") or leg.get("id") or ""),
                    "name": str(leg.get("name") or leg.get("question") or ""),
                    "current_p": float(leg.get("current_p", leg.get("mid", 0.0)) or 0.0),
                    "dtr": int(leg.get("dtr", leg.get("days_to_resolution", 0)) or 0),
                }
            )
        else:
            legs.append(
                {
                    "slug": str(getattr(leg, "slug", getattr(leg, "id", ""))),
                    "name": str(getattr(leg, "name", "")),
                    "current_p": float(getattr(leg, "current_p", getattr(leg, "mid", 0.0)) or 0.0),
                    "dtr": int(getattr(leg, "dtr", getattr(leg, "days_to_resolution", 0)) or 0),
                }
            )
    return {"cluster_id": cluster_id, "title": title, "legs": legs}


def _clusters_from_strat28(path: Path | None = None) -> list[dict[str, Any]]:
    """Reconstruct curated-style clusters from the strat-28 revalidation file.

    Each ``event`` in ``pairs_sample`` becomes one cluster; both sides of
    every pair contribute legs (deduped by id). This is *not* the
    canonical curated set — it's the best-available proxy until
    :mod:`pfm.terminal_calendar_curated` lands.
    """
    p = path or _STRAT28_PATH
    if not p.exists():
        logger.warning("strat28 fallback missing at %s — scanner has no clusters", p)
        return []
    with p.open() as f:
        doc = json.load(f)

    by_event: dict[str, dict[str, dict[str, Any]]] = {}
    for pair in doc.get("pairs_sample", []):
        event = pair.get("event")
        if not event:
            continue
        for side_key in ("short", "long"):
            side = pair.get(side_key)
            if not side:
                continue
            mid_id = str(side.get("id") or "")
            if not mid_id:
                continue
            by_event.setdefault(event, {})[mid_id] = {
                "slug": mid_id.replace("_", "-"),
                "name": str(side.get("name") or mid_id),
                "current_p": float(side.get("mid", 0.0)),
                "dtr": int(side.get("dtr", 0)),
            }

    clusters: list[dict[str, Any]] = []
    for event, legs_by_id in by_event.items():
        if len(legs_by_id) < 2:
            continue
        cluster_id = event.replace(" ", "_")
        clusters.append(
            {
                "cluster_id": cluster_id,
                "title": event,
                "legs": sorted(legs_by_id.values(), key=lambda d: d["dtr"]),
            }
        )
    return clusters


# --- signal computation -----------------------------------------------------


def _build_signal(
    cluster_id: str,
    title: str,
    near_leg: dict[str, Any],
    far_leg: dict[str, Any],
) -> ActionableSignal | None:
    """Build a signal from a (near, far) leg pair if it crosses the threshold.

    Returns ``None`` when the pair is below the actionability bar.
    """
    lam_near = _implied_lambda(near_leg["current_p"], near_leg["dtr"])
    lam_far = _implied_lambda(far_leg["current_p"], far_leg["dtr"])
    log_ratio = _log_lambda_ratio(lam_near, lam_far)
    abs_ratio = abs(log_ratio)
    if abs_ratio < ACTIONABLE_THRESHOLD:
        return None

    # log_ratio > 0 ⇒ λ_far > λ_near ⇒ far leg is "hot" in hazard terms.
    # We long the cold leg (low λ, overpriced in hazard) and short the hot
    # leg (high λ, underpriced in hazard).
    if log_ratio > 0:
        long_raw, short_raw = near_leg, far_leg
        trade_type: Literal["FLATTEN_CURVE", "STEEPEN_CURVE"] = "STEEPEN_CURVE"
    else:
        long_raw, short_raw = far_leg, near_leg
        trade_type = "FLATTEN_CURVE"

    long_leg = ScannerLeg(
        slug=long_raw["slug"],
        name=long_raw["name"],
        current_p=long_raw["current_p"],
        implied_lambda=_implied_lambda(long_raw["current_p"], long_raw["dtr"]),
    )
    short_leg = ScannerLeg(
        slug=short_raw["slug"],
        name=short_raw["name"],
        current_p=short_raw["current_p"],
        implied_lambda=_implied_lambda(short_raw["current_p"], short_raw["dtr"]),
    )

    return ActionableSignal(
        cluster_id=cluster_id,
        title=title,
        trade_type=trade_type,
        long_leg=long_leg,
        short_leg=short_leg,
        log_lambda_ratio=round(log_ratio, 4),
        expected_ev_pct=_expected_ev_pct(abs_ratio),
        hold_window_days=DEFAULT_HOLD_WINDOW_DAYS,
        conviction=_classify_conviction(abs_ratio),
        entry_signal=f"Long {long_leg.slug} + Short {short_leg.slug}",
        exit_rule=(
            f"log λ-ratio reverts below {EXIT_THRESHOLD:.2f} OR "
            f"{DEFAULT_HOLD_WINDOW_DAYS} days elapsed"
        ),
    )


def _scan_clusters(clusters: list[dict[str, Any]]) -> list[ActionableSignal]:
    """Return every actionable signal across all curated clusters."""
    signals: list[ActionableSignal] = []
    for cluster in clusters:
        legs = cluster.get("legs", [])
        if len(legs) < 2:
            continue
        cluster_id = cluster["cluster_id"]
        title = cluster["title"]
        # Pairwise across all legs (clusters can have ≥2). Iterate
        # near→far so the trade-direction labelling stays consistent.
        sorted_legs = sorted(legs, key=lambda d: d["dtr"])
        for i in range(len(sorted_legs)):
            for j in range(i + 1, len(sorted_legs)):
                signal = _build_signal(cluster_id, title, sorted_legs[i], sorted_legs[j])
                if signal is not None:
                    signals.append(signal)
    # Highest-conviction first.
    signals.sort(key=lambda s: abs(s.log_lambda_ratio), reverse=True)
    return signals


# --- historical backtest ----------------------------------------------------


def _fetch_pair_history(
    long_slug: str,
    short_slug: str,
    lookback_days: int,
) -> tuple[list[str], list[float], list[float]]:
    """Pull aligned daily YES-token prices for the two legs.

    Uses :func:`pfm.sources.polymarket.fetch_factor_history` and intersects
    the two indices. Empty lists on any failure — the caller treats that
    as "no backtest available" rather than 500ing the request.
    """
    try:
        import pandas as pd

        from pfm.sources.polymarket import PolymarketClient, fetch_factor_history
    except Exception:  # pragma: no cover - import-time errors only
        return [], [], []

    end = pd.Timestamp(datetime.now(tz=UTC)).normalize()
    start = end - pd.Timedelta(days=lookback_days + 1)
    try:
        with PolymarketClient() as client:
            long_df = fetch_factor_history(client, long_slug, start=start, end=end)
            short_df = fetch_factor_history(client, short_slug, start=start, end=end)
    except Exception as e:
        logger.info("history fetch failed for %s/%s: %s", long_slug, short_slug, e)
        return [], [], []

    if long_df.empty or short_df.empty:
        return [], [], []

    common = long_df.index.intersection(short_df.index)
    if len(common) == 0:
        return [], [], []

    long_p = long_df.loc[common].iloc[:, 0].astype(float)
    short_p = short_df.loc[common].iloc[:, 0].astype(float)
    dates = [str(ts.date()) for ts in common]
    return dates, list(long_p.values), list(short_p.values)


def _backtest_cluster(
    cluster: dict[str, Any],
    lookback_days: int = BACKTEST_LOOKBACK_DAYS,
    hold_days: int = BACKTEST_HOLD_DAYS,
) -> HistoricalBacktest:
    """Replay the threshold-crossing signal over the last ``lookback_days``.

    Algorithm:

    1. Take the cluster's near/far legs (sorted by ``dtr``).
    2. Pull aligned daily prices for both legs.
    3. For each day, recompute the implied hazard rates (using the
       *original* days-to-resolution offsets, decayed daily).
    4. When |log λ-ratio| > threshold and we are flat, enter the trade
       in the appropriate direction; mark-to-market for ``hold_days``;
       PnL ≈ ``Δlog(price)`` summed across both legs (long − short).
    5. Sharpe is the standard ``mean / std × sqrt(252)`` annualisation
       on the daily PnL series.
    """
    legs = sorted(cluster.get("legs", []), key=lambda d: d["dtr"])
    cluster_id = cluster.get("cluster_id", "")
    if len(legs) < 2:
        return HistoricalBacktest(
            cluster_id=cluster_id,
            n_days=0,
            n_trades=0,
            cum_pnl=0.0,
            sharpe=0.0,
            points=[],
        )

    near, far = legs[0], legs[1]
    dates, near_prices, far_prices = _fetch_pair_history(near["slug"], far["slug"], lookback_days)
    if not dates:
        return HistoricalBacktest(
            cluster_id=cluster_id,
            n_days=0,
            n_trades=0,
            cum_pnl=0.0,
            sharpe=0.0,
            points=[],
        )

    n = len(dates)
    near_dtr_today = int(near["dtr"])
    far_dtr_today = int(far["dtr"])

    # Reconstruct the *historical* days-to-resolution by offsetting back
    # from "today". Day i is (n - 1 - i) days before today.
    points: list[BacktestPoint] = []
    cum = 0.0
    n_trades = 0
    in_trade = False
    trade_dir = 0  # +1 = STEEPEN (long near, short far); -1 = FLATTEN.
    days_held = 0
    daily_pnls: list[float] = []

    for i in range(n):
        offset = n - 1 - i
        dtr_near = max(1, near_dtr_today + offset)
        dtr_far = max(1, far_dtr_today + offset)
        lam_n = _implied_lambda(near_prices[i], dtr_near)
        lam_f = _implied_lambda(far_prices[i], dtr_far)
        ratio = _log_lambda_ratio(lam_n, lam_f)

        pnl_today = 0.0
        if in_trade and i > 0:
            # Δlog(price) on each leg.
            d_long = math.log(
                max(near_prices[i] if trade_dir > 0 else far_prices[i], 1e-6)
                / max(
                    near_prices[i - 1] if trade_dir > 0 else far_prices[i - 1],
                    1e-6,
                )
            )
            d_short = math.log(
                max(far_prices[i] if trade_dir > 0 else near_prices[i], 1e-6)
                / max(
                    far_prices[i - 1] if trade_dir > 0 else near_prices[i - 1],
                    1e-6,
                )
            )
            pnl_today = d_long - d_short
            days_held += 1
            if days_held >= hold_days:
                in_trade = False
                days_held = 0
                trade_dir = 0
        elif not in_trade and abs(ratio) > ACTIONABLE_THRESHOLD:
            in_trade = True
            trade_dir = 1 if ratio > 0 else -1
            days_held = 0
            n_trades += 1
            # Pay half the round-trip cost on entry.
            pnl_today = -RT_COST / 2.0

        cum += pnl_today
        daily_pnls.append(pnl_today)
        points.append(
            BacktestPoint(
                date=dates[i],
                log_lambda_ratio=round(ratio, 4),
                in_trade=in_trade,
                pnl_today=round(pnl_today, 6),
                cum_pnl=round(cum, 6),
            )
        )

    # Sharpe (annualised, daily).
    if daily_pnls:
        mu = sum(daily_pnls) / len(daily_pnls)
        var = sum((x - mu) ** 2 for x in daily_pnls) / max(1, len(daily_pnls) - 1)
        sd = math.sqrt(var) if var > 0 else 0.0
        sharpe = round((mu / sd) * math.sqrt(252.0), 3) if sd > 0 else 0.0
    else:
        sharpe = 0.0

    return HistoricalBacktest(
        cluster_id=cluster_id,
        n_days=n,
        n_trades=n_trades,
        cum_pnl=round(cum, 6),
        sharpe=sharpe,
        points=points,
    )


# --- router -----------------------------------------------------------------

router = APIRouter(prefix="/terminal", tags=["terminal-calendar-scanner"])


@router.get("/calendar-scanner/active")
def get_active_signals() -> list[ActionableSignal]:
    """Return every currently-actionable calendar arb across curated clusters.

    A signal is *actionable* when the absolute log λ-ratio between the two
    legs equals or exceeds 0.75 — the threshold at which Strategy-28's
    revalidation cell shows a positive net Sharpe (+1.78) after the 3.6 %
    round-trip taker cost.

    Sorted by |log λ-ratio| descending so the highest-conviction signals
    appear first. Empty list when no cluster crosses the bar.
    """
    clusters = _load_curated_clusters()
    return _scan_clusters(clusters)


@router.get("/calendar-scanner/historical")
def get_historical_backtest(
    cluster_id: Annotated[str, Query(min_length=1, max_length=200)],
    lookback_days: Annotated[int, Query(ge=7, le=365)] = BACKTEST_LOOKBACK_DAYS,
) -> HistoricalBacktest:
    """90-day cluster-level PnL backtest of the threshold-crossing signal.

    For each day in the window, recomputes the pairwise log λ-ratio and,
    when it crosses the 0.75 threshold while the book is flat, enters a
    5-day-hold position in the direction implied by the sign of the
    ratio. PnL is Δlog(price) per leg, netted long − short, with half
    the round-trip cost charged on entry.

    Returns an empty point list (and zero stats) when live price history
    cannot be fetched — the endpoint never 500s on a data-source outage.
    """
    clusters = _load_curated_clusters()
    target = next(
        (c for c in clusters if c.get("cluster_id") == cluster_id),
        None,
    )
    if target is None:
        raise HTTPException(
            status_code=404,
            detail=f"unknown cluster_id: {cluster_id!r}",
        )
    return _backtest_cluster(target, lookback_days=lookback_days)


__all__ = [
    "ACTIONABLE_THRESHOLD",
    "ActionableSignal",
    "BacktestPoint",
    "HistoricalBacktest",
    "ScannerLeg",
    "router",
]


# Suppress unused-import warning in the timedelta path below — it's
# referenced indirectly via pandas.Timedelta in _fetch_pair_history.
_ = timedelta
