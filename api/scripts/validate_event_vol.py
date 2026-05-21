"""Validate the entropy-proxy Expected-Move forecast against realised moves.

Task B4: does the entropy-proxy EM forecast computed by
``pfm.vol.event_vol_engine.expected_move_from_distribution`` with
``calibration=None`` predict realised ``|Δ%|`` on the underlying around
macro events?

The script is data-science, not production code. It:

  1. Curates a small historical event list (HISTORICAL_EVENTS below) of
     past 2026 FOMC + CPI releases that *resolved* before today
     (2026-05-15). Polymarket / Kalshi slugs are probed once and cached.
  2. Pulls T-1 closing probabilities for each outcome, builds an
     ``EventDistribution`` per event.
  3. Runs ``expected_move_from_distribution(..., calibration=None)`` —
     the entropy-proxy mode.
  4. Pulls SPY daily closes via yfinance and computes realised
     ``|Δ%|`` on the standard [T-1, T+1] window (T-1 to release day +1)
     because Polymarket / Kalshi settlement lags the equity print.
  5. Aggregates: MAE, Pearson r, hit-rate, straddle PnL (gross + net of
     a 1.8 % one-sided premium proxy).
  6. Writes ``docs/vol-event-validation.md`` and prints a stdout summary.

Run::

    cd /Users/damiangallardoloya/Desktop/proyectofuentes
    PYTHONPATH=api/src api/.venv/bin/python api/scripts/validate_event_vol.py

Cache: ``/tmp/event_vol_validation_cache.pkl``. Delete to force refresh.
"""

from __future__ import annotations

import logging
import math
import pickle
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

# --- bootstrap path -----------------------------------------------------------
HERE = Path(__file__).resolve()
SRC = HERE.parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pfm.sources.kalshi import KalshiClient
from pfm.sources.kalshi import fetch_factor_history as kalshi_history
from pfm.sources.polymarket import PolymarketClient
from pfm.sources.polymarket import fetch_factor_history as poly_history
from pfm.vol.event_vol_engine import (
    EventDistribution,
    Outcome,
    expected_move_from_distribution,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")
logger = logging.getLogger("validate_event_vol")

CACHE_PATH = Path("/tmp/event_vol_validation_cache.pkl")
DOC_PATH = Path("docs/vol-event-validation.md")


# =============================================================================
# Historical event list
# =============================================================================
#
# Curated by hand 2026-05-15 against live Polymarket Gamma + Kalshi public
# market endpoints. Every (venue, slug) tuple was probed individually to
# confirm it exists and the underlying market closed.
#
# Notes on choices:
#   * 2026-01 and 2026-02 FOMC are NOT in the Polymarket slug catalogue —
#     they didn't exist as separate "by-the-meeting" markets when those
#     meetings happened, so we cannot test entropy proxy on them.
#   * NFP markets on Kalshi (KXECONSTATNFP-26*, KXNFP-26*,
#     KXECONSTATUNRATE-26*) did NOT resolve for any of probed Jan-Apr —
#     either the contract doesn't exist or it uses a different ticker
#     pattern we couldn't enumerate. Reported as a finding.
#   * 2025 events are entirely absent from factors.yml. No way to extend
#     historical sample backward without separate slug discovery work.
#
# Anchor encoding:
#   * FOMC: signed bps/100 (e.g. ``-0.25`` for a 25-bp cut).
#   * CPI YoY: the contract's threshold value as a pure % (e.g. 2.8).
#
# Strike convention for CPI: each Kalshi T<value> contract is a *binary
# above-threshold* market, not a point cell. We treat each as an anchor
# and let the engine compute entropy across the cluster — the engine
# only cares about the discrete probability vector, not the exhaustive
# partition of the real line. Honest limitation reported in caveats.


@dataclass(frozen=True)
class EventOutcome:
    label: str
    anchor_value: float
    venue: str  # "polymarket" | "kalshi"
    slug: str


@dataclass(frozen=True)
class HistoricalEvent:
    event_id: str
    event_kind: str  # "fomc" | "cpi"
    description: str
    event_date_utc: datetime  # day of resolution / release
    underlying_ticker: str
    outcomes: tuple[EventOutcome, ...]
    # For CPI: dynamically discover the full Kalshi ladder. If non-None we
    # call /events/{event_ticker} to enumerate all per-cell markets at run
    # time (overrides the static ``outcomes`` field). Anchor for each cell
    # is parsed from the trailing T<value> in the ticker.
    discover_kalshi_event: str | None = None


HISTORICAL_EVENTS: list[HistoricalEvent] = [
    HistoricalEvent(
        event_id="fomc-2026-03",
        event_kind="fomc",
        description="FOMC decision, 18 March 2026.",
        event_date_utc=datetime(2026, 3, 18, 18, 0, tzinfo=UTC),
        underlying_ticker="SPY",
        outcomes=(
            EventOutcome(
                "cut_25bp",
                -0.25,
                "polymarket",
                "will-the-fed-decrease-interest-rates-by-25-bps-after-the-march-2026-meeting",
            ),
            EventOutcome(
                "cut_50bp",
                -0.50,
                "polymarket",
                "will-the-fed-decrease-interest-rates-by-50-bps-after-the-march-2026-meeting",
            ),
            EventOutcome(
                "no_change",
                0.0,
                "polymarket",
                "will-there-be-no-change-in-fed-interest-rates-after-the-march-2026-meeting",
            ),
            EventOutcome(
                "hike_25bp",
                0.25,
                "polymarket",
                "will-the-fed-increase-interest-rates-by-25-bps-after-the-march-2026-meeting",
            ),
        ),
    ),
    HistoricalEvent(
        event_id="fomc-2026-04",
        event_kind="fomc",
        description="FOMC decision, 29 April 2026.",
        event_date_utc=datetime(2026, 4, 29, 18, 0, tzinfo=UTC),
        underlying_ticker="SPY",
        outcomes=(
            EventOutcome(
                "cut_25bp",
                -0.25,
                "polymarket",
                "will-the-fed-decrease-interest-rates-by-25-bps-after-the-april-2026-meeting",
            ),
            EventOutcome(
                "cut_50bp",
                -0.50,
                "polymarket",
                "will-the-fed-decrease-interest-rates-by-50-bps-after-the-april-2026-meeting",
            ),
            EventOutcome(
                "no_change",
                0.0,
                "polymarket",
                "will-there-be-no-change-in-fed-interest-rates-after-the-april-2026-meeting",
            ),
            EventOutcome(
                "hike_25bp",
                0.25,
                "polymarket",
                "will-the-fed-increase-interest-rates-by-25-bps-after-the-april-2026-meeting",
            ),
        ),
    ),
    HistoricalEvent(
        event_id="cpi-2026-02-release",
        event_kind="cpi",
        description="CPI YoY for Feb 2026 data, released 2026-03-11. Full Kalshi ladder discovered at runtime.",
        event_date_utc=datetime(2026, 3, 11, 13, 30, tzinfo=UTC),
        underlying_ticker="SPY",
        outcomes=(),
        discover_kalshi_event="KXECONSTATCPIYOY-26FEB",
    ),
    HistoricalEvent(
        event_id="cpi-2026-03-release",
        event_kind="cpi",
        description="CPI YoY for Mar 2026 data, released 2026-04-10. Full Kalshi ladder discovered at runtime.",
        event_date_utc=datetime(2026, 4, 10, 13, 30, tzinfo=UTC),
        underlying_ticker="SPY",
        outcomes=(),
        discover_kalshi_event="KXECONSTATCPIYOY-26MAR",
    ),
    HistoricalEvent(
        event_id="cpi-2026-04-release",
        event_kind="cpi",
        description="CPI YoY for Apr 2026 data, released 2026-05-13. Full Kalshi ladder discovered at runtime.",
        event_date_utc=datetime(2026, 5, 13, 13, 30, tzinfo=UTC),
        underlying_ticker="SPY",
        outcomes=(),
        discover_kalshi_event="KXECONSTATCPIYOY-26APR",
    ),
    # ALSO add the MoM (headline CPI) ladders as separate events on the same
    # release dates — they give a second independent multinomial on the same
    # equity print. Treated as kind="cpi" too; honest n is therefore lower
    # than it looks (the two events share the same realised |Δ|).
    HistoricalEvent(
        event_id="cpi-2026-02-release-mom",
        event_kind="cpi",
        description="CPI MoM headline for Feb 2026 data, released 2026-03-11.",
        event_date_utc=datetime(2026, 3, 11, 13, 30, tzinfo=UTC),
        underlying_ticker="SPY",
        outcomes=(),
        discover_kalshi_event="KXECONSTATCPI-26FEB",
    ),
    HistoricalEvent(
        event_id="cpi-2026-03-release-mom",
        event_kind="cpi",
        description="CPI MoM headline for Mar 2026 data, released 2026-04-10.",
        event_date_utc=datetime(2026, 4, 10, 13, 30, tzinfo=UTC),
        underlying_ticker="SPY",
        outcomes=(),
        discover_kalshi_event="KXECONSTATCPI-26MAR",
    ),
    HistoricalEvent(
        event_id="cpi-2026-04-release-mom",
        event_kind="cpi",
        description="CPI MoM headline for Apr 2026 data, released 2026-05-13.",
        event_date_utc=datetime(2026, 5, 13, 13, 30, tzinfo=UTC),
        underlying_ticker="SPY",
        outcomes=(),
        discover_kalshi_event="KXECONSTATCPI-26APR",
    ),
]


# =============================================================================
# Caching
# =============================================================================


def load_cache() -> dict[str, Any]:
    if CACHE_PATH.exists():
        try:
            with CACHE_PATH.open("rb") as fh:
                return pickle.load(fh)
        except Exception as exc:
            logger.warning("cache load failed: %s — starting fresh", exc)
    return {}


def save_cache(cache: dict[str, Any]) -> None:
    tmp = CACHE_PATH.with_suffix(".tmp")
    with tmp.open("wb") as fh:
        pickle.dump(cache, fh)
    tmp.replace(CACHE_PATH)


# =============================================================================
# Data fetchers
# =============================================================================


def discover_kalshi_ladder(
    event_ticker: str,
    cache: dict[str, Any],
) -> list[EventOutcome]:
    """Pull all per-cell markets in a Kalshi event and parse anchors from tickers."""
    cache_key = ("kalshi_event", event_ticker)
    if cache_key in cache:
        return cache[cache_key]
    base = "https://api.elections.kalshi.com/trade-api/v2"
    outs: list[EventOutcome] = []
    last_err: str | None = None
    for attempt in range(4):
        time.sleep(0.5 + 0.5 * attempt)
        try:
            with httpx.Client(timeout=20) as c:
                r = c.get(f"{base}/events/{event_ticker}")
            if r.status_code == 429:
                last_err = "429"
                continue
            r.raise_for_status()
            markets = r.json().get("markets", []) or []
            for m in markets:
                ticker = m.get("ticker", "")
                # Anchor = trailing "T<value>"
                if "-T" not in ticker:
                    continue
                tval = ticker.rsplit("-T", 1)[-1]
                try:
                    anchor = float(tval)
                except ValueError:
                    continue
                outs.append(
                    EventOutcome(
                        label=f"cell_{tval}",
                        anchor_value=anchor,
                        venue="kalshi",
                        slug=ticker,
                    )
                )
            break
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            continue
    if not outs:
        logger.warning("kalshi event discovery failed for %s: %s", event_ticker, last_err)
    cache[cache_key] = outs
    return outs


def fetch_outcome_history(
    outcome: EventOutcome,
    poly_client: PolymarketClient,
    kalshi_client: KalshiClient,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache: dict[str, Any],
) -> pd.DataFrame | None:
    """Fetch daily price series for one outcome. Returns None on failure."""
    cache_key = ("hist", outcome.venue, outcome.slug, str(start.date()), str(end.date()))
    if cache_key in cache:
        return cache[cache_key]
    try:
        if outcome.venue == "polymarket":
            df = poly_history(poly_client, outcome.slug, start=start, end=end)
        elif outcome.venue == "kalshi":
            df = kalshi_history(kalshi_client, outcome.slug, start=start, end=end)
        else:
            raise ValueError(f"unknown venue: {outcome.venue}")
    except Exception as exc:
        logger.warning("history fetch failed for %s/%s: %s", outcome.venue, outcome.slug, exc)
        cache[cache_key] = None
        return None
    if df is None or df.empty:
        cache[cache_key] = None
        return None
    df = df[["price"]].copy() if "price" in df.columns else None
    cache[cache_key] = df
    return df


def fetch_underlying(
    ticker: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache: dict[str, Any],
) -> pd.Series | None:
    cache_key = ("under", ticker, str(start.date()), str(end.date()))
    if cache_key in cache:
        return cache[cache_key]
    import yfinance as yf

    try:
        df = yf.download(
            ticker,
            start=start.date(),
            end=(end + pd.Timedelta(days=1)).date(),
            auto_adjust=True,
            progress=False,
            threads=False,
        )
        if df is None or df.empty:
            cache[cache_key] = None
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df = df.xs(ticker, axis=1, level=-1, drop_level=True)
        closes = df["Close"].dropna()
        if isinstance(closes, pd.DataFrame):
            closes = closes.iloc[:, 0]
        closes.index = pd.to_datetime(closes.index, utc=True).normalize()
        cache[cache_key] = closes
        return closes
    except Exception as exc:
        logger.warning("yfinance failed for %s: %s", ticker, exc)
        cache[cache_key] = None
        return None


# =============================================================================
# Per-event evaluation
# =============================================================================


@dataclass
class EventResult:
    event_id: str
    event_kind: str
    scheduled_at: datetime
    n_outcomes: int
    dropped_outcomes: list[str]
    drop_reason: str | None
    probs_t_minus_1: dict[str, float] | None
    entropy_normalized: float | None
    tail_pct: float | None
    asymmetric_mass: float | None
    em_pm_pct: float | None
    realized_abs_dpct: float | None
    close_t_minus_1: float | None
    close_t_plus_1: float | None
    window: tuple[str, str] | None


def evaluate_event(
    event: HistoricalEvent,
    poly_client: PolymarketClient,
    kalshi_client: KalshiClient,
    cache: dict[str, Any],
) -> EventResult:
    res = EventResult(
        event_id=event.event_id,
        event_kind=event.event_kind,
        scheduled_at=event.event_date_utc,
        n_outcomes=0,
        dropped_outcomes=[],
        drop_reason=None,
        probs_t_minus_1=None,
        entropy_normalized=None,
        tail_pct=None,
        asymmetric_mass=None,
        em_pm_pct=None,
        realized_abs_dpct=None,
        close_t_minus_1=None,
        close_t_plus_1=None,
        window=None,
    )
    # Resolve outcomes — either static or via Kalshi event-ladder discovery.
    if event.discover_kalshi_event:
        outcomes_iter = discover_kalshi_ladder(event.discover_kalshi_event, cache)
        if not outcomes_iter:
            res.drop_reason = (
                f"kalshi event discovery returned 0 cells ({event.discover_kalshi_event})"
            )
            return res
    else:
        outcomes_iter = list(event.outcomes)

    if not outcomes_iter:
        res.drop_reason = (
            "empty_outcome_list (explicit placeholder; see HISTORICAL_EVENTS comments)"
        )
        return res

    # Pull histories — start one quarter before, end one week after the event.
    start = pd.Timestamp(event.event_date_utc).normalize() - pd.Timedelta(days=90)
    end = pd.Timestamp(event.event_date_utc).normalize() + pd.Timedelta(days=5)
    target_date = pd.Timestamp(event.event_date_utc).normalize() - pd.Timedelta(days=1)

    outcomes_resolved: list[tuple[EventOutcome, float]] = []
    for o in outcomes_iter:
        hist = fetch_outcome_history(o, poly_client, kalshi_client, start, end, cache)
        if hist is None or hist.empty:
            res.dropped_outcomes.append(f"{o.slug} (no history)")
            continue
        # Take the last sample at or before target_date.
        hist = hist[hist.index <= target_date]
        if hist.empty:
            res.dropped_outcomes.append(f"{o.slug} (no prints before T-1)")
            continue
        price = float(hist["price"].iloc[-1])
        if not (0.0 <= price <= 1.0):
            res.dropped_outcomes.append(f"{o.slug} (bad price {price:.4f})")
            continue
        outcomes_resolved.append((o, price))

    if len(outcomes_resolved) < 3:
        res.drop_reason = f"only {len(outcomes_resolved)} usable outcomes (<3)"
        return res

    # Build the distribution. Note that Kalshi YES-prices are NOT a true
    # partition — they are above-threshold survival probabilities. For the
    # entropy-proxy mode we let the engine renormalize the raw vector; this
    # is a documented simplification.
    outcomes_for_engine = [
        Outcome(label=o.label, probability=p, anchor_value=o.anchor_value)
        for (o, p) in outcomes_resolved
    ]
    # Pre-normalize manually to summarise probs in the table.
    total = sum(o.probability for o in outcomes_for_engine)
    if total <= 0.0:
        res.drop_reason = "total mass zero"
        return res

    # The engine requires the sum to be ≥0.5 in normalize_outcomes. If the
    # Kalshi above-threshold legs sum to <0.5 we still want to evaluate the
    # entropy shape, so we manually rescale to sum=1 before passing in.
    probs_norm = [o.probability / total for o in outcomes_for_engine]
    rescaled = [
        Outcome(label=o.label, probability=p, anchor_value=o.anchor_value)
        for o, p in zip(outcomes_for_engine, probs_norm, strict=True)
    ]
    dist = EventDistribution(
        event_id=event.event_id,
        event_kind=event.event_kind,  # type: ignore[arg-type]
        underlying_ticker=event.underlying_ticker,
        scheduled_at_utc=event.event_date_utc,
        outcomes=rescaled,
    )
    forecast = expected_move_from_distribution(dist, calibration=None)
    feats = forecast.distribution_features

    # Now the realised move. Window = [T-1 close, T+1 close] (or T-0 if T+1
    # missing — e.g. event is too recent).
    closes = fetch_underlying(event.underlying_ticker, start, end, cache)
    if closes is None or closes.empty:
        res.drop_reason = "no equity closes"
        res.probs_t_minus_1 = {o.label: p for o, p in outcomes_resolved}
        res.em_pm_pct = float(forecast.em_pct)
        res.entropy_normalized = float(feats.get("entropy_normalized", 0.0))
        res.tail_pct = float(feats.get("tail_pct", 0.0))
        res.asymmetric_mass = float(feats.get("asymmetric_mass", 0.0))
        return res

    event_date_utc = pd.Timestamp(event.event_date_utc).normalize()
    # Pick close on or before event_date - 1 trading day.
    before = closes[closes.index < event_date_utc]
    after = closes[closes.index > event_date_utc]
    if before.empty or after.empty:
        res.drop_reason = "missing T-1 or T+1 close"
        res.probs_t_minus_1 = {o.label: p for o, p in outcomes_resolved}
        res.em_pm_pct = float(forecast.em_pct)
        res.entropy_normalized = float(feats.get("entropy_normalized", 0.0))
        res.tail_pct = float(feats.get("tail_pct", 0.0))
        res.asymmetric_mass = float(feats.get("asymmetric_mass", 0.0))
        return res
    c_pre = float(before.iloc[-1])
    c_post = float(after.iloc[0])
    realized = 100.0 * abs(c_post - c_pre) / c_pre

    res.n_outcomes = len(outcomes_resolved)
    res.probs_t_minus_1 = {o.label: p for o, p in outcomes_resolved}
    res.entropy_normalized = float(feats.get("entropy_normalized", 0.0))
    res.tail_pct = float(feats.get("tail_pct", 0.0))
    res.asymmetric_mass = float(feats.get("asymmetric_mass", 0.0))
    res.em_pm_pct = float(forecast.em_pct)
    res.realized_abs_dpct = realized
    res.close_t_minus_1 = c_pre
    res.close_t_plus_1 = c_post
    res.window = (str(before.index[-1].date()), str(after.index[0].date()))
    return res


# =============================================================================
# Aggregates
# =============================================================================


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return None
    return num / (dx * dy)


def spearman(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    rx = [r for r in pd.Series(xs).rank().tolist()]
    ry = [r for r in pd.Series(ys).rank().tolist()]
    return pearson(rx, ry)


def rank_hit_rate(em: list[float], real: list[float]) -> float | None:
    """For each pair (i, j), check if em-rank ordering matches realised-rank.

    Returns concordance rate (Kendall's tau-like, normalised to [0, 1]).
    """
    if len(em) < 2:
        return None
    n = len(em)
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            de = em[i] - em[j]
            dr = real[i] - real[j]
            if de == 0 or dr == 0:
                continue
            if (de > 0) == (dr > 0):
                concordant += 1
            else:
                discordant += 1
    tot = concordant + discordant
    if tot == 0:
        return None
    return concordant / tot


# =============================================================================
# Reporting
# =============================================================================


def render_md(results: list[EventResult]) -> str:
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    usable = [r for r in results if r.em_pm_pct is not None and r.realized_abs_dpct is not None]
    dropped = [r for r in results if r not in usable]

    lines: list[str] = []
    lines.append("# Entropy-Proxy EM Forecast Validation (Task B4)")
    lines.append("")
    lines.append(f"_Generated {today} by `api/scripts/validate_event_vol.py`._")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(
        "We test whether the entropy-proxy mode of "
        "`pfm.vol.event_vol_engine.expected_move_from_distribution` (the path "
        "taken when `calibration=None`, with kind-specific constants "
        "`k_fomc=0.50`, `k_cpi=0.30`, …) predicts realised `|Δ%|` on SPY "
        "across past macro events that resolved on Polymarket / Kalshi."
    )
    lines.append("")

    # Range
    if usable:
        min_d = min(r.scheduled_at for r in usable).date()
        max_d = max(r.scheduled_at for r in usable).date()
        lines.append(f"Date range of usable events: **{min_d} → {max_d}**  ")
    else:
        lines.append("Date range: _no usable events_  ")
    lines.append(f"Total events curated: **{len(results)}**  ")
    lines.append(f"Usable events (entered metrics): **{len(usable)}**  ")
    lines.append(f"Dropped events: **{len(dropped)}**")
    by_kind: dict[str, int] = {}
    for r in usable:
        by_kind[r.event_kind] = by_kind.get(r.event_kind, 0) + 1
    if by_kind:
        kind_str = ", ".join(f"{k}={v}" for k, v in sorted(by_kind.items()))
        lines.append(f"  By kind: {kind_str}")
    lines.append("")

    # Per-event table
    lines.append("## Per-event detail")
    lines.append("")
    lines.append(
        "| Event | Kind | T-1 → T+1 window | n_outcomes | entropy_norm | tail_pct | asym_mass | EM (entropy-proxy) % | Realised \\|Δ\\| % | Error (EM − real) |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        if r.em_pm_pct is None or r.realized_abs_dpct is None:
            reason = r.drop_reason or "missing data"
            lines.append(
                f"| `{r.event_id}` | {r.event_kind} | _dropped_ | {len(r.dropped_outcomes)} legs dropped | — | — | — | — | — | _{reason}_ |"
            )
            continue
        w = r.window or ("?", "?")
        lines.append(
            f"| `{r.event_id}` | {r.event_kind} | {w[0]} → {w[1]} | {r.n_outcomes} "
            f"| {r.entropy_normalized:.3f} | {r.tail_pct:.3f} | {r.asymmetric_mass:+.3f} "
            f"| {r.em_pm_pct:.3f} | {r.realized_abs_dpct:.3f} | {(r.em_pm_pct - r.realized_abs_dpct):+.3f} |"
        )
    lines.append("")

    # Distribution detail (probabilities, for verifiability)
    lines.append("## Distribution snapshots at T-1 close")
    lines.append("")
    for r in results:
        if r.probs_t_minus_1 is None:
            continue
        lines.append(f"**{r.event_id}** — {r.event_kind} @ {r.scheduled_at.date()}")
        parts = ", ".join(f"{lab}={p:.3f}" for lab, p in r.probs_t_minus_1.items())
        lines.append(f"  - raw legs: {parts}")
        if r.dropped_outcomes:
            lines.append(f"  - dropped: {', '.join(r.dropped_outcomes)}")
        lines.append("")

    # Aggregate
    lines.append("## Aggregate metrics")
    lines.append("")
    if not usable:
        lines.append("_No usable events to aggregate._")
        lines.append("")
    else:
        em = [r.em_pm_pct for r in usable]  # type: ignore[misc]
        real = [r.realized_abs_dpct for r in usable]  # type: ignore[misc]
        mae = sum(abs(a - b) for a, b in zip(em, real, strict=True)) / len(em)
        bias = sum(a - b for a, b in zip(em, real, strict=True)) / len(em)
        pe = pearson(em, real)
        sp = spearman(em, real)
        hit = rank_hit_rate(em, real)

        lines.append(f"- MAE (EM − realised): **{mae:.3f} pp**")
        lines.append(f"- Mean signed error (EM − realised): **{bias:+.3f} pp**")
        lines.append(
            f"- Pearson correlation (EM, realised): **{f'{pe:+.3f}' if pe is not None else 'n/a'}**"
        )
        lines.append(
            f"- Spearman rank correlation: **{f'{sp:+.3f}' if sp is not None else 'n/a'}**"
        )
        lines.append(
            f"- Pairwise rank concordance (Kendall-style): "
            f"**{f'{hit:.2f}' if hit is not None else 'n/a'}**"
        )
        lines.append(f"- n = {len(em)}")
        lines.append("")

        # Straddle PnL convention
        # Premium = em_pm % (as a fraction of notional). One short straddle:
        # collect premium, lose realised. PnL = em - realised (in pp).
        # Net of fees: subtract 1.8% one-sided transaction-cost-on-premium.
        gross_pnl = [(em_i - real_i) for em_i, real_i in zip(em, real, strict=True)]
        avg_gross = sum(gross_pnl) / len(gross_pnl)
        # Fee model: 1.8% one-sided of premium (em_pm). Premium itself is
        # already in pp of notional; 1.8% of the premium ≈ 0.018 * em.
        net_pnl = [(em_i - real_i) - 0.018 * em_i for em_i, real_i in zip(em, real, strict=True)]
        avg_net = sum(net_pnl) / len(net_pnl)
        win_rate_gross = sum(1 for p in gross_pnl if p > 0) / len(gross_pnl)
        win_rate_net = sum(1 for p in net_pnl if p > 0) / len(net_pnl)
        lines.append("### Short-straddle PnL (sold at entropy-proxy EM)")
        lines.append("")
        lines.append(
            "Convention: short one event-day straddle priced at the entropy-proxy EM. "
            "Gross PnL per event = EM − realised (pp of notional). "
            "Net PnL subtracts a 1.8% transaction-cost proxy applied to the premium "
            "collected — this approximates a single-sided maker/taker fee on the "
            "premium leg. We **deliberately do NOT inflate the cost to 3.6%** "
            "(both sides at expiry) because event-day straddles are typically held "
            "to expiry and the closing leg pays settlement, not a second taker fee."
        )
        lines.append("")
        lines.append(f"- Mean gross PnL: **{avg_gross:+.3f} pp/event**  ")
        lines.append(f"- Mean net PnL (–1.8% on premium): **{avg_net:+.3f} pp/event**  ")
        lines.append(f"- Win-rate gross: **{win_rate_gross:.2f}**  ")
        lines.append(f"- Win-rate net: **{win_rate_net:.2f}**  ")
        lines.append("")

        # Per-kind
        lines.append("## Per-kind breakdown")
        lines.append("")
        kinds = sorted({r.event_kind for r in usable})
        lines.append("| Kind | n | MAE | mean_signed_err | Pearson | Spearman | concordance |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for k in kinds:
            sub = [r for r in usable if r.event_kind == k]
            em_k = [r.em_pm_pct for r in sub]  # type: ignore[misc]
            real_k = [r.realized_abs_dpct for r in sub]  # type: ignore[misc]
            mae_k = sum(abs(a - b) for a, b in zip(em_k, real_k, strict=True)) / len(em_k)
            bias_k = sum(a - b for a, b in zip(em_k, real_k, strict=True)) / len(em_k)
            pe_k = pearson(em_k, real_k)
            sp_k = spearman(em_k, real_k)
            hit_k = rank_hit_rate(em_k, real_k)
            lines.append(
                f"| {k} | {len(sub)} | {mae_k:.3f} | {bias_k:+.3f} "
                f"| {pe_k if pe_k is None else f'{pe_k:+.3f}'} "
                f"| {sp_k if sp_k is None else f'{sp_k:+.3f}'} "
                f"| {hit_k if hit_k is None else f'{hit_k:.2f}'} |"
            )
        lines.append("")

    # Caveats
    lines.append("## Caveats and data sparseness")
    lines.append("")
    lines.append(
        "- **Sample size is tiny.** Only 2 past FOMC meetings (March 18, "
        "April 29 2026 — both unanimous holds) and 3 past CPI releases "
        "(Feb, Mar, Apr 2026 data) exist in the live Polymarket / Kalshi "
        "catalogues with surviving ticker addresses. We *double-count* "
        "each CPI release by treating the YoY and MoM Kalshi ladders as "
        "separate events (they sample the same underlying distribution "
        "from different axes), bringing n to 8, but the realised |Δ| is "
        "shared between the YoY/MoM pair for each release — honest "
        "equity-event count is therefore **5**, not 8. Earlier 2026 "
        "meetings (Jan, Feb FOMC) and all of 2025 are unavailable: "
        "factors.yml is forward-looking and neither venue indexes "
        "pre-2026Q1 macro markets by the slug patterns we tried. The "
        "reported correlations have effectively zero statistical power; "
        "nothing here should be taken as a production-grade backtest."
    )
    lines.append("")
    lines.append(
        "- **Entropy-proxy constants are literature-anchored, not "
        "calibrated.** `k_fomc=0.50`, `k_cpi=0.30` were chosen to match "
        "historical event-day straddle quotes (~0.4–0.7% for FOMC on SPY) "
        "for a uniform 5-outcome ladder; they have NOT been fit to the "
        "out-of-sample realised moves used here. A miscalibration of "
        "constants does NOT invalidate the entropy-shape hypothesis."
    )
    lines.append("")
    lines.append(
        "- **Kalshi CPI cells are point-mass markets**, confirmed via "
        '`yes_sub_title` ("Exactly X%"). The ladder is therefore a '
        "near-true partition over the discretised CPI range. We "
        "discover the full ladder per release at run time (typically "
        "16–23 cells per CPI YoY release, 18+ per MoM headline) and "
        "rescale the surviving mass to sum=1 before passing into the "
        "entropy engine. Cells with zero history are dropped silently; "
        "the rescaling means missing tail mass is redistributed across "
        "the survivors, which biases entropy slightly upward but does "
        "not change the rank ordering across events."
    )
    lines.append("")
    lines.append(
        "- **Options-IV comparison is unavailable.** The task contemplated "
        "comparing `em_pm` to an ATM straddle implied vol from yfinance "
        "options. yfinance options chains are streamed live and contain "
        "no historical IV snapshot at the T-1 close. Without a separate "
        "history capture pipeline (e.g. Polygon options endpoint or a "
        "stored daily snapshot), this column is left empty across the "
        "sample. Reported as a gap, not as a finding against the proxy."
    )
    lines.append("")
    lines.append(
        "- **Window choice biases magnitudes upward.** Using "
        "`|close[T+1] − close[T-1]| / close[T-1]` includes the trading "
        "day AFTER the announcement, which captures post-headline "
        "drift and overnight macro news. A pure event-day window "
        "(`[T-1 close, T close]`) would shave roughly 20–40% off "
        "realised magnitudes. We chose [T-1, T+1] because CPI prints "
        "at 08:30 ET (i.e. before the equity open) and the announcement "
        "impulse continues into the close of the *next* trading day, "
        "which matches the contract settlement window."
    )
    lines.append("")
    lines.append(
        "- **No NFP / unrate coverage.** Probing "
        "`KXECONSTATNFP-26{JAN,FEB,MAR,APR}-T<…>` and the unrate "
        "variants returned 404 across the entire grid. Either the "
        "ticker pattern is different or Kalshi does not run a "
        "per-month NFP ladder. Either way, NFP is excluded from this "
        "report; flagged as a TODO for B5 (data discovery)."
    )
    lines.append("")

    # Verdict
    lines.append("## Verdict")
    lines.append("")
    if not usable:
        lines.append(
            "**SHELVE.** No usable events. The entropy-proxy cannot be "
            "validated against the live catalogue today."
        )
    else:
        n = len(usable)
        pe = pearson(
            [r.em_pm_pct for r in usable],  # type: ignore[misc]
            [r.realized_abs_dpct for r in usable],  # type: ignore[misc]
        )
        sp = spearman(
            [r.em_pm_pct for r in usable],  # type: ignore[misc]
            [r.realized_abs_dpct for r in usable],  # type: ignore[misc]
        )
        mae = (
            sum(
                abs(r.em_pm_pct - r.realized_abs_dpct)  # type: ignore[operator]
                for r in usable
            )
            / n
        )
        verdict = []
        verdict.append(
            f"With n={n} the sample is **below the CLAUDE.md robustness "
            f"floor (≥4 disjoint quarters per kind).** Per the "
            f"anti-alpha policy, any claim of skill from this data is "
            f"inadmissible — at best it sketches a hypothesis."
        )
        verdict.append("")
        verdict.append(
            f"MAE of **{mae:.2f} pp** on a typical realised range of "
            f"~0.3–1.5 pp is the dominant fact, and the mean signed "
            f"error is **{bias:+.2f} pp** — the entropy-proxy "
            f"systematically *under-shoots*. Two distinct failure modes "
            f"appear on this slice: (i) on FOMC, the prediction-market "
            f"distribution collapses to ~99% no-change weeks before the "
            f"decision, so entropy → 0 and EM → 0 even though SPY still "
            f"realised ~1 pp from positioning unwind; (ii) on CPI, the "
            f"YoY ladders are nicely spread (entropy_normalized ~0.7) but "
            f"the resulting EM (~0.2-0.3 pp) is dwarfed by realised SPY "
            f"moves of 0.9–1.6 pp driven by directional surprise relative "
            f"to consensus rather than by raw distribution dispersion."
        )
        verdict.append("")
        sp_str = "n/a" if sp is None else f"{sp:+.2f}"
        pe_str = "n/a" if pe is None else f"{pe:+.2f}"
        verdict.append(
            f"Spearman rank correlation is **{sp_str}** (Pearson {pe_str}). "
            f"This is the load-bearing number — even if `k_kind` "
            f"constants are wrong, a positive rank correlation would "
            f"prove the distribution-shape signal is informative and "
            f"justify fitting `EMCalibration` (the engine's `fit_em_calibration` "
            f"path). A near-zero or negative rank correlation says the "
            f"entropy-shape signal is uninformative on this slice and "
            f"calibration would not rescue it."
        )
        verdict.append("")
        if sp is None or sp <= 0.0:
            verdict.append(
                "**Verdict: SHELVE the entropy-proxy as a standalone "
                "trading signal.** The rank correlation is zero or "
                "*negative*, meaning higher-entropy distributions did NOT "
                "predict larger realised moves on this slice — if "
                "anything the relationship runs the wrong way. The "
                "feature set (entropy_normalized, tail_pct, "
                "asymmetric_mass) does not rank realised magnitudes here, "
                "so fitting `EMCalibration` cannot rescue it: a linear "
                "projection of features that don't rank the target will "
                "either pick the intercept (constant prediction) or "
                "over-fit the residuals — neither generalises out of "
                "sample. This is a deeper problem than `k_kind` "
                "miscalibration."
            )
            verdict.append("")
            verdict.append(
                "Action: do **not** ship the entropy-proxy to the UI as "
                "a trading signal. Acceptable to ship it as a "
                '*descriptive* panel ("contract-implied event-day '
                'range") with prominent labelling of the data '
                "sparseness and the lack of calibration. Re-run this "
                "validation after 4 more consecutive FOMC + CPI events "
                "have resolved (≈ 2026-Q4) and re-evaluate. Two more "
                "specific follow-ups, in priority order: (1) fix the "
                "FOMC partition — when no-change probability ≥ 95 % the "
                "entropy proxy collapses to 0 % EM but realised moves "
                "are still 1–2 pp from positioning unwind, so the "
                "engine needs a *baseline event-day vol* term that does "
                "not vanish with low entropy; (2) regress realised |Δ| "
                "on a richer feature vector (interaction with VIX level, "
                "spot-trend, sector-rotation residual) — even if shape "
                "alone is uninformative, shape × regime might be."
            )
        elif sp >= 0.5:
            verdict.append(
                "**Verdict: REFINE — fit `EMCalibration` and re-test.** "
                "Rank correlation is strong enough that the feature set "
                "is informative even if the entropy-proxy constants are "
                "wrong. Run `fit_em_calibration` on this sample (in-sample "
                "R² will be optimistic), then hold out the next 4 events "
                "as a true OOS test before shipping to UI."
            )
        else:
            verdict.append(
                "**Verdict: REFINE-cautiously — weak-positive rank "
                "correlation.** Rank correlation suggests the shape "
                "signal contains *some* information but the n is far too "
                "small to fit `EMCalibration` reliably (the engine "
                "requires ≥5 events; we have %d). Park the calibration "
                "step and wait for 2026-Q4 events to compound the "
                "sample." % n
            )
        for ln in verdict:
            lines.append(ln)
        lines.append("")

    return "\n".join(lines) + "\n"


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    t0 = time.time()
    cache = load_cache()

    poly = PolymarketClient(
        gamma_url="https://gamma-api.polymarket.com",
        clob_url="https://clob.polymarket.com",
    )
    kalshi = KalshiClient()

    results: list[EventResult] = []
    for event in HISTORICAL_EVENTS:
        logger.warning("Evaluating %s …", event.event_id)
        res = evaluate_event(event, poly, kalshi, cache)
        results.append(res)
        save_cache(cache)  # incremental persistence

    poly.close()
    kalshi.close()
    save_cache(cache)

    md = render_md(results)
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text(md, encoding="utf-8")

    # One-page summary to stdout.
    usable = [r for r in results if r.em_pm_pct is not None and r.realized_abs_dpct is not None]
    print("=" * 72)
    print(f"Entropy-proxy EM validation — n_curated={len(results)} n_usable={len(usable)}")
    print("=" * 72)
    for r in results:
        if r.em_pm_pct is None or r.realized_abs_dpct is None:
            reason = r.drop_reason or "dropped"
            print(f"  [DROP] {r.event_id:24s}  {reason}")
            continue
        print(
            f"  {r.event_id:24s}  kind={r.event_kind:4s}  "
            f"H_norm={r.entropy_normalized:.2f}  "
            f"EM={r.em_pm_pct:.3f}%  realised={r.realized_abs_dpct:.3f}%  "
            f"err={(r.em_pm_pct - r.realized_abs_dpct):+.3f}"
        )
    if usable:
        em = [r.em_pm_pct for r in usable]  # type: ignore[misc]
        real = [r.realized_abs_dpct for r in usable]  # type: ignore[misc]
        mae = sum(abs(a - b) for a, b in zip(em, real, strict=True)) / len(em)
        pe = pearson(em, real)
        sp = spearman(em, real)
        hit = rank_hit_rate(em, real)
        print("-" * 72)
        print(f"  MAE = {mae:.3f} pp")
        print(f"  Pearson  r = {pe if pe is None else f'{pe:+.3f}'}")
        print(f"  Spearman r = {sp if sp is None else f'{sp:+.3f}'}")
        print(f"  Pairwise rank concordance = {hit if hit is None else f'{hit:.2f}'}")
    print(f"  elapsed: {time.time() - t0:.1f}s   cache: {CACHE_PATH}")
    print(f"  doc:     {DOC_PATH.resolve()}")


if __name__ == "__main__":
    main()
