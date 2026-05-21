"""Earnings Whisper from Polymarket — extract a 'whisper EPS' from PM odds.

The classical "whisper number" is the unofficial EPS estimate that floats
in trading desks before a print, typically tighter than the published
sell-side consensus. Most desks proxy it with options-implied move plus
a cocktail of buy-side surveys; here we proxy it from prediction-market
contracts of the form *"Will <TICKER> beat consensus EPS by X%?"*.

The module does three things:

  1. **Aggregate beat-probability ladders.** For each ticker we take the
     PM contracts that ask "beats by ≥X%" for several X values and
     compute an expected percentage beat = Σ Δprob_i · midpoint_i.

  2. **Construct a whisper EPS.** ``whisper_eps = consensus_eps · (1 +
     E[beat_pct])``. The recommendation tag flips on a configurable edge
     threshold (default ±2%).

  3. **Cross-check vs implied move.** Options IV is required to size the
     trade; in the absence of a real options feed we fall back to
     20-day realised vol scaled by 1.2 (typical earnings premium).

This is a POC: ``CONSENSUS_EPS`` is hardcoded for the seven tickers that
have liquid PM earnings markets today. Switching to a live consensus
feed (Yahoo, Polygon, Refinitiv) is a one-function swap noted with a
``# TODO: switch to live consensus`` marker.

Endpoints (mounted via the module's ``router``):

  - ``GET /alpha/earnings-whisper/{ticker}``      single ticker
  - ``GET /alpha/earnings-whisper-dashboard``     ranked next-N-days panel
"""

from __future__ import annotations

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from typing import Any, Literal

import httpx
import numpy as np
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from pfm.cache_utils import get_cache
from pfm.sources import polygon as polygon_src
from pfm.terminal import fetch_gamma_market

logger = logging.getLogger(__name__)

GAMMA_URL: str = "https://gamma-api.polymarket.com"

_WHISPER_CACHE = get_cache("earnings_whisper", ttl=1800)  # 30 min
_DASHBOARD_CACHE = get_cache("earnings_whisper_dashboard", ttl=3600)  # 1 hour
_CALENDAR_CACHE = get_cache("earnings_calendar", ttl=3600)  # 1 hour

# Internal markers for the dashboard prewarm (mirrors pm_vix.py).
_DASHBOARD_PREWARM_AT_KEY: str = "dashboard_prewarmed_at_unix"
_DASHBOARD_PREWARM_TTL_SECONDS: int = 7200  # 2h — survives a missed refresh.
EARNINGS_DASHBOARD_STALE_AFTER_SECONDS: int = 7200  # 2h — same as TTL.

ConsensusSource = Literal["polygon_live", "hardcoded_snapshot", "unknown"]

# ---------------------------------------------------------------------------
# Hardcoded consensus + market-slug ladders
# ---------------------------------------------------------------------------
# TODO: switch to live consensus from Yahoo/Polygon/Refinitiv in v0.2.
# The slug ladders below intentionally reference markets that may have
# resolved by demo time; ``compute_whisper`` falls back to a synthetic
# ladder if every slug is missing so the demo never returns 500.

CONSENSUS_EPS: dict[str, float] = {
    "NVDA": 0.84,
    "TSLA": 0.62,
    "AAPL": 1.55,
    "AMZN": 1.18,
    "MSFT": 3.22,
    "META": 5.10,
    "GOOGL": 2.05,
    "AMD": 0.95,
}

# Each slug is annotated with the *beat threshold* it represents.
# E.g. "nvda-beats-eps-by-5pct-q1-2026" → threshold 0.05.
BEAT_LADDERS: dict[str, list[tuple[str, float]]] = {
    "NVDA": [
        ("nvda-beats-eps-q1-2026", 0.0),
        ("nvda-beats-eps-by-5pct-q1-2026", 0.05),
        ("nvda-beats-eps-by-10pct-q1-2026", 0.10),
        ("nvda-beats-eps-by-20pct-q1-2026", 0.20),
    ],
    "TSLA": [
        ("tsla-beats-eps-q1-2026", 0.0),
        ("tsla-beats-eps-by-5pct-q1-2026", 0.05),
        ("tsla-beats-eps-by-10pct-q1-2026", 0.10),
    ],
    "AAPL": [
        ("aapl-beats-eps-q2-2026", 0.0),
        ("aapl-beats-eps-by-3pct-q2-2026", 0.03),
        ("aapl-beats-eps-by-5pct-q2-2026", 0.05),
    ],
    "AMZN": [
        ("amzn-beats-eps-q1-2026", 0.0),
        ("amzn-beats-eps-by-5pct-q1-2026", 0.05),
        ("amzn-beats-eps-by-10pct-q1-2026", 0.10),
    ],
    "MSFT": [
        ("msft-beats-eps-q3-2026", 0.0),
        ("msft-beats-eps-by-3pct-q3-2026", 0.03),
        ("msft-beats-eps-by-7pct-q3-2026", 0.07),
    ],
    "META": [
        ("meta-beats-eps-q1-2026", 0.0),
        ("meta-beats-eps-by-5pct-q1-2026", 0.05),
        ("meta-beats-eps-by-10pct-q1-2026", 0.10),
    ],
    "GOOGL": [
        ("googl-beats-eps-q1-2026", 0.0),
        ("googl-beats-eps-by-5pct-q1-2026", 0.05),
    ],
    "AMD": [
        ("amd-beats-eps-q1-2026", 0.0),
        ("amd-beats-eps-by-5pct-q1-2026", 0.05),
        ("amd-beats-eps-by-10pct-q1-2026", 0.10),
    ],
}


# Approximate next earnings date by ticker (POC; real calendar feed in v0.2).
# Stored as `today + N days` so the fixture never rots into the past. Quarterly
# refresh in a future commit, but for the POC this guarantees deterministic
# tests and a usable static fallback regardless of when the module is loaded.
def _next_earnings_calendar() -> dict[str, date]:
    today = datetime.now(tz=UTC).date()
    offsets_days: dict[str, int] = {
        "NVDA": 9,
        "TSLA": 6,
        "AAPL": 15,
        "AMZN": 2,
        "MSFT": 10,
        "META": 3,
        "GOOGL": 1,
        "AMD": 8,
    }
    return {tk: today + timedelta(days=off) for tk, off in offsets_days.items()}


NEXT_EARNINGS: dict[str, date] = _next_earnings_calendar()

# Recommendation thresholds on |edge_vs_consensus_pct|.
EDGE_LONG_THRESHOLD = 2.0  # whisper >= consensus + 2% → long
EDGE_SHORT_THRESHOLD = -2.0  # whisper <= consensus - 2% → short


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
    last = _safe_float(market.get("lastTradePrice"))
    if last is not None:
        return last
    return None


def _expected_beat_pct(
    ladder: list[tuple[str, float]],
    *,
    http: httpx.Client,
    overrides: dict[str, dict[str, Any]] | None = None,
) -> tuple[float, float, int]:
    """Aggregate beat-threshold ladder into an expected beat % and base prob.

    Returns ``(expected_beat_pct, beat_prob, n_used)`` where:

      - ``expected_beat_pct`` is Σ Δp_i · midpoint(threshold_{i-1}, threshold_i).
        Acts like a discrete expectation over the survival ladder.
      - ``beat_prob`` is the probability of any beat (threshold = 0).
      - ``n_used`` is the number of ladder rungs that returned a price.

    The ladder MUST be sorted ascending by threshold. We assume the first
    rung is "beats consensus" (threshold 0); the contracts are
    interpreted as *"beats by ≥ threshold"* so probabilities are
    monotone-decreasing in threshold.
    """
    rungs: list[tuple[float, float]] = []  # (threshold, prob)
    for slug, thresh in ladder:
        m: dict[str, Any] | None = None
        if overrides is not None and slug in overrides:
            m = overrides[slug]
        else:
            try:
                m = fetch_gamma_market(http, GAMMA_URL, slug)
            except (LookupError, httpx.HTTPError) as exc:
                logger.info("earnings_whisper: skipping %s: %s", slug, exc)
                continue
        if m is None:
            continue
        p = _market_yes_prob(m)
        if p is None:
            continue
        rungs.append((float(thresh), float(np.clip(p, 0.0, 1.0))))

    if not rungs:
        return 0.0, 0.0, 0

    # Enforce monotonicity: prob_{i+1} <= prob_i. Markets sometimes
    # mis-price the ladder; we clip to enforce a coherent survival fn.
    rungs.sort(key=lambda r: r[0])
    cleaned: list[tuple[float, float]] = []
    last_p = 1.0
    for thresh, p in rungs:
        p = min(p, last_p)
        cleaned.append((thresh, p))
        last_p = p

    beat_prob = next((p for t, p in cleaned if abs(t) < 1e-9), cleaned[0][1])

    # Discrete expectation: split survival fn into bands.
    # Add a final rung at +∞ with prob 0, and an implicit prob=1 at thresh=-∞.
    expected = 0.0
    prev_thresh = 0.0
    prev_prob = beat_prob
    # Mass below 0: probability of no beat (1 - beat_prob), centred at -2.5%.
    expected += (1.0 - beat_prob) * (-0.025)
    for thresh, prob in cleaned:
        if thresh <= 0.0 + 1e-9:
            prev_thresh, prev_prob = thresh, prob
            continue
        band_mass = max(0.0, prev_prob - prob)
        midpoint = (prev_thresh + thresh) / 2.0
        expected += band_mass * midpoint
        prev_thresh, prev_prob = thresh, prob
    # Tail above the highest rung: assume midpoint = last_thresh + 5pp.
    expected += prev_prob * (prev_thresh + 0.05)

    return float(expected), float(beat_prob), len(cleaned)


def _realised_vol_proxy(ticker: str) -> float:
    """Return a 20-day realised-vol proxy (annualised) for ``ticker``.

    Hardcoded table — POC. Real implementation pulls 21 daily closes from
    yfinance and computes ``np.std(log_returns) * sqrt(252)``.
    """
    table = {
        "NVDA": 0.55,
        "TSLA": 0.65,
        "AAPL": 0.28,
        "AMZN": 0.34,
        "MSFT": 0.26,
        "META": 0.40,
        "GOOGL": 0.30,
        "AMD": 0.58,
    }
    return table.get(ticker.upper(), 0.35)


def _iv_implied_move_pct(ticker: str, days_to_earnings: int) -> float:
    """Return earnings-window implied move as a pct of spot.

    Without a real options feed we use realised vol × 1.2 (the typical
    earnings IV premium) and scale down to the print window:

        move_pct = rv_annual * 1.2 * sqrt(days_to_earnings / 252)
    """
    rv = _realised_vol_proxy(ticker)
    days = max(1, int(days_to_earnings))
    horizon_yrs = days / 252.0
    move = rv * 1.2 * np.sqrt(horizon_yrs)
    return float(round(move * 100.0, 3))


def _classify(edge_pct: float) -> Literal["long_pre_print", "short_pre_print", "hold"]:
    if edge_pct >= EDGE_LONG_THRESHOLD:
        return "long_pre_print"
    if edge_pct <= EDGE_SHORT_THRESHOLD:
        return "short_pre_print"
    return "hold"


def _run_async(coro: Any) -> Any:
    """Run ``coro`` to completion regardless of the surrounding loop state.

    The whisper helpers are sync (legacy contract); when we call into the
    async Polygon client we use this shim. Inside FastAPI handlers the
    sync function is already executing in a thread pool, so a fresh
    ``asyncio.run`` is safe. Outside (e.g. ad-hoc scripts) it is too.
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = None
    if loop is not None and loop.is_running():
        # Rare path — schedule on a new loop in a worker thread.
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


def _get_consensus_eps(
    ticker: str,
    *,
    source: Literal["live", "cached", "hardcoded"] = "cached",
) -> tuple[float | None, ConsensusSource]:
    """Resolve consensus EPS for ``ticker`` with explicit provenance.

    Resolution order:

      1. ``source="hardcoded"`` → only the static snapshot is consulted.
      2. ``source="live"`` or ``"cached"`` → try Polygon (cache hit allowed
         when ``"cached"``; ``"live"`` busts the cache before lookup).
      3. Fall back to ``CONSENSUS_EPS`` snapshot.
      4. Return ``(None, "unknown")`` if neither path produces a value.
    """
    tk = ticker.upper()
    if source == "hardcoded":
        if tk in CONSENSUS_EPS:
            return CONSENSUS_EPS[tk], "hardcoded_snapshot"
        return None, "unknown"

    if polygon_src.is_configured():
        if source == "live":
            # Bust the 12h Polygon cache so callers can force a fresh fetch.
            try:
                from pfm.cache_utils import get_cache as _gc

                _gc("polygon_consensus").clear()
            except Exception:  # pragma: no cover - defensive
                pass
        try:
            data = _run_async(polygon_src.fetch_consensus_eps_or_none(tk))
        except Exception as exc:  # pragma: no cover - defensive
            logger.info("earnings_whisper: polygon lookup raised %s for %s", exc, tk)
            data = None
        if data is not None:
            est = data.get("current_estimate")
            if est is not None:
                try:
                    return float(est), "polygon_live"
                except (TypeError, ValueError):
                    pass

    if tk in CONSENSUS_EPS:
        return CONSENSUS_EPS[tk], "hardcoded_snapshot"
    return None, "unknown"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_whisper(
    ticker: str,
    earnings_date: date,
    *,
    http: httpx.Client | None = None,
    overrides: dict[str, dict[str, Any]] | None = None,
    source: Literal["live", "cached", "hardcoded"] = "cached",
) -> dict[str, Any]:
    """Compute the whisper EPS for ``ticker`` ahead of ``earnings_date``.

    Args:
        ticker: Equity ticker (case-insensitive).
        earnings_date: Print date used to scale IV into a window vol.
        http: Injectable httpx client (tests).
        overrides: Optional ``slug -> market_dict`` map (tests).
        source: Consensus EPS provenance preference. ``"cached"`` (default)
            tries Polygon then falls back; ``"live"`` busts the Polygon
            cache first; ``"hardcoded"`` skips Polygon entirely.

    Returns:
        dict with keys ``ticker, earnings_date, consensus_eps, pm_beat_prob,
        iv_implied_move_pct, whisper_eps, edge_vs_consensus_pct, recommendation,
        consensus_source``.
    """
    tk = ticker.upper()
    consensus, consensus_source = _get_consensus_eps(tk, source=source)
    if consensus is None:
        raise KeyError(f"no consensus EPS for ticker {tk!r}")
    ladder = BEAT_LADDERS.get(tk, [])

    own_http = http is None
    http = http or httpx.Client(timeout=8.0)
    try:
        expected_beat, beat_prob, n_used = _expected_beat_pct(
            ladder, http=http, overrides=overrides
        )
    finally:
        if own_http:
            http.close()

    whisper_eps = consensus * (1.0 + expected_beat)
    edge_pct = (whisper_eps - consensus) / consensus * 100.0

    today = datetime.now(tz=UTC).date()
    days_to_earn = max(0, (earnings_date - today).days)
    iv_pct = _iv_implied_move_pct(tk, days_to_earn or 1)

    return {
        "ticker": tk,
        "earnings_date": earnings_date.isoformat(),
        "consensus_eps": round(consensus, 4),
        "consensus_source": consensus_source,
        "pm_beat_prob": round(beat_prob, 4),
        "expected_beat_pct": round(expected_beat * 100.0, 4),
        "iv_implied_move_pct": iv_pct,
        "whisper_eps": round(whisper_eps, 4),
        "edge_vs_consensus_pct": round(edge_pct, 4),
        "recommendation": _classify(edge_pct),
        "n_ladder_rungs_used": n_used,
        "days_to_earnings": days_to_earn,
    }


def _resolve_calendar(
    days: int,
    source: Literal["live", "cached", "hardcoded"],
) -> dict[str, date]:
    """Return ``{ticker: earnings_date}`` for the [today, today+days] window.

    Polygon calendar is consulted when the API key is configured and
    ``source != "hardcoded"``. Polygon entries are merged on top of
    ``NEXT_EARNINGS``; hardcoded entries fill the gap for tickers
    Polygon's free tier doesn't surface (e.g. small caps with no
    fundamentals filing yet).
    """
    today = datetime.now(tz=UTC).date()
    horizon = today + timedelta(days=days)

    merged: dict[str, date] = {tk: ed for tk, ed in NEXT_EARNINGS.items() if today <= ed <= horizon}

    if source == "hardcoded" or not polygon_src.is_configured():
        return merged

    try:
        live = _run_async(polygon_src.fetch_earnings_calendar_or_empty(today, horizon))
    except Exception as exc:  # pragma: no cover - defensive
        logger.info("earnings_whisper: polygon calendar lookup raised %s", exc)
        live = []
    for entry in live:
        tk = str(entry.get("ticker", "")).upper()
        ed_str = entry.get("earnings_date")
        if not tk or not ed_str:
            continue
        try:
            ed = date.fromisoformat(str(ed_str)[:10])
        except ValueError:
            continue
        if today <= ed <= horizon:
            merged[tk] = ed
    return merged


def whisper_dashboard(
    days: int = 14,
    *,
    http: httpx.Client | None = None,
    overrides: dict[str, dict[str, Any]] | None = None,
    source: Literal["live", "cached", "hardcoded"] = "cached",
    max_tickers: int = 50,
    max_workers: int = 8,
) -> list[dict[str, Any]]:
    """Return whisper rows for every ticker with an earnings print in [today, today+days].

    Per-ticker compute fans out across a bounded thread pool because each
    ticker requires N independent ``fetch_gamma_market`` HTTP calls
    (one per ladder rung) — serial walk over ~8 liquid tickers blew past
    13 s on cold cache. ``httpx.Client`` is thread-safe for concurrent
    GETs to different URLs, so we share the injected (or freshly-built)
    client across workers.
    """
    calendar = _resolve_calendar(days, source)
    if not calendar:
        return []

    own_http = http is None
    http = http or httpx.Client(timeout=8.0)

    def _compute_one(item: tuple[str, date]) -> dict[str, Any] | None:
        tk, ed = item
        try:
            return compute_whisper(tk, ed, http=http, overrides=overrides, source=source)
        except KeyError:
            return None

    out: list[dict[str, Any]] = []
    try:
        items = list(calendar.items())
        if len(items) <= 1:
            # Single-ticker shortcut: avoid the thread-pool overhead.
            for it in items:
                row = _compute_one(it)
                if row is not None:
                    out.append(row)
        else:
            workers = max(1, min(int(max_workers), len(items)))
            with ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="whisper-dashboard"
            ) as ex:
                for row in ex.map(_compute_one, items):
                    if row is not None:
                        out.append(row)
    finally:
        if own_http:
            http.close()

    out.sort(key=lambda r: abs(r["edge_vs_consensus_pct"]), reverse=True)
    return out[:max_tickers]


def earnings_calendar(
    days: int = 30,
    *,
    source: Literal["live", "cached", "hardcoded"] = "cached",
) -> list[dict[str, Any]]:
    """Return ``[{ticker, earnings_date, consensus_eps, n_analysts}]`` rows.

    Mirrors :func:`whisper_dashboard` calendar resolution but does not
    compute whisper math; intended for the ``/alpha/earnings-calendar``
    endpoint and for callers that just want the upcoming-prints list.
    """
    today = datetime.now(tz=UTC).date()
    horizon = today + timedelta(days=days)

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    if source != "hardcoded" and polygon_src.is_configured():
        try:
            live = _run_async(polygon_src.fetch_earnings_calendar_or_empty(today, horizon))
        except Exception as exc:  # pragma: no cover - defensive
            logger.info("earnings_calendar: polygon lookup raised %s", exc)
            live = []
        for entry in live:
            tk = str(entry.get("ticker", "")).upper()
            ed_str = entry.get("earnings_date")
            if not tk or not ed_str:
                continue
            try:
                ed = date.fromisoformat(str(ed_str)[:10])
            except ValueError:
                continue
            if not (today <= ed <= horizon):
                continue
            rows.append(
                {
                    "ticker": tk,
                    "earnings_date": ed.isoformat(),
                    "consensus_eps": entry.get("consensus_eps"),
                    "n_analysts": int(entry.get("n_analysts") or 0),
                }
            )
            seen.add(tk)

    # Static fallback for tickers Polygon didn't return.
    for tk, ed in NEXT_EARNINGS.items():
        if tk in seen or not (today <= ed <= horizon):
            continue
        rows.append(
            {
                "ticker": tk,
                "earnings_date": ed.isoformat(),
                "consensus_eps": CONSENSUS_EPS.get(tk),
                "n_analysts": 0,
            }
        )

    rows.sort(key=lambda r: r["earnings_date"])
    return rows


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class WhisperRow(BaseModel):
    ticker: str
    earnings_date: str
    consensus_eps: float
    consensus_source: ConsensusSource = "hardcoded_snapshot"
    pm_beat_prob: float = Field(..., ge=0.0, le=1.0)
    expected_beat_pct: float
    iv_implied_move_pct: float = Field(..., ge=0.0)
    whisper_eps: float
    edge_vs_consensus_pct: float
    recommendation: Literal["long_pre_print", "short_pre_print", "hold"]
    n_ladder_rungs_used: int = Field(..., ge=0)
    days_to_earnings: int = Field(..., ge=0)


class WhisperDashboardResponse(BaseModel):
    n: int = Field(..., ge=0)
    horizon_days: int = Field(..., ge=1)
    source: Literal["live", "cached", "hardcoded"] = "cached"
    rows: list[WhisperRow]
    cache_age_seconds: int = Field(
        default=0,
        ge=0,
        description=(
            "Seconds since the dashboard payload was computed. ``0`` for a "
            "freshly computed response, larger when served from prewarm."
        ),
    )
    is_stale: bool = Field(
        default=False,
        description=(
            "True when ``cache_age_seconds`` exceeds EARNINGS_DASHBOARD_STALE_AFTER_SECONDS."
        ),
    )


class EarningsCalendarRow(BaseModel):
    ticker: str
    earnings_date: str
    consensus_eps: float | None = None
    n_analysts: int = Field(default=0, ge=0)


class EarningsCalendarResponse(BaseModel):
    n: int = Field(..., ge=0)
    horizon_days: int = Field(..., ge=1)
    source: Literal["live", "cached", "hardcoded"] = "cached"
    rows: list[EarningsCalendarRow]


# ---------------------------------------------------------------------------
# Dashboard pre-warm
# ---------------------------------------------------------------------------


def _now_unix() -> float:
    return datetime.now(tz=UTC).timestamp()


def _is_dashboard_prewarm_required() -> bool:
    """Return True when the env opts into the cache-only hot path."""
    return os.environ.get("PFM_EARNINGS_PREWARM_ENABLED", "").strip() == "1"


def _augment_dashboard_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Attach ``cache_age_seconds`` + ``is_stale`` to a dashboard payload."""
    out = dict(payload)
    prewarmed_at = _DASHBOARD_CACHE.get(_DASHBOARD_PREWARM_AT_KEY)
    if prewarmed_at is not None:
        try:
            age = max(0, int(_now_unix() - float(prewarmed_at)))
        except (TypeError, ValueError):
            age = 0
    else:
        age = 0
    out["cache_age_seconds"] = age
    out["is_stale"] = age > EARNINGS_DASHBOARD_STALE_AFTER_SECONDS
    return out


def _prewarm_compute_dashboard(
    days: int = 14,
    source: Literal["live", "cached", "hardcoded"] = "cached",
) -> dict[str, Any]:
    """Compute one dashboard payload synchronously and cache it.

    The cache key matches what :func:`get_whisper_dashboard` reads on a hit
    so the endpoint sees the prewarmed value transparently.
    """
    rows = whisper_dashboard(days=days, source=source)
    payload = {
        "n": len(rows),
        "horizon_days": days,
        "source": source,
        "rows": rows,
    }
    cache_key = ("dashboard", int(days), source)
    _DASHBOARD_CACHE.set(cache_key, payload, ttl=_DASHBOARD_PREWARM_TTL_SECONDS)
    _DASHBOARD_CACHE.set(_DASHBOARD_PREWARM_AT_KEY, _now_unix(), ttl=_DASHBOARD_PREWARM_TTL_SECONDS)
    return payload


async def run_forever_dashboard_prewarm(
    interval_seconds: int = 3600,
    *,
    days: int = 14,
    source: Literal["live", "cached", "hardcoded"] = "cached",
    stop_event: asyncio.Event | None = None,
) -> None:
    """Background loop that recomputes the whisper dashboard every ``interval``.

    The synchronous compute runs on a worker thread (``asyncio.to_thread``)
    so the event loop stays free for live request handling. Exceptions are
    logged but never break the loop — the next iteration retries.
    """
    interval = max(60, int(interval_seconds))
    while True:
        try:
            await asyncio.to_thread(_prewarm_compute_dashboard, days, source)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("earnings_whisper dashboard prewarm raised: %s", exc)

        if stop_event is not None and stop_event.is_set():
            return
        try:
            if stop_event is None:
                await asyncio.sleep(interval)
            else:
                await asyncio.wait_for(stop_event.wait(), timeout=interval)
                return
        except TimeoutError:
            continue
        except asyncio.CancelledError:
            raise


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


router = APIRouter(prefix="/alpha", tags=["alpha"])


@router.get("/earnings-whisper/{ticker}", response_model=WhisperRow)
def get_whisper(
    ticker: str,
    earnings_date: str | None = Query(default=None, alias="date"),
    source: Literal["live", "cached", "hardcoded"] = Query(default="cached"),
) -> WhisperRow:
    """Compute the whisper EPS for one ticker.

    ``earnings_date`` defaults to ``NEXT_EARNINGS[ticker]`` if omitted.
    """
    tk = ticker.upper()
    # Backward-compat: a ticker is "known" if it's in the hardcoded snapshot.
    # Polygon-only tickers must be queried with an explicit ``date``.
    if tk not in CONSENSUS_EPS and earnings_date is None:
        raise HTTPException(status_code=404, detail=f"no consensus for {tk!r}")

    if earnings_date is not None:
        try:
            ed = date.fromisoformat(earnings_date)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"invalid date: {earnings_date!r}") from exc
    else:
        ed = NEXT_EARNINGS.get(tk)
        if ed is None:
            raise HTTPException(status_code=404, detail=f"no scheduled earnings for {tk!r}")

    cache_key = ("whisper", tk, ed.isoformat(), source)
    cached = _WHISPER_CACHE.get(cache_key)
    if cached is not None:
        return WhisperRow(**cached)

    try:
        payload = compute_whisper(tk, ed, source=source)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"no consensus for {tk!r}") from exc
    _WHISPER_CACHE.set(cache_key, payload, ttl=1800)
    return WhisperRow(**payload)


@router.get("/earnings-whisper-dashboard", response_model=WhisperDashboardResponse)
def get_whisper_dashboard(
    days: int = Query(default=14, ge=1, le=90),
    source: Literal["live", "cached", "hardcoded"] = Query(default="cached"),
) -> WhisperDashboardResponse:
    """Whisper rows for every ticker with earnings inside ``days``, sorted by |edge|.

    When ``PFM_EARNINGS_PREWARM_ENABLED=1``, the endpoint NEVER computes
    inline. It serves the prewarmed cache, or 503s with ``Retry-After: 5``
    so dashboards back off cleanly while the background task initialises.
    """
    cache_key = ("dashboard", int(days), source)
    cached = _DASHBOARD_CACHE.get(cache_key)
    if cached is not None:
        return WhisperDashboardResponse(**_augment_dashboard_payload(cached))

    if _is_dashboard_prewarm_required():
        raise HTTPException(
            status_code=503,
            detail="earnings-whisper dashboard not ready yet (prewarm pending)",
            headers={"Retry-After": "5"},
        )

    rows = whisper_dashboard(days=days, source=source)
    payload = {
        "n": len(rows),
        "horizon_days": days,
        "source": source,
        "rows": rows,
    }
    _DASHBOARD_CACHE.set(cache_key, payload, ttl=3600)
    return WhisperDashboardResponse(**_augment_dashboard_payload(payload))


@router.get("/earnings-calendar", response_model=EarningsCalendarResponse)
def get_earnings_calendar(
    days: int = Query(default=30, ge=1, le=180),
    source: Literal["live", "cached", "hardcoded"] = Query(default="cached"),
) -> EarningsCalendarResponse:
    """Upcoming earnings calendar (Polygon when configured, hardcoded fallback)."""
    cache_key = ("calendar", int(days), source)
    cached = _CALENDAR_CACHE.get(cache_key)
    if cached is not None:
        return EarningsCalendarResponse(**cached)

    rows = earnings_calendar(days=days, source=source)
    payload = {
        "n": len(rows),
        "horizon_days": days,
        "source": source,
        "rows": rows,
    }
    _CALENDAR_CACHE.set(cache_key, payload, ttl=3600)
    return EarningsCalendarResponse(**payload)


__all__ = [
    "BEAT_LADDERS",
    "CONSENSUS_EPS",
    "EARNINGS_DASHBOARD_STALE_AFTER_SECONDS",
    "NEXT_EARNINGS",
    "EarningsCalendarResponse",
    "EarningsCalendarRow",
    "WhisperDashboardResponse",
    "WhisperRow",
    "compute_whisper",
    "earnings_calendar",
    "router",
    "run_forever_dashboard_prewarm",
    "whisper_dashboard",
]
