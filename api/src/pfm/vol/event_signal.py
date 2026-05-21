"""Live event-driven EM signal — composes B1 (engine) + B2 (calendar).

This is the wiring layer between the curated event calendar
(:mod:`pfm.vol.event_calendar`) and the pure-math EM engine
(:mod:`pfm.vol.event_vol_engine`). It fetches current outcome
probabilities from Polymarket (Gamma) and Kalshi (markets endpoint /
candlesticks) and projects them into an :class:`EventSignal`.

Wiring conventions
------------------
* **Polymarket** — we go through the supplied :class:`PolymarketClient`'s
  underlying ``httpx.Client`` (``polymarket_client._client``) to fetch
  the raw Gamma market dict via :func:`pfm.terminal.fetch_gamma_market`.
  ``MarketMetadata`` does not expose live ``bestBid`` / ``bestAsk``, so a
  raw fetch is required. Midpoint = ``(bestBid + bestAsk) / 2`` when both
  are present, else whichever one is available, else ``lastTradePrice``.

* **Kalshi** — we use :meth:`KalshiClient.get_candlesticks` (1-day
  window) and take the most recent bar's ``(yes_bid + yes_ask) / 2``.
  This matches the convention in :mod:`pfm.arb_scanner._kalshi_mid`.

Per-slug failures are caught and turned into ``fetch_failed:{slug}``
warnings; the slug is dropped from the resulting outcome list. The
:attr:`EventSignal.fetch_completeness` ratio surfaces how much of the
declared partition actually returned a live price.

Thin-book degradation
---------------------
When ``fetch_completeness < 0.5`` we deliberately switch to the
entropy-proxy mode (even if a calibration was provided), because a
linear projection on half-a-distribution is structurally untrustworthy.
We also append ``thin_book`` to the forecast warnings.

Out of scope (v1)
-----------------
* ``options_em_pct`` always comes back ``None``. Wiring real options
  IV data is a separate workstream; the gap metric is included in the
  schema so downstream consumers can integrate it later without a
  schema break.
* ``calibration`` defaults to ``None``. Callers pass their own
  :class:`EMCalibration` once a per-(kind, ticker) fit exists.
"""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from pfm.vol.event_calendar import (
    EventEntry,
    OutcomeSlug,
    get_event,
    list_upcoming,
)
from pfm.vol.event_vol_engine import (
    EMCalibration,
    EventDistribution,
    EventEMForecast,
    Outcome,
    distribution_features,
    expected_move_from_distribution,
    normalize_outcomes,
)

if TYPE_CHECKING:
    from pfm.sources.kalshi import KalshiClient
    from pfm.sources.polymarket import PolymarketClient

logger = logging.getLogger(__name__)

# Polymarket Gamma base URL. We don't bother making this configurable
# here because the existing :class:`PolymarketClient` already carries
# its own ``gamma_url`` and we read it off the supplied client.
_DEFAULT_GAMMA_URL = "https://gamma-api.polymarket.com"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class EventSignal(BaseModel):
    """Live EM forecast for one calendar event."""

    model_config = ConfigDict(extra="forbid")

    event_id: str
    event_kind: str
    underlying_ticker: str
    scheduled_at_utc: datetime
    distribution: EventDistribution
    forecast: EventEMForecast
    options_em_pct: float | None = None
    em_gap_pct: float | None = None
    fetch_completeness: float = Field(..., ge=0.0, le=1.0)
    as_of_utc: datetime
    warnings: list[str]


# ---------------------------------------------------------------------------
# Per-venue midpoint fetchers
# ---------------------------------------------------------------------------


def _safe_float(value: Any) -> float | None:
    """Cast ``value`` to ``float`` defensively. Return ``None`` on failure."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    # NaN / inf are useless as probabilities — reject them.
    if not math.isfinite(f):
        return None
    return f


def _polymarket_midpoint(
    slug: str,
    polymarket_client: PolymarketClient,
) -> float | None:
    """Fetch current YES-leg midpoint for a Polymarket ``slug``.

    Returns ``None`` if the market is missing, the response is malformed,
    or neither bid/ask nor ``lastTradePrice`` is available. Errors are
    swallowed (and logged at INFO) so the per-slug failure mode is
    "drop with warning" at the caller level.
    """
    # Use the lazy import-friendly helper from pfm.terminal which knows
    # how to handle the closed=true fallback for resolved markets.
    from pfm.terminal import fetch_gamma_market

    gamma_url = getattr(polymarket_client, "gamma_url", _DEFAULT_GAMMA_URL)
    http = getattr(polymarket_client, "_client", None)
    if http is None:
        logger.info("polymarket client has no httpx client; cannot fetch %s", slug)
        return None
    try:
        market = fetch_gamma_market(http, gamma_url, slug)
    except (LookupError, httpx.HTTPError) as exc:
        logger.info("polymarket fetch failed for %s: %s", slug, exc)
        return None

    bb = _safe_float(market.get("bestBid"))
    ba = _safe_float(market.get("bestAsk"))
    if bb is not None and ba is not None:
        return (bb + ba) / 2.0
    if bb is not None:
        return bb
    if ba is not None:
        return ba
    return _safe_float(market.get("lastTradePrice"))


def _kalshi_midpoint(
    ticker: str,
    kalshi_client: KalshiClient,
) -> float | None:
    """Fetch current YES-leg midpoint for a Kalshi market ``ticker``.

    We pull a one-week candlestick window and take the most recent bar's
    ``(yes_bid + yes_ask) / 2``. Mirrors :func:`pfm.arb_scanner._kalshi_mid`.
    """
    try:
        end = datetime.now(tz=UTC)
        end_ts = int(end.timestamp())
        start_ts = end_ts - 86_400 * 7
        df = kalshi_client.get_candlesticks(ticker, start_ts=start_ts, end_ts=end_ts)
    except Exception as exc:
        # Any upstream failure (HTTP, parse, rate-limit, …) is downgraded
        # to a per-slug miss — the caller's warnings list captures it.
        logger.info("kalshi fetch failed for %s: %s", ticker, exc)
        return None
    if df.empty:
        return None
    try:
        last = df.iloc[-1]
        bid = _safe_float(last["yes_bid"])
        ask = _safe_float(last["yes_ask"])
    except (KeyError, IndexError):
        return None
    if bid is not None and ask is not None:
        return (bid + ask) / 2.0
    if bid is not None:
        return bid
    if ask is not None:
        return ask
    return None


# ---------------------------------------------------------------------------
# Outcome-probability fetching
# ---------------------------------------------------------------------------


def fetch_outcome_probabilities(
    event: EventEntry,
    *,
    polymarket_client: PolymarketClient,
    kalshi_client: KalshiClient | None = None,
    http: httpx.Client | None = None,
) -> tuple[list[Outcome], float, list[str]]:
    """Fetch live midpoint probabilities for every ``OutcomeSlug`` in ``event``.

    Returns:
        Tuple ``(outcomes, completeness, warnings)`` where

        * ``outcomes`` is the list of successfully-fetched
          :class:`Outcome` items (slugs that failed are dropped),
        * ``completeness`` is ``n_successful / n_total`` in ``[0, 1]``,
        * ``warnings`` is a list of human-readable tags
          (e.g. ``"fetch_failed:cpi-yoy-2.8"`` or
          ``"kalshi_client_missing"``).

    Per-slug failures are caught and reported as warnings rather than
    bubbling up, so a partial book still produces a usable signal
    (degraded via the ``fetch_completeness`` field).

    Args:
        event: Curated event entry. Slugs are fetched in declared order.
        polymarket_client: Required — even Kalshi-only events keep this
            in their signature for symmetry.
        kalshi_client: Optional. ``None`` means Kalshi-venue slugs are
            skipped with a ``"kalshi_client_missing"`` warning.
        http: Reserved for future fetcher overrides; currently unused.
    """
    # ``http`` is reserved for future fetcher overrides (e.g. a shared
    # async client); currently the venue-specific helpers carry their own
    # connection state and we don't need to thread one through.
    del http
    outcomes: list[Outcome] = []
    warnings: list[str] = []
    total = len(event.outcome_slugs)
    successful = 0
    kalshi_missing_warned = False

    for slug_def in event.outcome_slugs:
        prob = _fetch_one_outcome(
            slug_def,
            polymarket_client=polymarket_client,
            kalshi_client=kalshi_client,
        )
        if prob is None and slug_def.venue == "kalshi" and kalshi_client is None:
            if not kalshi_missing_warned:
                warnings.append("kalshi_client_missing")
                kalshi_missing_warned = True
            continue
        if prob is None:
            warnings.append(f"fetch_failed:{slug_def.slug}")
            continue
        # Clip to [0, 1] defensively — bestBid/bestAsk are well-behaved
        # in normal conditions but we don't want a malformed upstream
        # value to leak into the engine.
        clipped = max(0.0, min(1.0, float(prob)))
        outcomes.append(
            Outcome(
                label=slug_def.label,
                probability=clipped,
                anchor_value=float(slug_def.anchor_value),
            )
        )
        successful += 1

    completeness = (successful / total) if total > 0 else 0.0
    return outcomes, completeness, warnings


def _fetch_one_outcome(
    slug_def: OutcomeSlug,
    *,
    polymarket_client: PolymarketClient,
    kalshi_client: KalshiClient | None,
) -> float | None:
    """Single-outcome dispatch by venue. Returns midpoint or ``None``."""
    if slug_def.venue == "polymarket":
        return _polymarket_midpoint(slug_def.slug, polymarket_client)
    if slug_def.venue == "kalshi":
        if kalshi_client is None:
            return None
        return _kalshi_midpoint(slug_def.slug, kalshi_client)
    # Unknown venue — treat as a per-slug failure.
    logger.warning("unknown venue %r for slug %s", slug_def.venue, slug_def.slug)
    return None


# ---------------------------------------------------------------------------
# Full signal composition
# ---------------------------------------------------------------------------


def compute_event_signal(
    event_id: str,
    *,
    polymarket_client: PolymarketClient,
    kalshi_client: KalshiClient | None = None,
    calibration: EMCalibration | None = None,
    http: httpx.Client | None = None,
) -> EventSignal:
    """Compose B1 + B2 into a live :class:`EventSignal`.

    Pipeline:
        1. Look up ``event_id`` in the curated calendar.
        2. Fetch outcome midpoints from each venue.
        3. Build an :class:`EventDistribution` and defensively normalise.
        4. Run :func:`expected_move_from_distribution` with the optional
           calibration.
        5. Bundle into :class:`EventSignal`. ``options_em_pct`` stays
           ``None`` (v1).

    Thin-book fallback: if ``fetch_completeness < 0.5`` we re-run the
    forecast with ``calibration=None`` and append ``"thin_book"`` to
    the warnings — a half-empty distribution is structurally unsafe
    for a linear projection regardless of historical R².

    Raises:
        KeyError: ``event_id`` is not in the calendar.
        ValueError: zero outcomes survived the fetch step (impossible
            to build a distribution).
    """
    event = get_event(event_id)
    if event is None:
        raise KeyError(f"unknown event_id: {event_id!r}")

    fetched_outcomes, completeness, warnings = fetch_outcome_probabilities(
        event,
        polymarket_client=polymarket_client,
        kalshi_client=kalshi_client,
        http=http,
    )

    if not fetched_outcomes:
        raise ValueError(
            f"compute_event_signal: no outcomes fetched for {event_id!r} (warnings={warnings!r})"
        )

    # Defensive normalisation. Bare ``normalize_outcomes`` raises when the
    # pre-normalisation mass is < 0.5, which is exactly the "thin book"
    # condition we already detect via completeness — we let the engine
    # handle that path internally and just hand it the raw probs.
    raw_dist = EventDistribution(
        event_id=event.event_id,
        event_kind=event.event_kind,
        underlying_ticker=event.underlying_ticker,
        scheduled_at_utc=event.scheduled_at_utc,
        outcomes=fetched_outcomes,
    )

    # Try to normalise for downstream consumers — but tolerate the
    # too-thin-book failure mode by keeping the raw probs and letting
    # the engine emit ``normalize_failed:...`` itself.
    try:
        normalised_list = normalize_outcomes(fetched_outcomes)
        distribution = raw_dist.model_copy(update={"outcomes": normalised_list})
    except ValueError as exc:
        warnings.append(f"normalize_failed:{exc}")
        distribution = raw_dist

    thin_book = completeness < 0.5

    # When the book is thin, force entropy-proxy mode regardless of
    # whether a calibration was supplied — the linear projection on
    # half a partition is not trustworthy.
    if thin_book:
        forecast = expected_move_from_distribution(distribution, calibration=None)
        forecast_warnings = list(forecast.warnings)
        if "thin_book" not in forecast_warnings:
            forecast_warnings.append("thin_book")
        # Forecast is a frozen Pydantic — clone with updated warnings.
        forecast = forecast.model_copy(update={"warnings": forecast_warnings})
    else:
        forecast = expected_move_from_distribution(distribution, calibration=calibration)

    # Re-emit distribution_features on the post-normalisation dist to
    # keep them consistent with what the engine actually used. The
    # engine already does this internally; we just rely on its output.
    _ = distribution_features  # keep import lint-clean for future expansion

    options_em_pct: float | None = None
    em_gap_pct: float | None = (
        forecast.em_pct - options_em_pct if options_em_pct is not None else None
    )

    return EventSignal(
        event_id=event.event_id,
        event_kind=event.event_kind,
        underlying_ticker=event.underlying_ticker,
        scheduled_at_utc=event.scheduled_at_utc,
        distribution=distribution,
        forecast=forecast,
        options_em_pct=options_em_pct,
        em_gap_pct=em_gap_pct,
        fetch_completeness=completeness,
        as_of_utc=datetime.now(tz=UTC),
        warnings=warnings,
    )


def compute_all_upcoming_signals(
    *,
    now_utc: datetime,
    lookahead_days: int = 30,
    polymarket_client: PolymarketClient,
    kalshi_client: KalshiClient | None = None,
    http: httpx.Client | None = None,
) -> list[EventSignal]:
    """Apply :func:`compute_event_signal` to every upcoming event.

    Per-event failures (bad slug, total fetch failure, unexpected
    upstream error) are caught and logged; the returned list is the
    partial set of signals that succeeded, in calendar order.
    """
    events = list_upcoming(now_utc, lookahead_days=lookahead_days)
    signals: list[EventSignal] = []
    for event in events:
        try:
            sig = compute_event_signal(
                event.event_id,
                polymarket_client=polymarket_client,
                kalshi_client=kalshi_client,
                http=http,
            )
        except (KeyError, ValueError) as exc:
            # Expected per-event failures: keep going.
            logger.info(
                "compute_all_upcoming_signals: skipping %s (%s)",
                event.event_id,
                exc,
            )
            continue
        except Exception as exc:
            # Surface the cause but stay live — one bad event must not
            # take out the whole upcoming-signal sweep.
            logger.warning(
                "compute_all_upcoming_signals: unexpected failure on %s: %s",
                event.event_id,
                exc,
            )
            continue
        signals.append(sig)
    return signals


__all__ = [
    "EventSignal",
    "compute_all_upcoming_signals",
    "compute_event_signal",
    "fetch_outcome_probabilities",
]
