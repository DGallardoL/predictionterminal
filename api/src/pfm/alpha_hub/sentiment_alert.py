"""Sentiment-regression alert detector.

Task W11-54 (T54). When *enough* of the recent
``GET /terminal/jumps/{slug}/backtest`` runs land on the
**"DISAGREES IS REAL ALPHA"** verdict, that is itself a meta-signal:
across the curated leaderboard of markets, the contrarian "fade-the-jump"
trade is durably profitable right now. This module turns that observation
into a single alert object the rest of the platform (e.g.
``GET /alerts/digest``) can surface.

Trigger rule (per task spec)
----------------------------
* ``market_count = len(backtest_results)`` — the number of backtest
  payloads we considered.
* ``disagrees_count`` — how many of those payloads carried the
  "DISAGREES IS REAL ALPHA" verdict.
* Fire iff **both**::

      market_count >= min_markets             # default 5, strict ≥
      disagrees_count / market_count > threshold_pct / 100   # default 40 %, strict >

The strict ``>`` on the percentage means "exactly at the threshold does
not fire" — this matters because the spec calls out "above 40 % of >5
markets" and a wave of agents doing 5×40 %=2 fires would be a false-alarm
storm.

The detector is a *pure function* over a list of dicts so unit tests can
build synthetic backtest payloads without any I/O. The
``collect_recent_backtests`` async helper is the I/O-binding seam: it
reads ``app.state.warm_jumps`` (populated by the lifespan prewarm) and
calls the backtest endpoint for the warmest ``top_n`` slugs.

Verdict-text matching
---------------------
The backtest endpoint embeds its verdict as a substring of the
``interpretation`` field. The canonical winning verdict starts with
``"DISAGREES IS REAL ALPHA"`` — see
:mod:`pfm.terminal.jumps_backtest`. We do a case-insensitive substring
match so future copy tweaks (e.g. lowercase, additional context) keep
working. The task also uses the slugified form
``"disagrees-is-real-alpha"``; we accept both.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

#: Default minimum number of backtest results required before any
#: percentage is taken seriously. Below this we always return ``None``
#: regardless of ratio — small samples are statistically meaningless.
DEFAULT_MIN_MARKETS: int = 5

#: Default percentage of "DISAGREES IS REAL ALPHA" verdicts that must
#: be **strictly exceeded** before the alert fires.
DEFAULT_THRESHOLD_PCT: float = 40.0

#: Default fan-out for :func:`collect_recent_backtests`.
DEFAULT_TOP_N: int = 20

#: Per-slug timeout in seconds when collecting backtests, so a single
#: stuck Polymarket call cannot stall the whole sweep.
PER_SLUG_TIMEOUT_S: float = 25.0

#: Concurrency cap for the parallel backtest fan-out.
DEFAULT_CONCURRENCY: int = 4

#: Substrings (case-folded) we look for in ``interpretation`` to flag a
#: "disagrees is real alpha" verdict. Stored as a tuple so the match is
#: cheap and additions don't require code changes elsewhere.
_DISAGREES_VERDICT_MARKERS: tuple[str, ...] = (
    "disagrees is real alpha",
    "disagrees-is-real-alpha",
)


# ─────────────────────────────────────────────────────────────────────────────
# Alert dataclass
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SentimentRegressionAlert:
    """Snapshot of a sentiment-regression alert at a moment in time.

    Frozen so consumers can stash it in sets / use it as a dict key and
    so we can never mutate a published alert by accident.

    The fields mirror the task spec; ``threshold_pct`` and
    ``min_markets`` are carried alongside the trigger values so a
    downstream renderer can show "12/20 = 60 % vs 40 % threshold"
    without re-deriving anything.
    """

    triggered_at: datetime
    market_count: int
    disagrees_count: int
    disagrees_pct: float
    threshold_pct: float = DEFAULT_THRESHOLD_PCT
    min_markets: int = DEFAULT_MIN_MARKETS
    slugs: tuple[str, ...] = field(default_factory=tuple)

    def to_alert_row(self) -> dict[str, Any]:
        """Render as a ``/alerts/digest``-compatible row.

        The digest router expects rows with ``kind``, ``severity``,
        and a label/identifier. We bucket severity by how far past
        the threshold we are: ≥2× threshold → "high",
        ≥1.25× → "med", else "low".
        """
        ratio = (self.disagrees_pct / self.threshold_pct) if self.threshold_pct > 0 else 1.0
        if ratio >= 2.0:
            severity = "high"
        elif ratio >= 1.25:
            severity = "med"
        else:
            severity = "low"
        return {
            "kind": "sentiment-regression",
            "severity": severity,
            "label": (
                f"{self.disagrees_count}/{self.market_count} markets "
                f"({self.disagrees_pct:.1f}%) flagged DISAGREES-IS-REAL-ALPHA"
            ),
            "market_count": self.market_count,
            "disagrees_count": self.disagrees_count,
            "disagrees_pct": self.disagrees_pct,
            "threshold_pct": self.threshold_pct,
            "triggered_at": self.triggered_at.isoformat().replace("+00:00", "Z"),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Verdict matching
# ─────────────────────────────────────────────────────────────────────────────


def _is_disagrees_verdict(result: dict[str, Any]) -> bool:
    """Return ``True`` iff a backtest result carries the disagrees verdict.

    We tolerate both the field name ``interpretation`` (the live API
    surface) and the convenience field ``verdict`` (used in unit-test
    fixtures and in some downstream summarisers). Either may be missing
    or non-string; in that case we silently return ``False`` so the
    caller can keep counting.
    """
    if not isinstance(result, dict):
        return False
    haystacks: list[str] = []
    for key in ("verdict", "interpretation"):
        v = result.get(key)
        if isinstance(v, str) and v:
            haystacks.append(v.casefold())
    if not haystacks:
        return False
    for hay in haystacks:
        for marker in _DISAGREES_VERDICT_MARKERS:
            if marker in hay:
                return True
    return False


def _percentage(num: int, denom: int) -> float:
    """``num / denom * 100`` rounded to one decimal place; safe at denom=0."""
    if denom <= 0:
        return 0.0
    # round(..., 1) keeps the output stable for tests and humans; the
    # raw float is preserved upstream in the dataclass via the ratio,
    # so no precision is lost for downstream comparisons.
    return round(num / denom * 100.0, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Pure detector
# ─────────────────────────────────────────────────────────────────────────────


def check_sentiment_regression(
    backtest_results: list[dict[str, Any]] | None,
    *,
    threshold_pct: float = DEFAULT_THRESHOLD_PCT,
    min_markets: int = DEFAULT_MIN_MARKETS,
    now: datetime | None = None,
) -> SentimentRegressionAlert | None:
    """Detect a sentiment-regression alert from a batch of backtest payloads.

    Args:
        backtest_results: list of backtest response dicts, as returned
            by ``GET /terminal/jumps/{slug}/backtest`` (or its dict-form
            equivalent). May be empty / ``None`` — both yield ``None``.
        threshold_pct: percentage of "DISAGREES IS REAL ALPHA" verdicts
            that must be **strictly exceeded** to fire. Default 40.
        min_markets: minimum number of markets required in the sample.
            Below this we always return ``None``, regardless of ratio.
        now: clock injection for tests. Defaults to ``datetime.now(UTC)``.

    Returns:
        :class:`SentimentRegressionAlert` when the trigger condition is
        met; ``None`` otherwise.
    """
    if not backtest_results:
        return None
    market_count = len(backtest_results)
    if market_count < int(min_markets):
        # Sample size too small — no alert regardless of ratio.
        return None

    disagrees_count = 0
    slugs: list[str] = []
    for r in backtest_results:
        if _is_disagrees_verdict(r):
            disagrees_count += 1
            # Best-effort slug capture — useful in the digest row.
            slug = r.get("slug") if isinstance(r, dict) else None
            if isinstance(slug, str) and slug:
                slugs.append(slug)

    # Strict greater-than: "exactly at threshold" must not fire.
    pct = _percentage(disagrees_count, market_count)
    raw_ratio = disagrees_count / market_count * 100.0
    if raw_ratio <= float(threshold_pct):
        return None

    triggered = now or datetime.now(UTC)
    return SentimentRegressionAlert(
        triggered_at=triggered,
        market_count=market_count,
        disagrees_count=disagrees_count,
        disagrees_pct=pct,
        threshold_pct=float(threshold_pct),
        min_markets=int(min_markets),
        slugs=tuple(slugs),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Async collector (I/O seam)
# ─────────────────────────────────────────────────────────────────────────────


def _pick_slugs(app: Any, top_n: int) -> list[str]:
    """Choose which slugs to backtest.

    Preference order:
      1. ``app.state.warm_jumps['slugs']`` (the lifespan prewarm output)
         — these are the freshest, lowest-latency targets.
      2. :data:`pfm.terminal.jumps_prewarm.CURATED_TOP_SLUGS` — the
         demo-list fallback.
      3. ``[]`` — degraded mode; the caller will get an empty result
         list and ``check_sentiment_regression`` will return ``None``.
    """
    n = max(1, int(top_n))
    warm = None
    try:
        warm = getattr(app.state, "warm_jumps", None)
    except Exception:
        warm = None
    if isinstance(warm, dict):
        inner = warm.get("slugs") if isinstance(warm.get("slugs"), dict) else warm
        if isinstance(inner, dict) and inner:
            # Newest / fastest first — sort by elapsed asc (smaller = healthier).
            sorted_slugs = sorted(
                (k for k in inner if isinstance(k, str)),
                key=lambda k: float(inner.get(k) or 0.0),
            )
            if sorted_slugs:
                return sorted_slugs[:n]
    try:
        from pfm.terminal.jumps_prewarm import CURATED_TOP_SLUGS

        return list(CURATED_TOP_SLUGS)[:n]
    except Exception:
        return []


async def _one_backtest(
    app: Any,
    slug: str,
    *,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any] | None:
    """Best-effort single-slug backtest call.

    Returns the result dict on success, ``None`` on any failure. The
    function is deliberately silent on errors because the aggregator
    only cares about the verdict population — one failing slug is fine.
    """
    from pfm.sources.polymarket import PolymarketClient
    from pfm.terminal.jumps_backtest import get_jumps_backtest

    poly: PolymarketClient | None = getattr(app.state, "poly", None)
    if poly is None:
        return None
    async with semaphore:
        try:
            payload = await asyncio.wait_for(
                # The endpoint coroutine accepts a Request, but its body
                # only reaches into ``request.app.state.poly`` via the
                # ``_get_polymarket_client`` dependency — which we
                # bypass by passing ``poly`` explicitly. The other
                # required argument is ``request``; we don't need a
                # real one because the function only uses it for the
                # poly dep, so we pass a minimal stand-in.
                get_jumps_backtest(  # type: ignore[call-arg]
                    request=_FakeRequest(app),
                    slug=slug,
                    poly=poly,
                ),
                timeout=PER_SLUG_TIMEOUT_S,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            logger.debug("sentiment_alert: backtest failed for %s: %s", slug, exc)
            return None
    # ``get_jumps_backtest`` returns a Pydantic model; surface a dict
    # so downstream code stays I/O-shape agnostic.
    try:
        return payload.model_dump()  # type: ignore[union-attr]
    except AttributeError:
        return payload if isinstance(payload, dict) else None


class _FakeRequest:
    """Minimal Request-shaped stand-in for fan-out calls.

    The backtest endpoint only reads ``request.app.state.poly`` (via
    its Depends-injected ``poly`` arg, which we supply directly). A
    full ``starlette.requests.Request`` object isn't worth constructing
    here — we just need ``request.app`` to exist.
    """

    def __init__(self, app: Any) -> None:
        self.app = app


async def collect_recent_backtests(
    app: Any,
    top_n: int = DEFAULT_TOP_N,
    *,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> list[dict[str, Any]]:
    """Fan out backtest calls across the top-N warmest / curated slugs.

    The returned list contains only the dicts we successfully retrieved
    — failed calls (timeouts, missing polymarket client, upstream 5xx)
    are dropped silently. The caller is :func:`check_sentiment_regression`,
    which already treats short lists as no-alert.

    Args:
        app: FastAPI app whose ``state.poly`` and ``state.warm_jumps``
            drive both slug selection and the underlying fetch.
        top_n: cap on how many slugs to query. Default 20.
        concurrency: max in-flight backtests. Default 4 — matches the
            jumps prewarm's concurrency budget so we don't double up on
            outbound bandwidth when both run concurrently.

    Returns:
        List of backtest result dicts, length ≤ ``top_n``.
    """
    slugs = _pick_slugs(app, top_n)
    if not slugs:
        return []
    semaphore = asyncio.Semaphore(max(1, int(concurrency)))
    try:
        gathered = await asyncio.gather(
            *(_one_backtest(app, s, semaphore=semaphore) for s in slugs),
            return_exceptions=True,
        )
    except asyncio.CancelledError:
        raise
    out: list[dict[str, Any]] = []
    for item in gathered:
        if isinstance(item, BaseException):
            if isinstance(item, asyncio.CancelledError):
                raise item
            continue
        if isinstance(item, dict):
            out.append(item)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Digest contributor (lazy import seam for /alerts/digest)
# ─────────────────────────────────────────────────────────────────────────────


async def build_digest_rows(app: Any, *, top_n: int = DEFAULT_TOP_N) -> list[dict[str, Any]]:
    """Convenience seam for ``/alerts/digest`` integration.

    Runs :func:`collect_recent_backtests` + :func:`check_sentiment_regression`
    and returns a list with at most one alert row (the digest builder
    splays this across its bucket map). Returns ``[]`` when no alert
    fires or any upstream is unavailable — the digest must never 5xx
    on us.
    """
    try:
        results = await collect_recent_backtests(app, top_n=top_n)
    except Exception:
        return []
    alert = check_sentiment_regression(results)
    if alert is None:
        return []
    return [alert.to_alert_row()]


__all__ = [
    "DEFAULT_CONCURRENCY",
    "DEFAULT_MIN_MARKETS",
    "DEFAULT_THRESHOLD_PCT",
    "DEFAULT_TOP_N",
    "PER_SLUG_TIMEOUT_S",
    "SentimentRegressionAlert",
    "build_digest_rows",
    "check_sentiment_regression",
    "collect_recent_backtests",
]
