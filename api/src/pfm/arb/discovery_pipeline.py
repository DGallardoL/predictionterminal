"""Step-by-step, unlimited, newest-first cross-venue arb discovery orchestrator.

This module ties together the three discovery building blocks:

* :mod:`pfm.arb.market_crawler` — resumable, newest-first crawl of the full
  Kalshi + Polymarket market universe (one bounded step per call).
* :mod:`pfm.arb.discovery_matcher` — recall-plus-precision matcher that pairs
  candidate markets across venues and FP-gates them via ``score_match``.
* :mod:`pfm.arb.confirmed_store` — durable, unbounded store of every pair where
  an arbitrage was actually observed.

Three modes are supported:

* ``mode="sweep"`` — load the persisted checkpoint, do **one** bounded crawl
  step (``max_pages`` pages per venue), advance + save the checkpoint, and
  return. The *next* call resumes exactly where this one stopped, so over many
  cycles the orchestrator covers an effectively unlimited universe without ever
  blocking on a single long sweep. When a venue reports ``done`` the checkpoint
  resets that venue to begin a fresh newest-first sweep on the next cycle.
* ``mode="new"`` — explore freshly listed markets, but match each venue's NEW
  events against the OTHER venue's BROAD/LIQUID universe (not just the other
  venue's *new* events, which rarely overlap the same day). Concretely: pull
  the NEW events on each venue (``new_kalshi_events`` / ``new_poly_events``,
  ephemeral-filtered) AND crawl a small liquid counterparty universe (Kalshi
  events via ``crawl_kalshi_events`` ephemeral-filtered; Polymarket markets via
  ``crawl_poly_by_volume``). We then match ``new_k × (univ_p ∪ new_p)`` and
  ``new_p × (univ_k ∪ new_k)``, deduplicate the candidates by
  ``(kalshi_ticker, poly_slug)`` keeping the highest score, and tag each with a
  ``new_side`` field (``"kalshi"`` / ``"poly"`` / ``"both"``) recording which
  side(s) were freshly listed. By construction at least one side is always new.
  Bounded + recall-first; the checkpoint is not advanced in this mode.
* ``mode="liquid"`` — substantive/liquid coverage: crawl Polymarket *markets*
  highest-volume-first (``crawl_poly_by_volume``) so discovery covers the
  politics/macro/long-dated universe where real cross-venue arbs live, instead
  of the ephemeral sports/crypto flood that dominates the newest-first feed (and
  starves the substantive universe down to ~0-13 events/page after filtering).
  The volume-sorted poly side is NOT ephemeral-filtered (a high-volume market is
  by construction not a templated 5m series). Checkpoint untouched.

Recall-first
------------
Per the firm directive, discovery is RECALL over precision: ``run_discovery_step``
surfaces **every** matched candidate clearing ``recall_floor`` (default 0.25),
each tagged with its ``tier`` + matcher ``reject_reason`` + a ``confidence`` flag
(``"verified"`` when all hard gates pass and ``score >= min_score``, else
``"review"``). ``min_score`` only drives the ``n_high`` count and which
candidates get priced/recorded — it is NOT a visibility filter. Only hard-gate
rejects (jurisdiction / threshold / window) are dropped.

Arb detection is **optional**. When a ``price_fn`` is supplied it is called for
each surviving candidate to fetch current prices; a simple two-leg arb cost is
computed and, when profitable, recorded in the store. With no ``price_fn`` the
step is discovery-only (``n_recorded == 0``).

Mockability
-----------
Every call into the crawler, matcher and store is routed through a
module-level wrapper (``_crawl_kalshi``, ``_crawl_poly``, ``_new_kalshi``,
``_new_poly``, ``_match_markets``, ``_summarize``, plus the checkpoint helpers).
Tests monkeypatch those module attributes — e.g.
``monkeypatch.setattr(discovery_pipeline, "_crawl_kalshi", fake)`` — to stay
fully offline. The pipeline never imports the underlying functions by value at
call time; it always dereferences the module-level wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from pfm.arb import confirmed_store as _store_mod
from pfm.arb import discovery_matcher as _matcher_mod
from pfm.arb import market_crawler as _crawler_mod
from pfm.arb.confirmed_store import ConfirmedArbStore

__all__ = [
    "CONFIDENCE_REVIEW",
    "CONFIDENCE_VERIFIED",
    "DEFAULT_CHECKPOINT_PATH",
    "MAX_PRICED_CANDIDATES",
    "RECALL_FLOOR",
    "TOP_CANDIDATES",
    "DiscoveryStepResult",
    "PriceFn",
    "default_store",
    "run_discovery_step",
]

#: Default checkpoint location (shared with the crawler).
DEFAULT_CHECKPOINT_PATH = _crawler_mod.DEFAULT_CHECKPOINT_PATH

#: How many top candidates to surface (as dicts) for the UI.
TOP_CANDIDATES = 25

#: Cap on how many candidates we run ``price_fn`` against per step.
MAX_PRICED_CANDIDATES = 50

#: Recall-first floor. Per the firm directive ("prefiero ver falsos positivos a
#: que no vea reales" — recall over precision), discovery surfaces EVERY matched
#: candidate that clears this LOW score, tagged by confidence/tier, rather than
#: dropping anything below ``min_score``. ``min_score`` then only drives the
#: ``n_high`` count, not a filter. Hard-gate rejects (jurisdiction / threshold /
#: window mismatch) are still dropped — they are cross-venue impossibilities.
RECALL_FLOOR = 0.25

#: Confidence labels attached to every surfaced candidate so the UI can flag
#: low-confidence pairs (⚠) without hiding them.
CONFIDENCE_VERIFIED = "verified"
CONFIDENCE_REVIEW = "review"


class PriceFn(Protocol):
    """Callable that fetches current prices for a candidate pair.

    Implementations return a mapping with the two-leg ask/price quotes, or
    ``None`` when prices are unavailable. Recognised keys:

    * ``kalshi_yes_ask`` / ``poly_no_price`` — the "Kalshi-YES + Poly-NO" leg.
    * ``kalshi_no_ask`` / ``poly_yes_price`` — the "Kalshi-NO + Poly-YES" leg.
    """

    def __call__(
        self, kalshi_ticker: str, poly_slug: str
    ) -> dict[str, Any] | None: ...  # pragma: no cover - structural protocol


# ---------------------------------------------------------------------------
# Module-level wrappers — the seams tests monkeypatch.
# ---------------------------------------------------------------------------


def _crawl_kalshi(*, cursor: str | None, max_pages: int, session: Any):
    """Wrapper around :func:`market_crawler.crawl_kalshi_events`.

    The KALSHI side crawls *events* (not raw markets): events carry the real
    human ``title``/``sub_title`` and the nested ``markets[]`` we price against,
    whereas the markets endpoint exposes only templated/garbage titles.
    """
    return _crawler_mod.crawl_kalshi_events(cursor=cursor, max_pages=max_pages, session=session)


def _crawl_poly(*, offset: int, max_pages: int, session: Any):
    """Wrapper around :func:`market_crawler.crawl_poly_events`."""
    return _crawler_mod.crawl_poly_events(offset=offset, max_pages=max_pages, session=session)


def _crawl_poly_volume(*, offset: int, max_pages: int, session: Any):
    """Wrapper around :func:`market_crawler.crawl_poly_by_volume`.

    The ``"liquid"`` mode crawls Polymarket *markets* highest-volume-first to
    cover the substantive politics/macro/long-dated universe where real
    cross-venue arbs live, instead of the ephemeral sports/crypto flood the
    newest-first feed surfaces.
    """
    return _crawler_mod.crawl_poly_by_volume(offset=offset, max_pages=max_pages, session=session)


def _new_kalshi(*, within_hours: float, session: Any, now: datetime | None):
    """Wrapper around :func:`market_crawler.new_kalshi_events`.

    Returns event-level dicts (with real ``title`` + nested ``markets[]``) with
    ephemeral templated series already filtered out.
    """
    return _crawler_mod.new_kalshi_events(within_hours=within_hours, session=session, now=now)


def _new_poly(*, within_hours: float, session: Any, now: datetime | None):
    """Wrapper around :func:`market_crawler.new_poly_events`.

    ``new_poly_events`` already applies the ephemeral filter internally.
    """
    return _crawler_mod.new_poly_events(within_hours=within_hours, session=session, now=now)


def _load_checkpoint(path: str):
    """Wrapper around :func:`market_crawler.load_checkpoint`."""
    return _crawler_mod.load_checkpoint(path)


def _save_checkpoint(path: str, ckpt: Any) -> None:
    """Wrapper around :func:`market_crawler.save_checkpoint`."""
    _crawler_mod.save_checkpoint(path, ckpt)


def _advance_checkpoint(ckpt: Any, *, kalshi_page: Any, poly_page: Any):
    """Wrapper around :func:`market_crawler.advance_checkpoint`."""
    return _crawler_mod.advance_checkpoint(ckpt, kalshi_page=kalshi_page, poly_page=poly_page)


def _match_markets(
    kalshi_items: list[dict[str, Any]],
    poly_items: list[dict[str, Any]],
    *,
    min_score: float,
    keep_soft_rejects: bool = False,
):
    """Wrapper around :func:`discovery_matcher.match_markets`."""
    return _matcher_mod.match_markets(
        kalshi_items,
        poly_items,
        min_score=min_score,
        keep_soft_rejects=keep_soft_rejects,
    )


def _summarize(cands: list[Any]) -> dict[str, Any]:
    """Wrapper around :func:`discovery_matcher.summarize`."""
    return _matcher_mod.summarize(cands)


def _is_ephemeral(text: str) -> bool:
    """Wrapper around :func:`market_crawler.is_ephemeral_market`."""
    return _crawler_mod.is_ephemeral_market(text)


def _event_text(event: dict[str, Any]) -> str:
    """Human-facing text of a Kalshi event for the ephemeral filter."""
    return _crawler_mod._event_text(event)


def _poly_text(event: dict[str, Any]) -> str:
    """Human-facing text of a Polymarket event for the ephemeral filter."""
    return _crawler_mod._poly_event_text(event)


# ---------------------------------------------------------------------------
# Kalshi event -> matcher payload normalization.
# ---------------------------------------------------------------------------


def _representative_market_ticker(event: dict[str, Any]) -> str:
    """Pick the market ticker to price for a Kalshi event.

    The events endpoint identifies an event by ``event_ticker`` but pricing is
    per *market*; this picks a representative nested market's ``ticker`` (the
    most recently opened one, matching the freshness signal), falling back to
    the ``event_ticker`` when there are no nested markets.
    """
    best_ticker = ""
    best_ts = float("-inf")
    for m in event.get("markets") or []:
        if not isinstance(m, dict):
            continue
        ticker = str(m.get("ticker") or "").strip()
        if not ticker:
            continue
        dt = _crawler_mod._market_open_dt(m)
        ts = dt.timestamp() if dt is not None else float("-inf")
        if ts >= best_ts:
            best_ts = ts
            best_ticker = ticker
    if best_ticker:
        return best_ticker
    return str(event.get("event_ticker") or event.get("ticker") or "").strip()


def _event_to_kalshi_item(event: dict[str, Any]) -> dict[str, Any]:
    """Normalize a Kalshi *event* into a matcher payload with a real title.

    The matcher reads ``title``/``ticker``; events expose the real human title
    plus ``sub_title`` (good for the description signal) and ``event_ticker``.
    A representative nested-market ticker is injected as ``ticker`` so the
    pricing layer can fetch quotes for the matched market.
    """
    item = dict(event)
    item.setdefault("title", event.get("title") or "")
    item["ticker"] = _representative_market_ticker(event)
    sub = event.get("sub_title") or event.get("subtitle")
    if sub and not item.get("description"):
        item["description"] = sub
    return item


# ---------------------------------------------------------------------------
# New-mode matching: NEW events on each venue vs the OTHER venue's broad
# liquid universe (so a freshly-listed market finds a counterpart that is not
# itself new). Recall-first + bounded + deduplicated, with a ``new_side`` tag.
# ---------------------------------------------------------------------------


#: How many counterparty universe pages to crawl per venue in ``mode="new"``.
#: Kept small so new-event exploration stays fast; callers can lower it.
NEW_MODE_UNIV_MAX_PAGES = 2


def _poly_id(poly_item: dict[str, Any]) -> str:
    """Best-effort Polymarket identity (slug) for dedupe + ``new_side`` tagging."""
    for key in ("slug", "poly_slug"):
        val = poly_item.get(key)
        if val:
            return str(val).strip()
    return ""


def _kalshi_id(kalshi_item: dict[str, Any]) -> str:
    """Best-effort Kalshi identity (representative ticker) for dedupe + tagging."""
    for key in ("ticker", "kalshi_ticker"):
        val = kalshi_item.get(key)
        if val:
            return str(val).strip()
    return ""


def _match_new_against_universe(
    *,
    new_k: list[dict[str, Any]],
    new_p: list[dict[str, Any]],
    univ_k: list[dict[str, Any]],
    univ_p: list[dict[str, Any]],
    recall_floor: float,
) -> list[Any]:
    """Match NEW events against the OTHER venue's broad universe (recall-first).

    Runs two recall-first matches — ``new_k × (univ_p ∪ new_p)`` and
    ``new_p × (univ_k ∪ new_k)`` — then deduplicates the resulting candidates by
    ``(kalshi_ticker, poly_slug)`` keeping the highest score, and tags each kept
    candidate with ``new_side`` (``"kalshi"`` / ``"poly"`` / ``"both"`` / ``None``)
    according to which side's identity is present in ``new_k`` / ``new_p``.

    Args:
        new_k: NEW Kalshi items (already normalised matcher payloads).
        new_p: NEW Polymarket items.
        univ_k: Broad/liquid Kalshi universe items (matcher payloads).
        univ_p: Broad/liquid Polymarket universe items.
        recall_floor: Low score floor passed to the matcher (keep soft rejects).

    Returns:
        Deduplicated candidates (best score per pair), each with ``new_side`` set.
    """
    new_k_ids = {_kalshi_id(k) for k in new_k if _kalshi_id(k)}
    new_p_ids = {_poly_id(p) for p in new_p if _poly_id(p)}

    # Counterparty universes include the other venue's new events too, so a pair
    # of two freshly-listed markets is still discoverable (-> new_side="both").
    poly_universe = _dedupe_poly(new_p + univ_p)
    kalshi_universe = _dedupe_kalshi(new_k + univ_k)

    raw: list[Any] = []
    if new_k:
        raw.extend(
            _match_markets(new_k, poly_universe, min_score=recall_floor, keep_soft_rejects=True)
        )
    if new_p:
        raw.extend(
            _match_markets(kalshi_universe, new_p, min_score=recall_floor, keep_soft_rejects=True)
        )

    # Dedupe by (kalshi_ticker, poly_slug), keeping the highest-scoring instance.
    best: dict[tuple[str, str], Any] = {}
    for cand in raw:
        key = (cand.kalshi_ticker, cand.poly_slug)
        existing = best.get(key)
        if existing is None or cand.score > existing.score:
            best[key] = cand

    deduped = list(best.values())
    for cand in deduped:
        k_new = cand.kalshi_ticker in new_k_ids
        p_new = cand.poly_slug in new_p_ids
        if k_new and p_new:
            cand.new_side = "both"
        elif k_new:
            cand.new_side = "kalshi"
        elif p_new:
            cand.new_side = "poly"
        else:
            cand.new_side = None

    deduped.sort(key=lambda c: c.score, reverse=True)
    return deduped


def _dedupe_poly(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Dedupe Polymarket items by slug, keeping the first occurrence."""
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for it in items:
        pid = _poly_id(it)
        key = pid or id(it)  # un-slugged items stay distinct
        if key in seen:
            continue
        seen.add(key)  # type: ignore[arg-type]
        out.append(it)
    return out


def _dedupe_kalshi(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Dedupe Kalshi matcher payloads by representative ticker (first wins)."""
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for it in items:
        kid = _kalshi_id(it)
        key = kid or id(it)
        if key in seen:
            continue
        seen.add(key)  # type: ignore[arg-type]
        out.append(it)
    return out


# ---------------------------------------------------------------------------
# Result dataclass.
# ---------------------------------------------------------------------------


@dataclass
class DiscoveryStepResult:
    """Outcome of a single discovery step.

    Recall-first (firm directive: surface false positives rather than miss real
    ones): ``n_candidates`` and ``candidates`` now include EVERY matched pair
    that cleared the ``recall_floor`` — both ``"verified"`` and ``"review"``
    confidence — so nothing plausible is silently dropped. ``n_high`` still
    counts only the high-confidence (``score >= min_score``) subset.

    Attributes:
        n_kalshi: Number of Kalshi markets pulled this step.
        n_poly: Number of Polymarket markets/events pulled this step.
        n_candidates: Number of surfaced candidate pairs (verified + review)
            that cleared ``recall_floor``. Hard-gate rejects are excluded.
        n_high: Number of surfaced candidates with ``score >= min_score``
            (the ``"high"`` count; ``min_score`` is a count threshold, not a
            filter).
        n_recorded: Number of arbs recorded into the store this step.
        mode: ``"sweep"``, ``"new"`` or ``"liquid"``.
        checkpoint: The (advanced/persisted) checkpoint as a plain dict.
        summary: The matcher tier/reject summary (see ``discovery_matcher``).
        candidates: Top-N candidates as plain dicts for the UI, each carrying a
            ``confidence`` flag (``"verified"`` / ``"review"``) plus its
            ``tier`` and ``reject_reason``. In ``mode="new"`` each candidate dict
            additionally carries ``new_side`` (``"kalshi"`` / ``"poly"`` /
            ``"both"`` / ``None``) recording which venue side was freshly listed.
        n_review: Number of surfaced candidates flagged ``"review"`` (kept for
            eyeballing, not auto-trusted). Backward-compatible additive field.
        recall_floor: The low score floor used to retain candidates this step.
    """

    n_kalshi: int
    n_poly: int
    n_candidates: int
    n_high: int
    n_recorded: int
    mode: str
    checkpoint: dict[str, Any]
    summary: dict[str, Any]
    candidates: list[dict[str, Any]] = field(default_factory=list)
    n_review: int = 0
    recall_floor: float = RECALL_FLOOR

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the result."""
        return {
            "n_kalshi": self.n_kalshi,
            "n_poly": self.n_poly,
            "n_candidates": self.n_candidates,
            "n_high": self.n_high,
            "n_recorded": self.n_recorded,
            "mode": self.mode,
            "checkpoint": dict(self.checkpoint),
            "summary": dict(self.summary),
            "candidates": [dict(c) for c in self.candidates],
            "n_review": self.n_review,
            "recall_floor": self.recall_floor,
        }


# ---------------------------------------------------------------------------
# Store helper.
# ---------------------------------------------------------------------------


def default_store() -> ConfirmedArbStore:
    """Return a :class:`ConfirmedArbStore` at the default/env-configured path.

    Honors the ``PFM_ARB_CONFIRMED_STORE`` env var via the store's own
    resolution; passing ``None`` defers entirely to that logic.

    Returns:
        A fresh :class:`ConfirmedArbStore`.
    """
    return _store_mod.ConfirmedArbStore(path=None)


# ---------------------------------------------------------------------------
# Arb detection.
# ---------------------------------------------------------------------------


def _arb_cost(prices: dict[str, Any]) -> float | None:
    """Compute the cheaper two-leg arb cost from a price quote, or ``None``.

    The two complementary legs are:

    * ``kalshi_yes_ask + poly_no_price`` — buy YES on Kalshi, NO on Polymarket.
    * ``kalshi_no_ask + poly_yes_price`` — buy NO on Kalshi, YES on Polymarket.

    A cost below ``1.0`` on either leg is an arbitrage (guaranteed payout of 1
    for a combined stake below 1). Returns the minimum available leg cost, or
    ``None`` when neither leg is fully quoted.

    Args:
        prices: Mapping of leg quotes (see :class:`PriceFn`).

    Returns:
        The minimum leg cost across the two complementary legs, or ``None``.
    """
    legs: list[float] = []
    yes_ask = _as_float(prices.get("kalshi_yes_ask"))
    no_price = _as_float(prices.get("poly_no_price"))
    if yes_ask is not None and no_price is not None:
        legs.append(yes_ask + no_price)

    no_ask = _as_float(prices.get("kalshi_no_ask"))
    yes_price = _as_float(prices.get("poly_yes_price"))
    if no_ask is not None and yes_price is not None:
        legs.append(no_ask + yes_price)

    if not legs:
        return None
    return min(legs)


def _as_float(value: Any) -> float | None:
    """Best-effort float coercion; ``None``/garbage -> ``None``."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _detect_and_record(
    candidates: list[Any],
    *,
    price_fn: PriceFn,
    store: ConfirmedArbStore,
    now: datetime | None,
) -> int:
    """Price each candidate (best-first, capped) and record any arbs found.

    Args:
        candidates: Non-rejected candidates, best-first.
        price_fn: Price fetcher (see :class:`PriceFn`).
        store: Store to upsert confirmed arbs into.
        now: Optional timestamp override (forwarded to ``store.record``).

    Returns:
        The number of arbs recorded this step.
    """
    recorded = 0
    for cand in candidates[:MAX_PRICED_CANDIDATES]:
        ticker = cand.kalshi_ticker
        slug = cand.poly_slug
        prices = price_fn(ticker, slug)
        if not prices:
            continue
        cost = _arb_cost(prices)
        if cost is None or cost >= 1.0:
            continue
        profit_pct = (1.0 - cost) * 100.0
        store.record(
            arb_key=f"{ticker}|{slug}",
            kalshi_ticker=ticker,
            poly_slug=slug,
            profit_pct=profit_pct,
            confidence=cand.tier,
            now=now,
        )
        recorded += 1
    return recorded


# ---------------------------------------------------------------------------
# Confidence flagging (recall-first).
# ---------------------------------------------------------------------------


def _confidence_for(cand: Any, min_score: float) -> str:
    """Return the confidence flag for a surfaced candidate.

    ``"verified"`` when the matcher cleared every hard gate (outcome/threshold/
    window/same-venue — i.e. the candidate is *not* rejected) AND its score is at
    or above ``min_score`` (the high-confidence bar). Everything else surfaced —
    soft (``low_score``) rejects, and below-``min_score`` borderline pairs — is
    ``"review"`` so the UI shows ⚠ without hiding the candidate.
    """
    if not getattr(cand, "rejected", False) and cand.score >= min_score:
        return CONFIDENCE_VERIFIED
    return CONFIDENCE_REVIEW


def _candidate_dict(cand: Any, min_score: float) -> dict[str, Any]:
    """Candidate as a UI dict with its recall-first ``confidence`` flag added."""
    d = cand.as_dict()
    d["confidence"] = _confidence_for(cand, min_score)
    return d


# ---------------------------------------------------------------------------
# Orchestrator.
# ---------------------------------------------------------------------------


def run_discovery_step(
    *,
    checkpoint_path: str = DEFAULT_CHECKPOINT_PATH,
    store: ConfirmedArbStore | None = None,
    mode: str = "sweep",
    max_pages: int = 3,
    within_hours: float = 24.0,
    min_score: float = 0.5,
    recall_floor: float = RECALL_FLOOR,
    price_fn: PriceFn | None = None,
    session: Any = None,
    now: datetime | None = None,
) -> DiscoveryStepResult:
    """Run one discovery step over the cross-venue universe (recall-first).

    Per the firm directive ("prefiero ver falsos positivos a que no vea reales")
    this surfaces **every** matched candidate that clears ``recall_floor`` —
    nothing plausible is silently dropped. Each candidate is tagged with a
    ``tier`` (high/borderline/reject), the matcher ``reject_reason``, and a
    ``confidence`` flag (``"verified"`` / ``"review"``) so the UI can flag low
    confidence without hiding it. ``min_score`` is now a count threshold (drives
    ``n_high`` and which candidates we price+record), **not** a visibility filter.
    Only hard-gate rejects (jurisdiction / threshold / window mismatch — genuine
    cross-venue impossibilities) are excluded.

    Modes:

    * ``"sweep"`` — load the checkpoint, crawl one bounded newest-first step of
      both venues, advance + persist the checkpoint so the next call resumes.
    * ``"new"`` — explore freshly listed markets: pull the NEW events opened
      within ``within_hours`` on each venue (ephemeral series dropped) AND a
      small liquid counterparty universe, then match each venue's new events
      against the OTHER venue's broad universe (``new_k × (univ_p ∪ new_p)`` and
      ``new_p × (univ_k ∪ new_k)``), dedupe by ``(kalshi_ticker, poly_slug)``,
      and tag every candidate with ``new_side``. The checkpoint is untouched.
    * ``"liquid"`` — substantive/liquid coverage: crawl Polymarket *markets*
      highest-volume-first (``crawl_poly_by_volume``) so discovery covers the
      politics/macro/long-dated universe where real cross-venue arbs live,
      instead of the ephemeral sports/crypto flood the newest-first feed
      surfaces. Kalshi is still crawled newest-first (its events feed is the
      substantive one). The checkpoint is untouched (volume order is stable, not
      a resumable position).

    Args:
        checkpoint_path: Path to the resumable crawl checkpoint (sweep mode).
        store: Confirmed-arb store; defaults to :func:`default_store` when
            ``None`` and a ``price_fn`` is supplied.
        mode: ``"sweep"``, ``"new"`` or ``"liquid"``.
        max_pages: Pages per venue per step (sweep / liquid modes).
        within_hours: Freshness window in hours (new mode).
        min_score: High-confidence bar — drives the ``n_high`` count and which
            candidates get priced/recorded. NOT a visibility filter.
        recall_floor: Low floor below which a candidate is not even surfaced
            (default :data:`RECALL_FLOOR`). Set lower to surface even more.
        price_fn: Optional price fetcher enabling arb detection + recording.
        session: Optional HTTP session forwarded to the crawler.
        now: Optional timestamp override (for deterministic tests).

    Returns:
        A :class:`DiscoveryStepResult`.

    Raises:
        ValueError: If ``mode`` is not ``"sweep"``, ``"new"`` or ``"liquid"``.
    """
    if mode not in ("sweep", "new", "liquid"):
        raise ValueError(f"mode must be 'sweep', 'new' or 'liquid', got {mode!r}")

    precomputed_candidates: list[Any] | None = None

    if mode == "new":
        # Explore freshly listed markets, but match each venue's NEW events
        # against the OTHER venue's BROAD/LIQUID universe — the two venues'
        # same-day new listings rarely overlap, so new×new returns ~0. We pull
        # the NEW events on each venue AND a small liquid counterparty universe,
        # then match new_k × (univ_p ∪ new_p) and new_p × (univ_k ∪ new_k),
        # dedupe, and tag each candidate's ``new_side``.
        #
        # ``new_kalshi_events`` returns event dicts (real titles, ephemeral
        # series already dropped); ``new_poly_events`` filters ephemerals too.
        new_k_events = _new_kalshi(within_hours=within_hours, session=session, now=now)
        new_p_items = _new_poly(within_hours=within_hours, session=session, now=now)
        new_k_items = [_event_to_kalshi_item(e) for e in new_k_events]

        # Bounded liquid counterparty universes. Kalshi events are ephemeral-
        # filtered; the volume-sorted poly side is not (a high-volume market is
        # by construction not an ephemeral templated series).
        univ_pages = min(max_pages, NEW_MODE_UNIV_MAX_PAGES)
        univ_k_page = _crawl_kalshi(cursor=None, max_pages=univ_pages, session=session)
        univ_p_page = _crawl_poly_volume(offset=0, max_pages=univ_pages, session=session)
        univ_k_items = [
            _event_to_kalshi_item(e)
            for e in univ_k_page.events
            if not _is_ephemeral(_event_text(e))
        ]
        univ_p_items = list(univ_p_page.events)

        precomputed_candidates = _match_new_against_universe(
            new_k=new_k_items,
            new_p=new_p_items,
            univ_k=univ_k_items,
            univ_p=univ_p_items,
            recall_floor=recall_floor,
        )

        # n_kalshi / n_poly report the NEW-event counts (the freshness signal
        # this mode is about); the broad universe is the matched-against pool.
        kalshi_items = new_k_items
        poly_items = new_p_items
        checkpoint_dict: dict[str, Any] = {}
    elif mode == "liquid":
        # Substantive coverage: Kalshi events (newest-first, real titles) +
        # Polymarket markets sorted highest-volume-first. We deliberately do
        # NOT apply the ephemeral filter to the volume-sorted poly side — a
        # high-volume market is by construction not an ephemeral templated
        # series, and filtering risks dropping a liquid sports-final outright.
        # The checkpoint is untouched (volume order is stable, not resumable).
        kalshi_page = _crawl_kalshi(cursor=None, max_pages=max_pages, session=session)
        poly_page = _crawl_poly_volume(offset=0, max_pages=max_pages, session=session)
        kalshi_events = [e for e in kalshi_page.events if not _is_ephemeral(_event_text(e))]
        kalshi_items = [_event_to_kalshi_item(e) for e in kalshi_events]
        poly_items = list(poly_page.events)
        checkpoint_dict = {}
    else:
        ckpt = _load_checkpoint(checkpoint_path)
        # KALSHI side now crawls EVENTS (real titles + nested markets).
        kalshi_page = _crawl_kalshi(cursor=ckpt.kalshi_cursor, max_pages=max_pages, session=session)
        poly_page = _crawl_poly(offset=ckpt.poly_offset, max_pages=max_pages, session=session)
        kalshi_events = [e for e in kalshi_page.events if not _is_ephemeral(_event_text(e))]
        kalshi_items = [_event_to_kalshi_item(e) for e in kalshi_events]
        poly_items = [e for e in poly_page.events if not _is_ephemeral(_poly_text(e))]

        advanced = _advance_checkpoint(ckpt, kalshi_page=kalshi_page, poly_page=poly_page)
        _save_checkpoint(checkpoint_path, advanced)
        checkpoint_dict = {
            "kalshi_cursor": advanced.kalshi_cursor,
            "poly_offset": advanced.poly_offset,
            "last_seen_poly_start_iso": advanced.last_seen_poly_start_iso,
        }

    # Recall-first: keep every candidate above ``recall_floor`` (incl. soft
    # ``low_score`` rejects), tagged by tier/confidence. Only hard-gate rejects
    # are dropped inside the matcher. In ``mode="new"`` the candidates were
    # already produced by ``_match_new_against_universe`` (new × broad-universe,
    # deduped, ``new_side``-tagged); other modes match here.
    if precomputed_candidates is not None:
        all_candidates = precomputed_candidates
    else:
        all_candidates = _match_markets(
            kalshi_items,
            poly_items,
            min_score=recall_floor,
            keep_soft_rejects=True,
        )
    summary = _summarize(all_candidates)

    # ``kept`` = everything surfaced: non-rejected pairs AND soft (``low_score``)
    # rejects. Hard-gate rejects (jurisdiction / threshold / window / same-venue)
    # are dropped — they are genuine cross-venue impossibilities, never
    # low-confidence guesses. (The matcher already drops them in recall-first
    # mode; this is a defensive guard so a stub/raw list can't slip one through.)
    # ``min_score`` is a count threshold for n_high, NOT a visibility filter.
    kept = [c for c in all_candidates if not c.rejected or c.reject_reason == "low_score"]
    n_high = sum(1 for c in kept if not c.rejected and c.score >= min_score)
    n_review = sum(1 for c in kept if _confidence_for(c, min_score) == CONFIDENCE_REVIEW)

    n_recorded = 0
    if price_fn is not None:
        # Only price/record VERIFIED candidates: recording a low-confidence
        # mismatch as a confirmed arb would poison the durable store. The
        # review-confidence pairs are still surfaced for the human to eyeball.
        verified = [c for c in kept if _confidence_for(c, min_score) == CONFIDENCE_VERIFIED]
        active_store = store if store is not None else default_store()
        n_recorded = _detect_and_record(verified, price_fn=price_fn, store=active_store, now=now)

    top = [_candidate_dict(c, min_score) for c in kept[:TOP_CANDIDATES]]

    return DiscoveryStepResult(
        n_kalshi=len(kalshi_items),
        n_poly=len(poly_items),
        n_candidates=len(kept),
        n_high=n_high,
        n_recorded=n_recorded,
        mode=mode,
        checkpoint=checkpoint_dict,
        summary=summary,
        candidates=top,
        n_review=n_review,
        recall_floor=recall_floor,
    )
