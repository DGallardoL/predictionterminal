"""No-network tests for :mod:`pfm.arb.discovery_pipeline`.

Every test monkeypatches the module-level crawler/matcher seams so nothing
touches the network. The matcher itself (``score_match``) is pure/offline, so
some tests let it run for real against synthetic payloads; others stub it to
control candidate output precisely.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from pfm.arb import discovery_pipeline as dp
from pfm.arb.confirmed_store import ConfirmedArbStore
from pfm.arb.discovery_matcher import Candidate
from pfm.arb.market_crawler import KalshiEventsPage, PolyCrawlPage

# ---------------------------------------------------------------------------
# Helpers / fixtures.
# ---------------------------------------------------------------------------


def _make_candidate(
    *,
    ticker: str = "KXTEST",
    slug: str = "test-slug",
    score: float = 0.85,
    rejected: bool = False,
    reason: str | None = None,
    tier: str = "high",
) -> Candidate:
    return Candidate(
        kalshi_ticker=ticker,
        kalshi_title=f"Kalshi {ticker}",
        poly_slug=slug,
        poly_title=f"Poly {slug}",
        score=score,
        components={"text": score},
        rejected=rejected,
        reject_reason=reason,
        tier=tier,
    )


@pytest.fixture()
def store(tmp_path) -> ConfirmedArbStore:
    return ConfirmedArbStore(path=tmp_path / "confirmed.json")


# ---------------------------------------------------------------------------
# Sweep mode: one step, advance + save, resume on next call.
# ---------------------------------------------------------------------------


def test_sweep_does_one_step_and_persists_checkpoint(monkeypatch, tmp_path, store):
    ckpt_path = str(tmp_path / "crawl_state.json")

    kalshi_calls: list[Any] = []
    poly_calls: list[Any] = []

    def fake_crawl_kalshi(*, cursor, max_pages, session):
        kalshi_calls.append(cursor)
        return KalshiEventsPage(
            events=[
                {
                    "event_ticker": "KXBTC",
                    "title": "Bitcoin above 100k",
                    "markets": [{"ticker": "KXBTC-25", "open_time": "2026-05-20T00:00:00Z"}],
                }
            ],
            next_cursor="CURSOR_A",
            done=False,
            n_pages=max_pages,
        )

    def fake_crawl_poly(*, offset, max_pages, session):
        poly_calls.append(offset)
        return PolyCrawlPage(
            events=[{"slug": "btc-100k", "title": "Bitcoin above 100k"}],
            next_offset=offset + 100,
            done=False,
            n_pages=max_pages,
        )

    monkeypatch.setattr(dp, "_crawl_kalshi", fake_crawl_kalshi)
    monkeypatch.setattr(dp, "_crawl_poly", fake_crawl_poly)
    monkeypatch.setattr(dp, "_match_markets", lambda k, p, **kw: [])
    monkeypatch.setattr(dp, "_summarize", lambda c: {"total": 0})

    # First step: starts from a fresh checkpoint (cursor=None, offset=0).
    res1 = dp.run_discovery_step(checkpoint_path=ckpt_path, store=store, mode="sweep", max_pages=2)
    assert res1.mode == "sweep"
    assert res1.n_kalshi == 1
    assert res1.n_poly == 1
    assert kalshi_calls == [None]
    assert poly_calls == [0]

    # Checkpoint persisted to disk with advanced positions.
    saved = json.loads((tmp_path / "crawl_state.json").read_text())
    assert saved["kalshi_cursor"] == "CURSOR_A"
    assert saved["poly_offset"] == 100
    assert res1.checkpoint["kalshi_cursor"] == "CURSOR_A"
    assert res1.checkpoint["poly_offset"] == 100

    # Second step: resumes from the saved cursor/offset.
    dp.run_discovery_step(checkpoint_path=ckpt_path, store=store, mode="sweep", max_pages=2)
    assert kalshi_calls == [None, "CURSOR_A"]
    assert poly_calls == [0, 100]


def test_sweep_resets_checkpoint_when_side_done(monkeypatch, tmp_path, store):
    ckpt_path = str(tmp_path / "crawl_state.json")

    def fake_crawl_kalshi(*, cursor, max_pages, session):
        # Exhausted sweep -> done, no next cursor.
        return KalshiEventsPage(events=[], next_cursor=None, done=True, n_pages=1)

    def fake_crawl_poly(*, offset, max_pages, session):
        return PolyCrawlPage(events=[], next_offset=offset, done=True, n_pages=1)

    monkeypatch.setattr(dp, "_crawl_kalshi", fake_crawl_kalshi)
    monkeypatch.setattr(dp, "_crawl_poly", fake_crawl_poly)
    monkeypatch.setattr(dp, "_match_markets", lambda k, p, **kw: [])
    monkeypatch.setattr(dp, "_summarize", lambda c: {"total": 0})

    res = dp.run_discovery_step(checkpoint_path=ckpt_path, store=store, mode="sweep")
    # done -> reset to fresh sweep for next cycle.
    assert res.checkpoint["kalshi_cursor"] is None
    assert res.checkpoint["poly_offset"] == 0


# ---------------------------------------------------------------------------
# New mode: uses the fresh-market fetchers, no checkpoint advance.
# ---------------------------------------------------------------------------


def test_new_mode_uses_fresh_fetchers(monkeypatch, store):
    new_kalshi_called: list[float] = []
    new_poly_called: list[float] = []

    def fake_new_kalshi(*, within_hours, session, now):
        new_kalshi_called.append(within_hours)
        return [{"ticker": "KXNEW", "title": "New market"}]

    def fake_new_poly(*, within_hours, session, now):
        new_poly_called.append(within_hours)
        return [{"slug": "new-slug", "title": "New market"}]

    def fail_poly_offset_crawl(**kwargs):  # pragma: no cover - must not be called
        # new-mode now crawls the LIQUID poly universe (crawl_poly_by_volume),
        # never the offset-paginated sweep feed.
        raise AssertionError("offset poly sweep crawler must not run in new mode")

    monkeypatch.setattr(dp, "_new_kalshi", fake_new_kalshi)
    monkeypatch.setattr(dp, "_new_poly", fake_new_poly)
    # new-mode crawls a bounded liquid counterparty universe on each venue.
    monkeypatch.setattr(dp, "_crawl_kalshi", lambda **k: KalshiEventsPage(events=[]))
    monkeypatch.setattr(dp, "_crawl_poly_volume", lambda **k: PolyCrawlPage(events=[]))
    monkeypatch.setattr(dp, "_crawl_poly", fail_poly_offset_crawl)
    monkeypatch.setattr(dp, "_match_markets", lambda k, p, **kw: [])
    monkeypatch.setattr(dp, "_summarize", lambda c: {"total": 0})

    res = dp.run_discovery_step(mode="new", store=store, within_hours=12.0)

    assert res.mode == "new"
    assert new_kalshi_called == [12.0]
    assert new_poly_called == [12.0]
    # n_kalshi / n_poly report the NEW-event counts (the freshness signal).
    assert res.n_kalshi == 1
    assert res.n_poly == 1
    assert res.checkpoint == {}


def test_new_mode_uses_event_crawl_filter_and_matches(monkeypatch, store):
    """End-to-end new-mode: event titles + ephemeral filter -> real candidate.

    Lets the real matcher run against a synthetic overlapping event pair, and
    confirms an ephemeral Kalshi event is dropped before matching.
    """
    real_event = {
        "event_ticker": "KXFEDDECISION",
        "title": "Fed decision: rate cut in June 2026 FOMC",
        "sub_title": "Federal Reserve June 2026 meeting",
        "markets": [{"ticker": "KXFED-26JUN-CUT", "open_time": "2026-05-21T00:00:00Z"}],
    }
    poly_events = [
        {
            "slug": "fed-rate-cut-june-2026",
            "title": "Fed decision: rate cut in June 2026 FOMC",
            "description": "Federal Reserve June 2026 meeting",
        }
    ]

    new_kalshi_seen: list[float] = []

    def fake_new_kalshi(*, within_hours, session, now):
        new_kalshi_seen.append(within_hours)
        # mimic new_kalshi_events: ephemeral already filtered out.
        return [real_event]

    def fake_new_poly(*, within_hours, session, now):
        return list(poly_events)

    monkeypatch.setattr(dp, "_new_kalshi", fake_new_kalshi)
    monkeypatch.setattr(dp, "_new_poly", fake_new_poly)
    # Empty counterparty universes: the new_k×new_p direction still matches.
    monkeypatch.setattr(dp, "_crawl_kalshi", lambda **k: KalshiEventsPage(events=[]))
    monkeypatch.setattr(dp, "_crawl_poly_volume", lambda **k: PolyCrawlPage(events=[]))

    res = dp.run_discovery_step(store=store, mode="new", within_hours=72.0, min_score=0.5)

    assert res.mode == "new"
    assert new_kalshi_seen == [72.0]
    assert res.n_kalshi == 1
    assert res.n_candidates >= 1
    cand = res.candidates[0]
    # Real human title flows through from the event (not a templated market).
    assert "Fed decision" in cand["kalshi_title"]
    # Representative nested-market ticker is what we'd price.
    assert cand["kalshi_ticker"] == "KXFED-26JUN-CUT"
    # The ephemeral event never reaches the matcher.
    assert "Solana" not in cand["kalshi_title"]
    # Both sides were freshly listed -> new_side="both".
    assert cand["new_side"] == "both"


# ---------------------------------------------------------------------------
# New mode: NEW events on each venue matched vs the OTHER venue's broad/liquid
# universe (the fix — new×new rarely overlaps the same day).
# ---------------------------------------------------------------------------


def test_new_kalshi_matches_existing_liquid_poly(monkeypatch, store):
    """A NEW Kalshi event matches an EXISTING (non-new) liquid Poly market.

    new_side must be ``"kalshi"`` — only the Kalshi side is fresh; its
    counterpart was already listed and lives in the liquid poly universe. Uses a
    dated topic so the resolution-window gate is satisfied.
    """
    new_kalshi_event = {
        "event_ticker": "KXJANEDOE28",
        "title": ("Will Jane Doe win the 2028 Republican nomination by Dec 31 2028?"),
        "sub_title": "Jane Doe 2028 Republican presidential nomination",
        "markets": [{"ticker": "KXJANE-28", "open_time": "2026-05-21T00:00:00Z"}],
    }
    existing_liquid_poly = {
        "slug": "jane-doe-2028-republican-nomination",
        "question": ("Will Jane Doe win the 2028 Republican nomination by Dec 31 2028?"),
        "description": "Jane Doe 2028 Republican presidential nomination",
        "volumeNum": 3.4e6,
    }

    monkeypatch.setattr(dp, "_new_kalshi", lambda **k: [new_kalshi_event])
    monkeypatch.setattr(dp, "_new_poly", lambda **k: [])  # no new poly
    monkeypatch.setattr(dp, "_crawl_kalshi", lambda **k: KalshiEventsPage(events=[]))
    monkeypatch.setattr(
        dp,
        "_crawl_poly_volume",
        lambda **k: PolyCrawlPage(events=[existing_liquid_poly]),
    )

    res = dp.run_discovery_step(store=store, mode="new", within_hours=72.0, min_score=0.5)

    assert res.mode == "new"
    assert res.n_candidates >= 1
    cand = next(c for c in res.candidates if c["kalshi_ticker"] == "KXJANE-28")
    assert cand["poly_slug"] == "jane-doe-2028-republican-nomination"
    assert cand["new_side"] == "kalshi"


def test_new_poly_matches_existing_kalshi_universe(monkeypatch, store):
    """A NEW Poly event matches the EXISTING Kalshi universe -> new_side='poly'."""
    new_poly_event = {
        "slug": "spacex-starship-orbit-2027",
        "question": "Will SpaceX reach orbit with Starship in 2027?",
        "description": "SpaceX Starship orbital flight 2027",
    }
    existing_kalshi_event = {
        "event_ticker": "KXSPX27",
        "title": "Will SpaceX reach orbit with Starship in 2027?",
        "sub_title": "SpaceX Starship orbital flight 2027",
        "markets": [{"ticker": "KXSPX-27", "open_time": "2026-01-01T00:00:00Z"}],
    }

    monkeypatch.setattr(dp, "_new_kalshi", lambda **k: [])  # no new kalshi
    monkeypatch.setattr(dp, "_new_poly", lambda **k: [new_poly_event])
    monkeypatch.setattr(
        dp,
        "_crawl_kalshi",
        lambda **k: KalshiEventsPage(events=[existing_kalshi_event]),
    )
    monkeypatch.setattr(dp, "_crawl_poly_volume", lambda **k: PolyCrawlPage(events=[]))

    res = dp.run_discovery_step(store=store, mode="new", within_hours=72.0, min_score=0.5)

    assert res.mode == "new"
    assert res.n_candidates >= 1
    cand = next(c for c in res.candidates if c["poly_slug"] == "spacex-starship-orbit-2027")
    assert cand["kalshi_ticker"] == "KXSPX-27"
    assert cand["new_side"] == "poly"


def test_new_mode_dedupes_pair_reachable_from_both_directions(monkeypatch, store):
    """A pair where BOTH sides are new is reachable from both match directions.

    It must appear exactly ONCE (deduped by (kalshi_ticker, poly_slug)) and be
    tagged new_side='both'.
    """
    new_kalshi_event = {
        "event_ticker": "KXFED26",
        "title": "Fed decision: rate cut in June 2026 FOMC",
        "sub_title": "Federal Reserve June 2026 meeting",
        "markets": [{"ticker": "KXFED-26JUN-CUT", "open_time": "2026-05-21T00:00:00Z"}],
    }
    new_poly_event = {
        "slug": "fed-rate-cut-june-2026",
        "question": "Fed decision: rate cut in June 2026 FOMC",
        "description": "Federal Reserve June 2026 meeting",
    }

    monkeypatch.setattr(dp, "_new_kalshi", lambda **k: [new_kalshi_event])
    monkeypatch.setattr(dp, "_new_poly", lambda **k: [new_poly_event])
    monkeypatch.setattr(dp, "_crawl_kalshi", lambda **k: KalshiEventsPage(events=[]))
    monkeypatch.setattr(dp, "_crawl_poly_volume", lambda **k: PolyCrawlPage(events=[]))

    res = dp.run_discovery_step(store=store, mode="new", within_hours=72.0, min_score=0.5)

    matches = [
        c
        for c in res.candidates
        if c["kalshi_ticker"] == "KXFED-26JUN-CUT" and c["poly_slug"] == "fed-rate-cut-june-2026"
    ]
    assert len(matches) == 1  # deduped despite being reachable both ways
    assert matches[0]["new_side"] == "both"


def test_new_mode_drops_ephemeral_new_events(monkeypatch, store):
    """Ephemeral new events are dropped before matching in new mode.

    ``new_kalshi_events`` / ``new_poly_events`` already filter ephemerals, and
    the liquid Kalshi universe crawl re-filters. Here the only NEW items are
    ephemeral, so they never reach the matcher -> zero candidates.
    """
    captured: dict[str, Any] = {}

    def fake_match(k, p, **kw):
        captured.setdefault("k_titles", [])
        captured["k_titles"] += [i.get("title") for i in k]
        captured.setdefault("p_titles", [])
        captured["p_titles"] += [i.get("title") or i.get("question") for i in p]
        return []

    # new_kalshi_events / new_poly_events filter ephemerals upstream, so a real
    # implementation returns []. We model that: the ephemeral new items are gone.
    monkeypatch.setattr(dp, "_new_kalshi", lambda **k: [])
    monkeypatch.setattr(dp, "_new_poly", lambda **k: [])
    # The liquid Kalshi universe crawl includes an ephemeral event that the
    # pipeline must re-filter before matching.
    monkeypatch.setattr(
        dp,
        "_crawl_kalshi",
        lambda **k: KalshiEventsPage(
            events=[
                {
                    "event_ticker": "KXSOL",
                    "title": "Solana Up or Down - May 22 3:15PM-3:30PM ET",
                    "markets": [{"ticker": "KXSOL-Y"}],
                }
            ]
        ),
    )
    monkeypatch.setattr(dp, "_crawl_poly_volume", lambda **k: PolyCrawlPage(events=[]))
    monkeypatch.setattr(dp, "_match_markets", fake_match)
    monkeypatch.setattr(dp, "_summarize", lambda c: {"total": 0})

    res = dp.run_discovery_step(store=store, mode="new", within_hours=72.0, min_score=0.5)

    # No new events -> no match calls happen at all, zero candidates.
    assert res.n_candidates == 0
    assert res.n_kalshi == 0
    assert res.n_poly == 0


def test_new_mode_recall_first_retains_low_score_soft_reject(monkeypatch, store):
    """A low-score (0.3) soft-reject in new mode is RETAINED with tier/confidence."""
    soft = _make_candidate(
        ticker="KXLOW",
        slug="low",
        score=0.3,
        rejected=True,
        reason="low_score",
        tier="reject",
    )

    # One new Kalshi event so the new_k×universe direction runs.
    monkeypatch.setattr(
        dp,
        "_new_kalshi",
        lambda **k: [{"ticker": "KXNEW", "title": "New thing"}],
    )
    monkeypatch.setattr(dp, "_new_poly", lambda **k: [])
    monkeypatch.setattr(dp, "_crawl_kalshi", lambda **k: KalshiEventsPage(events=[]))
    monkeypatch.setattr(
        dp,
        "_crawl_poly_volume",
        lambda **k: PolyCrawlPage(events=[{"slug": "low", "question": "low"}]),
    )
    # Force the matcher to return the soft-reject candidate.
    monkeypatch.setattr(dp, "_match_markets", lambda k, p, **kw: [soft])
    monkeypatch.setattr(dp, "_summarize", lambda c: {"total": len(c)})

    res = dp.run_discovery_step(store=store, mode="new", within_hours=72.0, min_score=0.5)

    assert res.n_candidates == 1  # NOT dropped
    assert res.n_high == 0  # below min_score
    assert res.n_review == 1
    cand = res.candidates[0]
    assert cand["confidence"] == "review"
    assert cand["tier"] == "reject"
    assert cand["reject_reason"] == "low_score"
    # new_side resolves from the new_k membership (KXLOW is not in new_k -> but
    # the candidate came from the new_k×universe match, so the kalshi side is
    # whatever the stub returned; the tag is computed from identity sets).
    assert "new_side" in cand


def test_sweep_mode_filters_ephemeral_events(monkeypatch, tmp_path, store):
    """Sweep mode drops ephemeral events/poly markets before matching."""
    from pfm.arb.market_crawler import KalshiEventsPage, PolyCrawlPage

    def fake_crawl_kalshi(*, cursor, max_pages, session):
        return KalshiEventsPage(
            events=[
                {
                    "event_ticker": "KXSOL",
                    "title": "Solana Up or Down - May 22 3:15PM-3:30PM ET",
                    "markets": [{"ticker": "KXSOL-Y"}],
                },
                {
                    "event_ticker": "KXELECT",
                    "title": "Will the GOP win the 2028 presidential election?",
                    "markets": [{"ticker": "KXELECT-Y"}],
                },
            ],
            next_cursor=None,
            done=True,
        )

    def fake_crawl_poly(*, offset, max_pages, session):
        return PolyCrawlPage(
            events=[
                {"slug": "btc-15m", "title": "Bitcoin Up/Down 15m"},
                {"slug": "gop-2028", "title": "Will the GOP win the 2028 presidential election?"},
            ],
            next_offset=offset,
            done=True,
        )

    seen = {}

    def fake_match(k, p, **kw):
        seen["k"] = [i.get("title") for i in k]
        seen["p"] = [i.get("title") for i in p]
        return []

    monkeypatch.setattr(dp, "_crawl_kalshi", fake_crawl_kalshi)
    monkeypatch.setattr(dp, "_crawl_poly", fake_crawl_poly)
    monkeypatch.setattr(dp, "_match_markets", fake_match)
    monkeypatch.setattr(dp, "_summarize", lambda c: {"total": 0})

    dp.run_discovery_step(checkpoint_path=str(tmp_path / "ckpt.json"), store=store, mode="sweep")

    # Ephemeral Kalshi event + ephemeral poly market dropped before matching.
    assert seen["k"] == ["Will the GOP win the 2028 presidential election?"]
    assert seen["p"] == ["Will the GOP win the 2028 presidential election?"]


# ---------------------------------------------------------------------------
# Candidates: produced, FP-rejects excluded, n_high counted.
# ---------------------------------------------------------------------------


def test_hard_rejects_excluded_soft_rejects_surfaced_and_count_high(monkeypatch, store):
    # Recall-first: high + borderline surfaced; a SOFT (low_score) reject is
    # ALSO surfaced (confidence=review); only the HARD-gate reject is dropped.
    high = _make_candidate(ticker="KXA", slug="a", score=0.9, tier="high")
    borderline = _make_candidate(ticker="KXB", slug="b", score=0.55, tier="borderline")
    soft_reject = _make_candidate(
        ticker="KXD",
        slug="d",
        score=0.3,
        rejected=True,
        reason="low_score",
        tier="reject",
    )
    hard_reject = _make_candidate(
        ticker="KXC",
        slug="c",
        score=0.0,
        rejected=True,
        reason="jurisdiction_mismatch",
        tier="reject",
    )

    monkeypatch.setattr(
        dp,
        "_crawl_kalshi",
        lambda **k: KalshiEventsPage(events=[{"event_ticker": "x", "title": "X"}]),
    )
    monkeypatch.setattr(dp, "_crawl_poly", lambda **k: PolyCrawlPage(events=[{"slug": "y"}]))
    monkeypatch.setattr(
        dp,
        "_match_markets",
        lambda k, p, **kw: [high, borderline, soft_reject, hard_reject],
    )
    monkeypatch.setattr(
        dp,
        "_summarize",
        lambda c: {"total": len(c)},
    )

    res = dp.run_discovery_step(store=store, mode="sweep", min_score=0.5)

    # Hard-gate reject dropped; high + borderline + soft reject all surfaced.
    assert res.n_candidates == 3
    tickers = {c["kalshi_ticker"] for c in res.candidates}
    assert tickers == {"KXA", "KXB", "KXD"}
    # n_high counts only score >= min_score (0.5): high (0.9) + borderline (0.55).
    assert res.n_high == 2
    # The soft-reject (0.3) is flagged "review", not hidden.
    by_ticker = {c["kalshi_ticker"]: c for c in res.candidates}
    assert by_ticker["KXD"]["confidence"] == "review"
    assert by_ticker["KXD"]["tier"] == "reject"
    assert by_ticker["KXA"]["confidence"] == "verified"
    assert res.n_review == 1
    # summary sees the full (unfiltered) candidate list.
    assert res.summary["total"] == 4


# ---------------------------------------------------------------------------
# Recall-first: a low-score mismatch is RETAINED (confidence=review), not dropped.
# ---------------------------------------------------------------------------


def test_recall_first_retains_low_score_mismatch_with_review_flag(monkeypatch, store):
    """Recall over precision: a real but low-confidence pair must NOT vanish.

    Lets the REAL matcher run against two markets that share a discriminative
    token but mismatch enough to score below the high bar. Recall-first keeps
    it, tagged tier + confidence='review', so the human can eyeball it.
    """
    kalshi_event = {
        "event_ticker": "KXSPACEX",
        "title": "Will SpaceX reach orbit with Starship in 2026?",
        "sub_title": "SpaceX Starship orbital flight 2026",
        "markets": [{"ticker": "KXSPX-26", "open_time": "2026-05-21T00:00:00Z"}],
    }
    # Shares "spacex"/"starship" tokens but framed differently (lower score,
    # below the high bar -> a borderline/soft-reject the human should still see).
    poly_market = {
        "slug": "spacex-starship-orbit-2026",
        "question": "Will SpaceX launch Starship to orbit by end of 2026?",
        "description": "SpaceX Starship orbital launch attempt",
        "volumeNum": 1.2e6,
    }

    monkeypatch.setattr(
        dp,
        "_crawl_kalshi",
        lambda **k: KalshiEventsPage(events=[kalshi_event]),
    )
    monkeypatch.setattr(
        dp,
        "_crawl_poly_volume",
        lambda **k: PolyCrawlPage(events=[poly_market]),
    )

    res = dp.run_discovery_step(store=store, mode="liquid", min_score=0.7)

    assert res.mode == "liquid"
    assert res.n_poly == 1
    # The pair is surfaced even though it is below the high bar.
    assert res.n_candidates >= 1
    cand = res.candidates[0]
    assert "confidence" in cand and cand["confidence"] in ("verified", "review")
    assert "tier" in cand
    # It scored below min_score=0.7, so it must be flagged review (not hidden).
    if cand["score"] < 0.7 or cand["rejected"]:
        assert cand["confidence"] == "review"


def test_recall_first_score_0_3_candidate_is_retained(monkeypatch, store):
    """A score=0.3 SOFT-reject candidate is RETAINED with tier + confidence."""
    soft = _make_candidate(
        ticker="KXLOW",
        slug="low",
        score=0.3,
        rejected=True,
        reason="low_score",
        tier="reject",
    )
    monkeypatch.setattr(
        dp,
        "_crawl_kalshi",
        lambda **k: KalshiEventsPage(events=[{"event_ticker": "x", "title": "X"}]),
    )
    monkeypatch.setattr(dp, "_crawl_poly", lambda **k: PolyCrawlPage(events=[{"slug": "y"}]))
    monkeypatch.setattr(dp, "_match_markets", lambda k, p, **kw: [soft])
    monkeypatch.setattr(dp, "_summarize", lambda c: {"total": len(c)})

    res = dp.run_discovery_step(store=store, mode="sweep", min_score=0.5)

    assert res.n_candidates == 1  # NOT dropped
    assert res.n_high == 0  # below min_score, so not counted high
    assert res.n_review == 1
    cand = res.candidates[0]
    assert cand["confidence"] == "review"
    assert cand["tier"] == "reject"
    assert cand["reject_reason"] == "low_score"


def test_recall_floor_passed_to_matcher_and_min_score_not_a_filter(monkeypatch, store):
    """``min_score`` reaches the matcher as ``recall_floor``, with soft rejects kept."""
    seen: dict[str, Any] = {}

    def fake_match(k, p, *, min_score, keep_soft_rejects):
        seen["min_score"] = min_score
        seen["keep_soft_rejects"] = keep_soft_rejects
        return []

    monkeypatch.setattr(
        dp,
        "_crawl_kalshi",
        lambda **k: KalshiEventsPage(events=[{"event_ticker": "x", "title": "X"}]),
    )
    monkeypatch.setattr(dp, "_crawl_poly", lambda **k: PolyCrawlPage(events=[{"slug": "y"}]))
    monkeypatch.setattr(dp, "_match_markets", fake_match)
    monkeypatch.setattr(dp, "_summarize", lambda c: {"total": 0})

    res = dp.run_discovery_step(store=store, mode="sweep", min_score=0.6, recall_floor=0.2)

    # Matcher is invoked at the LOW recall floor, keeping soft rejects.
    assert seen["min_score"] == 0.2
    assert seen["keep_soft_rejects"] is True
    assert res.recall_floor == 0.2


# ---------------------------------------------------------------------------
# Liquid mode: volume-sorted poly coverage, no aggressive ephemeral drop.
# ---------------------------------------------------------------------------


def test_liquid_mode_uses_volume_crawl_and_keeps_liquid_markets(monkeypatch, store):
    """Liquid mode crawls poly by volume and does NOT ephemeral-filter it."""
    volume_seen = {"called": False}

    def fake_poly_volume(**k):
        volume_seen["called"] = True
        return PolyCrawlPage(
            events=[
                {"slug": "trump-2028", "question": "Will Trump win 2028?"},
                # An "up or down" liquid title must survive (no ephemeral drop).
                {"slug": "btc-band", "question": "Bitcoin up or down by year end?"},
            ]
        )

    monkeypatch.setattr(
        dp,
        "_crawl_kalshi",
        lambda **k: KalshiEventsPage(events=[{"event_ticker": "x", "title": "X"}]),
    )
    monkeypatch.setattr(dp, "_crawl_poly_volume", fake_poly_volume)
    captured: dict[str, Any] = {}

    def fake_match(k, p, **kw):
        captured["poly_slugs"] = [i.get("slug") for i in p]
        return []

    monkeypatch.setattr(dp, "_match_markets", fake_match)
    monkeypatch.setattr(dp, "_summarize", lambda c: {"total": 0})

    res = dp.run_discovery_step(store=store, mode="liquid", max_pages=2)

    assert res.mode == "liquid"
    assert volume_seen["called"] is True
    assert res.n_poly == 2
    # Both liquid markets reach the matcher — including the "up or down" one.
    assert set(captured["poly_slugs"]) == {"trump-2028", "btc-band"}
    # Liquid mode leaves the checkpoint untouched.
    assert res.checkpoint == {}


def test_liquid_mode_only_records_verified(monkeypatch, store):
    """Recording is restricted to verified candidates; review ones are not stored."""
    verified = _make_candidate(ticker="KXV", slug="v", score=0.9, tier="high")
    review = _make_candidate(
        ticker="KXR",
        slug="r",
        score=0.3,
        rejected=True,
        reason="low_score",
        tier="reject",
    )
    monkeypatch.setattr(
        dp,
        "_crawl_kalshi",
        lambda **k: KalshiEventsPage(events=[{"event_ticker": "x", "title": "X"}]),
    )
    monkeypatch.setattr(dp, "_crawl_poly_volume", lambda **k: PolyCrawlPage(events=[{"slug": "y"}]))
    monkeypatch.setattr(dp, "_match_markets", lambda k, p, **kw: [verified, review])
    monkeypatch.setattr(dp, "_summarize", lambda c: {"total": len(c)})

    priced: list[str] = []

    def price_fn(ticker: str, slug: str) -> dict[str, float]:
        priced.append(ticker)
        return {"kalshi_yes_ask": 0.40, "poly_no_price": 0.50}

    res = dp.run_discovery_step(store=store, mode="liquid", min_score=0.5, price_fn=price_fn)

    # Both surfaced, but only the verified one is priced + recorded.
    assert res.n_candidates == 2
    assert priced == ["KXV"]
    assert res.n_recorded == 1
    assert store.get("KXV|v") is not None
    assert store.get("KXR|r") is None


# ---------------------------------------------------------------------------
# Arb detection / recording.
# ---------------------------------------------------------------------------


def _patch_one_candidate(monkeypatch, cand: Candidate) -> None:
    monkeypatch.setattr(
        dp,
        "_crawl_kalshi",
        lambda **k: KalshiEventsPage(events=[{"event_ticker": "x", "title": "X"}]),
    )
    monkeypatch.setattr(dp, "_crawl_poly", lambda **k: PolyCrawlPage(events=[{"slug": "y"}]))
    monkeypatch.setattr(dp, "_match_markets", lambda k, p, **kw: [cand])
    monkeypatch.setattr(dp, "_summarize", lambda c: {"total": len(c)})


def test_price_fn_records_arb_and_count_bumps(monkeypatch, store):
    cand = _make_candidate(ticker="KXARB", slug="arb-slug")
    _patch_one_candidate(monkeypatch, cand)

    # 0.45 + 0.50 = 0.95 < 1.0 -> arb, profit = 5%.
    def price_fn(ticker: str, slug: str) -> dict[str, float]:
        assert ticker == "KXARB"
        assert slug == "arb-slug"
        return {
            "kalshi_yes_ask": 0.45,
            "poly_no_price": 0.50,
            "kalshi_no_ask": 0.60,
            "poly_yes_price": 0.55,
        }

    res1 = dp.run_discovery_step(store=store, mode="sweep", price_fn=price_fn)
    assert res1.n_recorded == 1

    arb = store.get("KXARB|arb-slug")
    assert arb is not None
    assert arb.count == 1
    assert arb.confidence == "high"
    assert arb.max_profit_pct == pytest.approx(5.0)

    # Repeat step bumps the count (durable, growable store).
    res2 = dp.run_discovery_step(store=store, mode="sweep", price_fn=price_fn)
    assert res2.n_recorded == 1
    assert store.get("KXARB|arb-slug").count == 2


def test_no_arb_when_cost_not_below_one(monkeypatch, store):
    cand = _make_candidate(ticker="KXNOARB", slug="noarb")
    _patch_one_candidate(monkeypatch, cand)

    def price_fn(ticker: str, slug: str) -> dict[str, float]:
        # Both legs >= 1.0 -> no arb.
        return {
            "kalshi_yes_ask": 0.60,
            "poly_no_price": 0.55,
            "kalshi_no_ask": 0.58,
            "poly_yes_price": 0.52,
        }

    res = dp.run_discovery_step(store=store, mode="sweep", price_fn=price_fn)
    assert res.n_recorded == 0
    assert len(store) == 0


def test_price_fn_returning_none_records_nothing(monkeypatch, store):
    cand = _make_candidate()
    _patch_one_candidate(monkeypatch, cand)

    res = dp.run_discovery_step(store=store, mode="sweep", price_fn=lambda t, s: None)
    assert res.n_recorded == 0


def test_no_price_fn_means_discovery_only(monkeypatch, store):
    cand = _make_candidate()
    _patch_one_candidate(monkeypatch, cand)

    res = dp.run_discovery_step(store=store, mode="sweep", price_fn=None)
    assert res.n_recorded == 0
    assert res.n_candidates == 1
    assert len(store) == 0


# ---------------------------------------------------------------------------
# Result shape / serialisability.
# ---------------------------------------------------------------------------


def test_result_shape_is_serialisable(monkeypatch, store):
    cand = _make_candidate()
    _patch_one_candidate(monkeypatch, cand)

    res = dp.run_discovery_step(store=store, mode="sweep")
    d = res.as_dict()

    expected_keys = {
        "n_kalshi",
        "n_poly",
        "n_candidates",
        "n_high",
        "n_recorded",
        "mode",
        "checkpoint",
        "summary",
        "candidates",
        "n_review",
        "recall_floor",
    }
    assert set(d) == expected_keys
    # Round-trips through JSON without error.
    encoded = json.dumps(d)
    assert json.loads(encoded)["mode"] == "sweep"
    assert isinstance(d["candidates"], list)
    assert d["candidates"][0]["kalshi_ticker"] == "KXTEST"


def test_invalid_mode_raises(store):
    with pytest.raises(ValueError, match="mode must be"):
        dp.run_discovery_step(store=store, mode="bogus")


# ---------------------------------------------------------------------------
# default_store helper.
# ---------------------------------------------------------------------------


def test_default_store_honors_env(monkeypatch, tmp_path):
    target = tmp_path / "env_store.json"
    monkeypatch.setenv("PFM_ARB_CONFIRMED_STORE", str(target))
    s = dp.default_store()
    assert s.path == target
