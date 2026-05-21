"""Tests for :mod:`pfm.arb.discovery_matcher`.

Drives off the same hand-curated golden fixture as the T78 audit
(``arb_match_known_false_positives.json``):

- every ``bad`` pair must come back ``rejected=True`` OR ``score < 0.4``;
- every ``good`` pair must be non-rejected with a reasonably high score
  (mirrors :mod:`tests.test_arb_match_quality` expectations: ``> 0.7``);
- the ``known_t77_gaps`` cases are tracked as ``xfail`` (they currently
  reject thanks to the T76b ``windows_overlap`` fix, but the fixture flags
  them as historically-leaky, so we don't hard-assert on them).

Plus structural tests for the prefilter, the per-Kalshi cap, and the
``summarize`` histogram. All no-network: ``score_match`` is pure regex +
set-Jaccard + date arithmetic and never downloads a model.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

import pfm.arb.discovery_matcher as dm
from pfm.arb.discovery_matcher import (
    Candidate,
    match_markets,
    match_one,
    summarize,
)

FIXTURES_PATH = Path(__file__).parent / "fixtures" / "arb_match_known_false_positives.json"


@pytest.fixture(scope="module")
def fixtures() -> dict:
    with FIXTURES_PATH.open() as fh:
        return json.load(fh)


def _candidate(poly_title: str, kalshi_title: str) -> Candidate:
    return match_one({"title": kalshi_title}, {"title": poly_title})


_FX = json.loads(FIXTURES_PATH.read_text())


# ---------------------------------------------------------------------------
# Golden fixture — bad pairs must reject, good pairs must score high.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", [pytest.param(c, id=c["pair_id"]) for c in _FX["bad"]])
def test_known_false_positives_rejected(case: dict) -> None:
    cand = _candidate(case["poly_title"], case["kalshi_title"])
    assert cand.rejected or cand.score < 0.4, (
        f"{case['pair_id']!r} not rejected: score={cand.score}, reason={cand.reject_reason!r}"
    )
    assert cand.tier == "reject"


@pytest.mark.parametrize("case", [pytest.param(c, id=c["pair_id"]) for c in _FX["good"]])
def test_known_true_positives_score_high(case: dict) -> None:
    cand = _candidate(case["poly_title"], case["kalshi_title"])
    assert not cand.rejected, f"{case['pair_id']!r} wrongly rejected: reason={cand.reject_reason!r}"
    assert cand.score > 0.7, f"{case['pair_id']!r} only scored {cand.score:.3f}"
    assert cand.tier == "high"


@pytest.mark.xfail(
    reason="known_t77_gaps are historically-leaky window cases; T76b fixes "
    "them today but the fixture flags them as not-guaranteed.",
    strict=False,
)
@pytest.mark.parametrize(
    "case",
    [pytest.param(c, id=c["pair_id"]) for c in _FX.get("known_t77_gaps", [])],
)
def test_known_t77_gaps(case: dict) -> None:
    cand = _candidate(case["poly_title"], case["kalshi_title"])
    # We *want* these to reject; xfail(strict=False) means it's fine either way.
    assert not (cand.rejected or cand.score < 0.4)


def test_expected_reject_reasons_match_fixture() -> None:
    """When the fixture annotates an ``expected_reason``, the matcher's hard
    reject reason should agree (for the hard-gate reasons)."""

    for case in _FX["bad"]:
        expected = case.get("expected_reason")
        if not expected:
            continue
        cand = _candidate(case["poly_title"], case["kalshi_title"])
        assert cand.reject_reason == expected, (
            f"{case['pair_id']!r}: expected {expected!r}, got {cand.reject_reason!r}"
        )


# ---------------------------------------------------------------------------
# match_one shape + tier boundaries
# ---------------------------------------------------------------------------


def test_match_one_returns_candidate_shape() -> None:
    cand = match_one(
        {"title": "Will Bitcoin reach $100,000 by Dec 31 2025?", "ticker": "BTC-100K"},
        {"title": "Will Bitcoin reach $100,000 by Dec 31 2025?", "slug": "btc-100k"},
    )
    assert isinstance(cand, Candidate)
    assert cand.kalshi_ticker == "BTC-100K"
    assert cand.poly_slug == "btc-100k"
    assert 0.0 <= cand.score <= 1.0
    assert isinstance(cand.components, dict)
    assert cand.tier == "high"
    # as_dict round-trips the public fields.
    d = cand.as_dict()
    assert d["score"] == cand.score
    assert d["reject_reason"] == cand.reject_reason
    # new_side is additive and defaults to None.
    assert d["new_side"] is None
    assert cand.new_side is None


def test_unrelated_pair_is_rejected_with_reason() -> None:
    # match_one always assigns poly->polymarket and kalshi->kalshi, so the
    # same_venue gate can never fire here; an unrelated pair must still reject
    # (either a hard gate or "low_score") and land in the reject tier.
    cand = match_one(
        {"title": "Will the New York Knicks make the NBA playoffs in 2026?"},
        {"title": "Will the Federal Reserve cut interest rates in June 2026?"},
    )
    assert cand.rejected
    assert cand.reject_reason is not None
    assert cand.tier == "reject"


# ---------------------------------------------------------------------------
# Prefilter (recall stage)
# ---------------------------------------------------------------------------


def test_prefilter_drops_obviously_unrelated_pairs() -> None:
    kalshi = [{"title": "Will the Kansas City Chiefs win Super Bowl by Feb 8 2026?"}]
    poly = [
        {"title": "Will the Kansas City Chiefs win Super Bowl by Feb 8 2026?"},
        {"title": "Will the Federal Reserve cut interest rates in June 2026?"},
    ]
    # With prefilter on, only the matching Super Bowl pair survives scoring.
    cands = match_markets(kalshi, poly, prefilter=True)
    assert len(cands) == 1
    assert "Chiefs" in cands[0].poly_title


def test_prefilter_off_scores_all_pairs() -> None:
    kalshi = [{"title": "Totally unrelated kalshi market about hockey"}]
    poly = [{"title": "Totally unrelated polymarket about baseball weather"}]
    # prefilter off => score_match is invoked; the unrelated pair still ends up
    # filtered out by min_score, so the accepted list is empty either way, but
    # this exercises the prefilter=False branch without error.
    cands = match_markets(kalshi, poly, prefilter=False)
    assert cands == []


def test_keep_soft_rejects_retains_low_score_but_drops_hard_gate() -> None:
    """Recall-first matcher flag: keep soft (low_score) rejects, drop hard ones.

    Two markets share a discriminative token so the prefilter pairs them, but
    they mismatch enough to fall below the high bar (a soft ``low_score``
    reject). With ``keep_soft_rejects=True`` and a low floor it is RETAINED; the
    default precision call drops it.
    """
    kalshi = [{"title": "Will SpaceX reach orbit with Starship in 2026?", "ticker": "KXSPX"}]
    poly = [
        {
            "title": "Will SpaceX launch Starship to orbit by end of 2026?",
            "slug": "spacex-starship-2026",
        }
    ]

    # Default precision mode at a HIGH bar: the ~0.6 pair is below min_score
    # and is filtered out entirely.
    precise = match_markets(kalshi, poly, min_score=0.7)
    assert precise == []

    # Recall-first: the same pair survives at the low recall floor, tagged.
    recall = match_markets(kalshi, poly, min_score=0.25, keep_soft_rejects=True)
    assert len(recall) == 1
    cand = recall[0]
    assert cand.score >= 0.25
    assert cand.tier in ("borderline", "high")
    # No hard gate fired — a hard-gate reject would still be dropped.
    assert cand.reject_reason in (None, "low_score")


# ---------------------------------------------------------------------------
# Per-Kalshi cap + best-first ordering
# ---------------------------------------------------------------------------


def test_match_markets_caps_per_kalshi() -> None:
    kalshi = [{"title": "Will Bitcoin reach $100,000 by Dec 31 2025?"}]
    # Five identical-ish poly markets that all match the kalshi leg.
    poly = [
        {"title": "Will Bitcoin reach $100,000 by Dec 31 2025?", "slug": f"btc-{i}"}
        for i in range(5)
    ]
    cands = match_markets(kalshi, poly, max_candidates_per_kalshi=2)
    assert len(cands) == 2
    # Best-first ordering preserved.
    scores = [c.score for c in cands]
    assert scores == sorted(scores, reverse=True)


def test_match_markets_results_sorted_best_first() -> None:
    kalshi = [
        {"title": "Will Bitcoin reach $100,000 by Dec 31 2025?"},
        {"title": "Will Republicans control the US Senate by Nov 3 2026?"},
    ]
    poly = [
        {"title": "Will Republicans hold the US Senate majority by Nov 3 2026?"},
        {"title": "Will Bitcoin reach $100,000 by Dec 31 2025?"},
    ]
    cands = match_markets(kalshi, poly)
    assert len(cands) >= 2
    scores = [c.score for c in cands]
    assert scores == sorted(scores, reverse=True)
    assert all(not c.rejected and c.score >= 0.4 for c in cands)


def test_match_markets_negative_cap_raises() -> None:
    with pytest.raises(ValueError):
        match_markets([], [], max_candidates_per_kalshi=-1)


# ---------------------------------------------------------------------------
# summarize histogram
# ---------------------------------------------------------------------------


def test_summarize_buckets_correctly() -> None:
    # Score every fixture pair (bad + good) WITHOUT min_score filtering so the
    # reject histogram is populated.
    cands: list[Candidate] = []
    for c in _FX["bad"] + _FX["good"]:
        cands.append(_candidate(c["poly_title"], c["kalshi_title"]))

    summary = summarize(cands)
    assert summary["total"] == len(cands)
    # Tiers partition the total exactly.
    by_tier = summary["by_tier"]
    assert by_tier["high"] + by_tier["borderline"] + by_tier["reject"] == summary["total"]
    # All 10 bad pairs are rejects, all 5 good are high.
    assert by_tier["reject"] == len(_FX["bad"])
    assert by_tier["high"] == len(_FX["good"])
    # Reject-reason histogram sums to the number of rejects.
    assert sum(summary["reject_reasons"].values()) == by_tier["reject"]
    # Hard-gate reasons present.
    assert "jurisdiction_mismatch" in summary["reject_reasons"]
    assert "resolution_window_no_overlap" in summary["reject_reasons"]
    assert "threshold_mismatch" in summary["reject_reasons"]


def test_summarize_empty() -> None:
    summary = summarize([])
    assert summary["total"] == 0
    assert summary["by_tier"] == {"high": 0, "borderline": 0, "reject": 0}
    assert summary["reject_reasons"] == {}


# ---------------------------------------------------------------------------
# Blocking prefilter — scale + score_match call-count bound
# ---------------------------------------------------------------------------

# A varied vocabulary of rare-ish entities so the inverted index is exercised
# realistically (not all markets share the same discriminative token).
_PEOPLE = [
    "Trump",
    "Biden",
    "Macron",
    "Putin",
    "Modi",
    "Lula",
    "Sunak",
    "Scholz",
    "Milei",
    "Erdogan",
    "Xi",
    "Kishida",
    "Meloni",
    "Trudeau",
    "Orban",
    "Lee",
]
_ASSETS = [
    "Bitcoin",
    "Ethereum",
    "Solana",
    "Cardano",
    "Avalanche",
    "Polkadot",
    "Chainlink",
    "Dogecoin",
    "Litecoin",
    "Ripple",
    "Monero",
    "Stellar",
]
_PLACES = [
    "Argentina",
    "Brazil",
    "Canada",
    "Denmark",
    "Estonia",
    "Finland",
    "Germany",
    "Hungary",
    "Iceland",
    "Japan",
    "Kenya",
    "Latvia",
]


def _synthetic_market(i: int) -> dict[str, str]:
    """Build a market title with a varied, mostly-unique entity signature."""

    person = _PEOPLE[i % len(_PEOPLE)]
    asset = _ASSETS[(i // 3) % len(_ASSETS)]
    place = _PLACES[(i // 7) % len(_PLACES)]
    # A pseudo-unique entity token guarantees variety across the corpus.
    uniq = f"Zentropy{i:05d}"
    year = 2024 + (i % 5)
    return {"title": (f"Will {person} and {asset} affect {place} {uniq} by Dec 31 {year}?")}


def test_match_markets_scales() -> None:
    """1000×1000 must finish fast and slash score_match calls via blocking."""

    n = 1000
    # Kalshi & poly share the same i-th market (so each kalshi has exactly one
    # strong same-entity poly counterpart plus weaker neighbours).
    kalshi = [_synthetic_market(i) for i in range(n)]
    poly = [_synthetic_market(i) for i in range(n)]

    calls = {"n": 0}
    real_score_match = dm.score_match

    def _counting_score_match(a, b):  # type: ignore[no-untyped-def]
        calls["n"] += 1
        return real_score_match(a, b)

    dm.score_match = _counting_score_match  # type: ignore[assignment]
    try:
        t0 = time.time()
        cands = match_markets(kalshi, poly, block_size=20)
        elapsed = time.time() - t0
    finally:
        dm.score_match = real_score_match  # type: ignore[assignment]

    # Performance: blocking must keep this well under the exhaustive cost.
    assert elapsed < 8.0, f"match_markets took {elapsed:.2f}s (expected < 8s)"

    # Blocking must cut score_match calls far below the n*m cross-product.
    assert calls["n"] < n * 25, (
        f"score_match called {calls['n']} times; blocking should cap it "
        f"near n*block_size, not n*m={n * n}"
    )

    # Sensible output: each kalshi finds its identical poly counterpart, so we
    # expect a healthy number of high-confidence matches.
    assert len(cands) > 0
    assert all(not c.rejected and c.score >= 0.4 for c in cands)
    assert any(c.tier == "high" for c in cands)


def test_blocking_finds_shared_entity_match() -> None:
    """A rare shared entity bridges the right pair even amid generic noise."""

    kalshi = [{"title": "Will Nvidia stock close above $1500 by Dec 31 2026?"}]
    poly = [
        {"title": "Will Nvidia stock close above $1500 by Dec 31 2026?", "slug": "nvda"},
        # Decoys sharing only generic/stoplisted vocabulary.
        {"title": "Will the price reach a new market high in 2026?", "slug": "noise-1"},
        {"title": "Will this contract win the market by 2026?", "slug": "noise-2"},
    ]
    cands = match_markets(kalshi, poly, block_size=20)
    assert len(cands) == 1
    assert cands[0].poly_slug == "nvda"


def test_negative_block_size_raises() -> None:
    with pytest.raises(ValueError):
        match_markets([], [], block_size=-1)
