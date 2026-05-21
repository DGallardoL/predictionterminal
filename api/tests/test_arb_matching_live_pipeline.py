"""End-to-end test for the T76 → T77 → T78 arb-matching pipeline.

Task **W11-35**. Validates the complete pipeline that goes from raw mocked
Polymarket / Kalshi payloads through to the T78 audit CSV:

    raw_payload  --build_market_desc-->  MarketDesc
        ↓
    MarketDesc(a), MarketDesc(b)  --score_match-->  SimilarityScore
        ↓
    rows = audit_pairs([...])
        ↓
    write_csv(rows, path) + summarise(rows) + write_blacklist_proposals(rows, path)

Fixture taxonomy:

- **5 good pairs**: identical events / cross-venue / overlapping windows.
  Expect ``score > 0.7`` and ``rejected_reason is None``.
- **5 user-flagged false positives**: the catalogue the user pointed out in
  the W11 spec — Trump 2024 vs 2028, Fed Mar vs Jun, BTC $80k vs $90k,
  US Senate vs FL Senate, and a *same-venue* dup. Each must be hard-rejected
  with the *expected* rejection reason.
- **3 borderline pairs**: deliberately ambiguous wording / partial entity
  overlap. Expect ``0.4 < score < 0.7`` and ``rejected_reason is None``.

The tests also cover:

- T78 audit-harness consumption (CSV columns, header, row count, every
  ``rejected`` cell is the literal ``True`` / ``False``).
- Performance: 100 synthetic pairs scored end-to-end in < 1 s wall-clock.
- Concurrent thread-safety: 16 workers × 25 pairs each, no exceptions, every
  worker returns identical results to the single-threaded baseline.

The pipeline modules under test are pure-Python and contain no I/O, so no
mocking of httpx / yfinance is needed — the "mocked Polymarket+Kalshi
payloads" are simply dict literals shaped like the Gamma / Kalshi API
responses we'd see at runtime.
"""

from __future__ import annotations

import csv
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
API_SRC = REPO_ROOT / "api" / "src"
if str(API_SRC) not in sys.path:  # pragma: no cover — path-bootstrapping
    sys.path.insert(0, str(API_SRC))

# Skip the whole module if T76/T77 haven't landed.
event_similarity = pytest.importorskip(
    "pfm.arb_matching.event_similarity",
    reason=(
        "W11-35 depends on T76 (pfm.arb_matching.date_extractor) + T77 "
        "(pfm.arb_matching.event_similarity). Skipping until they land."
    ),
)

# Import T78 audit harness (script-module).
SCRIPTS = REPO_ROOT / "api" / "scripts"
if str(SCRIPTS) not in sys.path:  # pragma: no cover
    sys.path.insert(0, str(SCRIPTS))
import audit_arb_matches

FIXTURES_PATH = Path(__file__).parent / "fixtures" / "arb_match_known_false_positives.json"


# ---------------------------------------------------------------------------
# Fixture construction. The "raw payloads" mimic the Polymarket Gamma /
# Kalshi shapes so build_market_desc(...) exercises the real production
# code-path: title/question/description coercion, threshold extraction,
# jurisdiction inference, ResolutionWindow extraction via T76.
# ---------------------------------------------------------------------------


def _poly(question: str, *, slug: str = "", description: str = "") -> dict[str, Any]:
    """Polymarket Gamma-shaped payload."""
    return {
        "question": question,
        "slug": slug or question.lower().replace(" ", "-")[:60],
        "description": description,
        "endDate": "2026-12-31T00:00:00Z",
    }


def _kalshi(title: str, *, ticker: str = "", subtitle: str = "") -> dict[str, Any]:
    """Kalshi-shaped payload."""
    return {
        "title": title,
        "ticker": ticker or title.upper().replace(" ", "_")[:24],
        "subtitle": subtitle,
        "close_time": "2026-12-31T00:00:00Z",
    }


# 5 good pairs — identical / near-identical events on opposite venues.
GOOD_PAIRS: list[dict[str, Any]] = [
    {
        "pair_id": "good-trump-2024-presidency",
        "poly": _poly("Will Donald Trump win the US presidential election by Nov 5 2024?"),
        "kalshi": _kalshi("Will Donald J. Trump win the US presidential election by Nov 5 2024?"),
        "expected_min_score": 0.7,
    },
    {
        "pair_id": "good-btc-100k-eoy-2025",
        "poly": _poly("Will Bitcoin reach $100,000 by Dec 31 2025?"),
        "kalshi": _kalshi("Will Bitcoin reach $100,000 by Dec 31 2025?"),
        "expected_min_score": 0.7,
    },
    {
        "pair_id": "good-superbowl-lx-chiefs",
        "poly": _poly("Will the Kansas City Chiefs win Super Bowl by Feb 8 2026?"),
        "kalshi": _kalshi("Will the Kansas City Chiefs win Super Bowl by Feb 8 2026?"),
        "expected_min_score": 0.7,
    },
    {
        "pair_id": "good-senate-control-2026",
        "poly": _poly("Will Republicans control the US Senate by Nov 3 2026?"),
        "kalshi": _kalshi("Will Republicans hold the US Senate majority by Nov 3 2026?"),
        "expected_min_score": 0.7,
    },
    {
        "pair_id": "good-eth-2000-eoy-2026",
        "poly": _poly("Will Ethereum reach $2,000 by Dec 31 2026?"),
        "kalshi": _kalshi("Will Ethereum reach $2,000 by Dec 31 2026?"),
        "expected_min_score": 0.7,
    },
]


# 5 user-flagged false positives. The user called out these patterns explicitly
# in the W11 brief; each must reject with the documented REJECT_REASON.
USER_FLAGGED_FALSE_POSITIVES: list[dict[str, Any]] = [
    {
        "pair_id": "fp-trump-2024-vs-2028",
        "poly": _poly("Will Donald Trump win the US presidential election by Nov 5 2024?"),
        "kalshi": _kalshi("Will Donald Trump win the US presidential election by Nov 7 2028?"),
        "expected_reason": event_similarity.REJECT_WINDOW_NO_OVERLAP,
        "notes": (
            "T76b's windows_overlap requires strict latest-date proximity for "
            "two point dates — Nov 2024 vs Nov 2028 reject."
        ),
    },
    {
        "pair_id": "fp-fed-march-vs-june-2026",
        "poly": _poly("Will the Fed cut rates at the FOMC meeting by March 18 2026?"),
        "kalshi": _kalshi("Will the Fed cut rates at the FOMC meeting by June 17 2026?"),
        "expected_reason": event_similarity.REJECT_WINDOW_NO_OVERLAP,
        "notes": "FOMC-meeting-specific point dates 3 months apart — must reject.",
    },
    {
        "pair_id": "fp-btc-80k-vs-90k",
        "poly": _poly("Will Bitcoin trade above $80,000 by Dec 31 2025?"),
        "kalshi": _kalshi("Will Bitcoin trade above $90,000 by Dec 31 2025?"),
        "expected_reason": event_similarity.REJECT_THRESHOLD_MISMATCH,
        "notes": "Threshold disagreement > 5% — must reject.",
    },
    {
        "pair_id": "fp-us-senate-vs-fl-state-senate",
        "poly": _poly("Will Democrats win the US Senate majority in 2026?"),
        "kalshi": _kalshi("Will Democrats win the Florida State Senate majority in 2026?"),
        "expected_reason": event_similarity.REJECT_JURISDICTION_MISMATCH,
        "notes": "US-Senate vs FL-State-Senate jurisdiction conflict.",
    },
    {
        "pair_id": "fp-same-venue-dup",
        "poly": _poly("Will Bitcoin reach $100,000 by Dec 31 2025?"),
        # Deliberately polymarket on BOTH sides — must reject as same-venue.
        "kalshi_venue_override": "polymarket",
        "kalshi": _poly("Will Bitcoin reach $100,000 by Dec 31 2025?"),
        "expected_reason": event_similarity.REJECT_SAME_VENUE,
        "notes": "Same-venue pair (no cross-venue arb possible).",
    },
]


# 3 borderline pairs. Hand-tuned to score in (0.4, 0.7): partial entity
# overlap, identical year, ambiguous wording. Values were probed against
# the live T77 scorer; if the scorer is re-tuned future Claude should
# re-probe and edit these — the ranges are intentionally wide.
BORDERLINE_PAIRS: list[dict[str, Any]] = [
    {
        "pair_id": "borderline-openai-ipo-2026",
        "poly": _poly("Will the OpenAI IPO happen in 2026?"),
        "kalshi": _kalshi("Will OpenAI go public in 2026?"),
        # ~0.585 in current scorer
    },
    {
        "pair_id": "borderline-spacex-starship-orbit-2025",
        "poly": _poly("Will SpaceX launch Starship to orbit by Dec 31 2025?"),
        "kalshi": _kalshi("Will SpaceX successfully send Starship to orbit by year-end 2025?"),
        # ~0.446 in current scorer
    },
    {
        "pair_id": "borderline-anthropic-fundraise-2026",
        "poly": _poly("Will Anthropic raise more than $5 billion in 2026?"),
        "kalshi": _kalshi("Will Anthropic close a $5B funding round by Dec 31 2026?"),
        # ~0.468 in current scorer
    },
]


# ---------------------------------------------------------------------------
# Pipeline helpers — exercise the full T76 → T77 → T78 chain in production
# orientation: build raw payload, run build_market_desc, run score_match, run
# audit_pairs.
# ---------------------------------------------------------------------------


def _run_pipeline(
    poly_payload: dict[str, Any],
    kalshi_payload: dict[str, Any],
    *,
    kalshi_venue: str = "kalshi",
) -> event_similarity.SimilarityScore:
    """Full T76 → T77 pipeline: raw payload → MarketDesc → SimilarityScore."""

    a = event_similarity.build_market_desc(poly_payload, "polymarket")
    b = event_similarity.build_market_desc(kalshi_payload, kalshi_venue)
    return event_similarity.score_match(a, b)


def _audit_pipeline(case: dict[str, Any]) -> dict[str, Any]:
    """T78 audit harness wrapper — what the CSV row looks like."""

    pairs = [
        {
            "pair_id": case["pair_id"],
            "poly_title": case["poly"].get("question") or case["poly"].get("title"),
            "kalshi_title": case["kalshi"].get("title") or case["kalshi"].get("question"),
            "poly_slug": case["poly"].get("slug", ""),
            "kalshi_ticker": case["kalshi"].get("ticker", ""),
            "profit_pct": 1.0,
            "cost": 0.1,
            "source": "w11_35_fixture",
        }
    ]
    rows = audit_arb_matches.audit_pairs(pairs)
    assert len(rows) == 1
    return rows[0]


# ---------------------------------------------------------------------------
# Part 1 — Good pairs must score > 0.7 with NO rejection reason
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [pytest.param(c, id=c["pair_id"]) for c in GOOD_PAIRS],
)
def test_good_pairs_score_above_threshold(case: dict[str, Any]) -> None:
    """Every good pair must score > 0.7 and not be rejected."""

    result = _run_pipeline(case["poly"], case["kalshi"])
    assert result.rejected_reason is None, (
        f"Good pair {case['pair_id']!r} was rejected: reason={result.rejected_reason!r}"
    )
    assert result.total > case["expected_min_score"], (
        f"Good pair {case['pair_id']!r} only scored {result.total:.3f}, "
        f"expected > {case['expected_min_score']}"
    )


# ---------------------------------------------------------------------------
# Part 2 — The 5 user-flagged false positives must reject with expected reason
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [pytest.param(c, id=c["pair_id"]) for c in USER_FLAGGED_FALSE_POSITIVES],
)
def test_user_flagged_false_positives_rejected(case: dict[str, Any]) -> None:
    """Every user-flagged false positive must be rejected with the right reason.

    The W11 brief calls out these specific patterns. Regressions here are
    high-priority bugs because they're the exact failures the matcher was
    designed to fix.
    """

    kalshi_venue = case.get("kalshi_venue_override", "kalshi")
    result = _run_pipeline(case["poly"], case["kalshi"], kalshi_venue=kalshi_venue)

    assert result.rejected_reason == case["expected_reason"], (
        f"False positive {case['pair_id']!r}: expected reason="
        f"{case['expected_reason']!r}, got reason={result.rejected_reason!r}, "
        f"score={result.total:.3f}"
    )
    # Hard rejects zero out the score by construction.
    assert result.total == 0.0, (
        f"Hard-reject must zero the score; got {result.total:.3f} for {case['pair_id']!r}"
    )


def test_all_five_user_flagged_categories_covered() -> None:
    """Sanity: the 5 user-flagged false-positive categories are each exercised.

    Trump 2024 vs 2028 (window), Fed Mar vs Jun (window), BTC $80k vs $90k
    (threshold), US Senate vs FL Senate (jurisdiction), and same-venue dup.
    """

    reasons = {c["expected_reason"] for c in USER_FLAGGED_FALSE_POSITIVES}
    assert event_similarity.REJECT_WINDOW_NO_OVERLAP in reasons
    assert event_similarity.REJECT_THRESHOLD_MISMATCH in reasons
    assert event_similarity.REJECT_JURISDICTION_MISMATCH in reasons
    assert event_similarity.REJECT_SAME_VENUE in reasons
    # And every reason is in the canonical REJECT_REASONS taxonomy.
    for r in reasons:
        assert r in event_similarity.REJECT_REASONS


# ---------------------------------------------------------------------------
# Part 3 — Borderline pairs land in (0.4, 0.7), no rejection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [pytest.param(c, id=c["pair_id"]) for c in BORDERLINE_PAIRS],
)
def test_borderline_pairs_in_ambiguous_band(case: dict[str, Any]) -> None:
    """Borderline pairs should score in (0.4, 0.7) — ambiguous-by-design.

    These are NOT clear matches and NOT clear non-matches. The matcher
    should expose them to humans for review, NOT auto-accept or auto-reject.
    """

    result = _run_pipeline(case["poly"], case["kalshi"])
    assert result.rejected_reason is None, (
        f"Borderline pair {case['pair_id']!r} was hard-rejected unexpectedly: "
        f"reason={result.rejected_reason!r}"
    )
    assert 0.4 < result.total < 0.7, (
        f"Borderline pair {case['pair_id']!r} scored {result.total:.3f}, expected in (0.4, 0.7)"
    )


# ---------------------------------------------------------------------------
# Part 4 — T78 audit harness consumes the pipeline output correctly
# ---------------------------------------------------------------------------


def test_audit_pipeline_consumes_good_pair() -> None:
    """T78 audit_pairs() correctly normalises a good-pair score."""

    row = _audit_pipeline(GOOD_PAIRS[0])
    assert row["pair_id"] == GOOD_PAIRS[0]["pair_id"]
    assert row["score"] > 0.7
    assert row["rejected"] is False
    # No rejection reason — should be empty string per _normalise_score_result.
    assert row["reason"] in {"", None}


def test_audit_pipeline_consumes_false_positive() -> None:
    """T78 audit_pairs() captures the expected rejection reason."""

    for case in USER_FLAGGED_FALSE_POSITIVES:
        # Same-venue case can't go through audit_pairs directly because the
        # harness hardcodes polymarket/kalshi venues; skip that one — it's
        # covered by test_user_flagged_false_positives_rejected.
        if case["pair_id"] == "fp-same-venue-dup":
            continue
        row = _audit_pipeline(case)
        assert row["rejected"] is True, f"{case['pair_id']!r} not rejected by audit"
        assert row["reason"] == case["expected_reason"], (
            f"{case['pair_id']!r}: audit reason={row['reason']!r}, "
            f"expected {case['expected_reason']!r}"
        )
        assert row["score"] == 0.0


def test_audit_pipeline_produces_correct_csv(tmp_path: Path) -> None:
    """End-to-end: build pairs from all fixtures, audit, write CSV, verify."""

    pairs: list[dict[str, Any]] = []
    for c in GOOD_PAIRS + USER_FLAGGED_FALSE_POSITIVES + BORDERLINE_PAIRS:
        # Skip same-venue (audit harness will rename venue to kalshi).
        if c["pair_id"] == "fp-same-venue-dup":
            continue
        pairs.append(
            {
                "pair_id": c["pair_id"],
                "poly_title": c["poly"].get("question") or c["poly"].get("title"),
                "kalshi_title": c["kalshi"].get("title") or c["kalshi"].get("question"),
                "poly_slug": c["poly"].get("slug", ""),
                "kalshi_ticker": c["kalshi"].get("ticker", ""),
                "profit_pct": 2.5,
                "cost": 0.1,
                "source": "w11_35_fixture",
            }
        )

    rows = audit_arb_matches.audit_pairs(pairs)
    csv_path = tmp_path / "w11-35-audit.csv"
    audit_arb_matches.write_csv(rows, csv_path)

    # Header parity
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        loaded = list(reader)
        fieldnames = reader.fieldnames

    assert fieldnames == audit_arb_matches.CSV_COLUMNS
    assert len(loaded) == len(pairs)

    # Every row well-formed
    for r in loaded:
        assert r["pair_id"]
        assert r["poly_title"]
        assert r["kalshi_title"]
        float(r["score"])  # round-trips
        assert r["rejected"] in {"True", "False"}

    # Tallies partition correctly
    summary = audit_arb_matches.summarise(rows)
    assert summary["total"] == len(pairs)
    assert (
        summary["high_confidence"] + summary["borderline"] + summary["rejected"] == summary["total"]
    )

    # We supplied 5 good + 4 reject (excluding same-venue) + 3 borderline = 12
    assert summary["high_confidence"] >= 5
    assert summary["rejected"] >= 4
    assert summary["borderline"] >= 3


def test_audit_pipeline_blacklist_proposals(tmp_path: Path) -> None:
    """T78 blacklist proposals only contain the rejected rows."""

    pairs = [
        {
            "pair_id": c["pair_id"],
            "poly_title": c["poly"].get("question") or c["poly"].get("title"),
            "kalshi_title": c["kalshi"].get("title") or c["kalshi"].get("question"),
            "poly_slug": "",
            "kalshi_ticker": "",
            "profit_pct": 1.5,
            "cost": 0.2,
            "source": "w11_35_fixture",
        }
        for c in USER_FLAGGED_FALSE_POSITIVES
        if c["pair_id"] != "fp-same-venue-dup"
    ]
    rows = audit_arb_matches.audit_pairs(pairs)
    out_path = tmp_path / "blacklist.json"
    n = audit_arb_matches.write_blacklist_proposals(rows, out_path)
    payload = json.loads(out_path.read_text())

    assert n == len(payload) == len(pairs)
    for entry in payload:
        assert entry["reason"] in event_similarity.REJECT_REASONS
        assert entry["score"] == 0.0
        assert entry["pair_id"]


# ---------------------------------------------------------------------------
# Part 5 — Performance: 100 pairs processed in < 1 s wall-clock
# ---------------------------------------------------------------------------


def _build_synthetic_pair_universe(n: int = 100) -> list[dict[str, Any]]:
    """Build ``n`` synthetic pairs by replicating the fixture set."""

    base = GOOD_PAIRS + USER_FLAGGED_FALSE_POSITIVES + BORDERLINE_PAIRS
    out: list[dict[str, Any]] = []
    for i in range(n):
        src = base[i % len(base)]
        # Skip same-venue when funneling through audit_pairs (it overrides venues).
        if src["pair_id"] == "fp-same-venue-dup":
            src = base[(i + 1) % len(base)]
        out.append(
            {
                "pair_id": f"{src['pair_id']}-{i}",
                "poly_title": src["poly"].get("question") or src["poly"].get("title"),
                "kalshi_title": src["kalshi"].get("title") or src["kalshi"].get("question"),
                "poly_slug": "",
                "kalshi_ticker": "",
                "profit_pct": 1.0,
                "cost": 0.05,
                "source": "w11_35_perf",
            }
        )
    return out


def test_pipeline_processes_100_pairs_under_1s() -> None:
    """100 pair-fixtures must process end-to-end in under 1 second.

    Generous margin: the matcher is pure-Python regex + dataclass work, so
    ~10k pairs/s on a modern laptop is typical. The 1s budget catches
    accidental O(n²) regressions (e.g. someone adding a per-pair file read).
    """

    pairs = _build_synthetic_pair_universe(100)
    assert len(pairs) == 100

    start = time.perf_counter()
    rows = audit_arb_matches.audit_pairs(pairs)
    elapsed = time.perf_counter() - start

    assert len(rows) == 100
    assert elapsed < 1.0, f"Pipeline too slow: {elapsed:.3f}s for 100 pairs"


# ---------------------------------------------------------------------------
# Part 6 — Concurrent calls are thread-safe
# ---------------------------------------------------------------------------


def test_pipeline_is_thread_safe() -> None:
    """16 worker threads × 25 pairs each must produce identical rows.

    The matcher is pure-Python with no shared mutable state, but this test
    exists to lock the contract in — a future regression that adds a
    module-level mutable cache would break here.
    """

    pairs = _build_synthetic_pair_universe(25)

    # Single-threaded baseline.
    baseline = audit_arb_matches.audit_pairs(pairs)
    baseline_key = [(r["pair_id"], r["score"], r["rejected"], r["reason"]) for r in baseline]

    errors: list[BaseException] = []
    results_lock = threading.Lock()
    all_results: list[list[tuple[Any, ...]]] = []

    def worker() -> list[tuple[Any, ...]]:
        try:
            rows = audit_arb_matches.audit_pairs(pairs)
            return [(r["pair_id"], r["score"], r["rejected"], r["reason"]) for r in rows]
        except BaseException as exc:  # pragma: no cover — explicit failure surface
            with results_lock:
                errors.append(exc)
            raise

    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(worker) for _ in range(16)]
        for fut in as_completed(futures):
            all_results.append(fut.result())

    assert not errors, f"Concurrent calls raised: {errors}"
    assert len(all_results) == 16
    for r in all_results:
        assert r == baseline_key, (
            "Concurrent run produced different output from single-threaded "
            "baseline; pipeline is NOT thread-safe."
        )


# ---------------------------------------------------------------------------
# Part 7 — Fixture self-checks (catch corrupted fixtures early)
# ---------------------------------------------------------------------------


def test_fixture_counts() -> None:
    """W11-35 requires exactly 5 good / 5 user-flagged / 3 borderline."""

    assert len(GOOD_PAIRS) == 5
    assert len(USER_FLAGGED_FALSE_POSITIVES) == 5
    assert len(BORDERLINE_PAIRS) == 3


def test_legacy_t78_fixture_still_loads() -> None:
    """Sanity: the T78 fixture catalogue is readable and well-shaped.

    The W11-35 pipeline test does not consume the T78 JSON directly (we
    embed our own fixtures here for clearer error messages), but if the
    T78 catalogue goes corrupt we want to surface it from this file too.
    """

    with FIXTURES_PATH.open() as fh:
        data = json.load(fh)
    assert isinstance(data.get("good"), list) and len(data["good"]) >= 5
    assert isinstance(data.get("bad"), list) and len(data["bad"]) >= 10
