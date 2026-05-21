"""Test suite for T78 — arb match-quality audit.

Verifies:
1. The 10 hand-curated false-positive pairs all get rejected (score < 0.4 or
   ``rejected=True``) by :func:`pfm.arb_matching.event_similarity.score_match`.
2. The 5 hand-curated known-good pairs all score > 0.7.
3. The audit script writes a well-formed CSV with the documented columns.
4. The script's summary tallies sum correctly.

If T76 (date_extractor) or T77 (event_similarity) is missing at test time,
the whole module is skipped with a helpful message — the test will run
unchanged once those modules land.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
API_SRC = REPO_ROOT / "api" / "src"
if str(API_SRC) not in sys.path:  # pragma: no cover — path-bootstrapping
    sys.path.insert(0, str(API_SRC))

# Skip the whole module if T76/T77 haven't landed yet.
event_similarity = pytest.importorskip(
    "pfm.arb_matching.event_similarity",
    reason=(
        "T78 depends on T76 (pfm.arb_matching.date_extractor) + T77 "
        "(pfm.arb_matching.event_similarity). Skipping until they land."
    ),
)

# Import the audit script as a module (it lives under api/scripts/).
SCRIPTS = REPO_ROOT / "api" / "scripts"
if str(SCRIPTS) not in sys.path:  # pragma: no cover
    sys.path.insert(0, str(SCRIPTS))
import audit_arb_matches

FIXTURES_PATH = Path(__file__).parent / "fixtures" / "arb_match_known_false_positives.json"


@pytest.fixture(scope="module")
def fixtures() -> dict:
    with FIXTURES_PATH.open() as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def score_match():
    return event_similarity.score_match


@pytest.fixture(scope="module")
def MarketDesc():
    return event_similarity.MarketDesc


def _build(MarketDescCls, title: str, venue: str) -> object:
    return audit_arb_matches.build_market_desc(
        MarketDescCls,
        title,
        venue=venue,
        slug="",
        build_helper=getattr(event_similarity, "build_market_desc", None),
    )


def _score(MarketDesc, score_match, poly_title: str, kalshi_title: str) -> dict:
    a = _build(MarketDesc, poly_title, "polymarket")
    b = _build(MarketDesc, kalshi_title, "kalshi")
    return audit_arb_matches._normalise_score_result(score_match(a, b))


# ---------------------------------------------------------------------------
# Part 1 — All 10 hand-curated false positives must be REJECTED
# ---------------------------------------------------------------------------


def test_fixture_file_has_required_shape(fixtures: dict) -> None:
    assert isinstance(fixtures.get("bad"), list)
    assert isinstance(fixtures.get("good"), list)
    assert len(fixtures["bad"]) >= 10, "need at least 10 known-bad pairs"
    assert len(fixtures["good"]) >= 5, "need at least 5 known-good pairs"
    for entry in fixtures["bad"] + fixtures["good"]:
        assert entry["pair_id"]
        assert entry["poly_title"]
        assert entry["kalshi_title"]
    # Gap catalogue is OK to be empty but each entry must still be shaped.
    for entry in fixtures.get("known_t77_gaps", []):
        assert entry["pair_id"]
        assert entry["poly_title"]
        assert entry["kalshi_title"]
        assert entry.get("gap_explanation"), (
            "every known_t77_gap entry must explain WHY current T77 misses it"
        )


@pytest.mark.parametrize(
    "case",
    [pytest.param(c, id=c["pair_id"]) for c in json.load(FIXTURES_PATH.open())["bad"]],
)
def test_known_false_positives_rejected(case: dict, MarketDesc, score_match) -> None:
    """Every hand-curated false positive must be rejected (score<0.4 OR
    ``rejected=True``)."""
    result = _score(MarketDesc, score_match, case["poly_title"], case["kalshi_title"])
    assert result["rejected"] or result["score"] < 0.4, (
        f"False positive {case['pair_id']!r} was NOT rejected: "
        f"score={result['score']}, reason={result['reason']!r}"
    )


# ---------------------------------------------------------------------------
# Part 1b — Previously-known T77 gaps, FIXED by T76b. These are user-flagged
# false positives that the OLD T76+T77 windows_overlap semantics missed
# because half-open ``(None, latest)`` windows always intersected via
# ``date.min``. T76b rewrites :func:`windows_overlap` to require strict
# latest-date proximity for same-shape half-open windows AND for two point
# dates, so these five cases now reject correctly.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(c, id=c["pair_id"])
        for c in json.load(FIXTURES_PATH.open()).get("known_t77_gaps", [])
    ],
)
def test_known_t77_gaps_should_eventually_reject(case: dict, MarketDesc, score_match) -> None:
    result = _score(MarketDesc, score_match, case["poly_title"], case["kalshi_title"])
    assert result["rejected"] or result["score"] < 0.4, (
        f"T77 gap {case['pair_id']!r} regressed: score={result['score']}, "
        f"reason={result['reason']!r}. T76b's windows_overlap should reject "
        f"it via {case.get('expected_reason', 'resolution_window_no_overlap')!r}."
    )


# ---------------------------------------------------------------------------
# Part 2 — All 5 hand-curated true positives must score > 0.7
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case",
    [pytest.param(c, id=c["pair_id"]) for c in json.load(FIXTURES_PATH.open())["good"]],
)
def test_known_true_positives_score_high(case: dict, MarketDesc, score_match) -> None:
    result = _score(MarketDesc, score_match, case["poly_title"], case["kalshi_title"])
    assert not result["rejected"], (
        f"True positive {case['pair_id']!r} was rejected: reason={result['reason']!r}"
    )
    assert result["score"] > 0.7, (
        f"True positive {case['pair_id']!r} only scored {result['score']:.3f}, expected > 0.7"
    )


# ---------------------------------------------------------------------------
# Part 3 — CSV output is well-formed
# ---------------------------------------------------------------------------


def test_audit_csv_is_well_formed(tmp_path: Path, fixtures: dict) -> None:
    # Build a synthetic pair list directly (don't depend on live arb scanner).
    pairs = []
    for c in fixtures["bad"][:3] + fixtures["good"][:2]:
        pairs.append(
            {
                "pair_id": c["pair_id"],
                "poly_title": c["poly_title"],
                "kalshi_title": c["kalshi_title"],
                "poly_slug": "",
                "kalshi_ticker": "",
                "profit_pct": 1.0,
                "cost": 0.5,
                "source": "fixture",
            }
        )
    rows = audit_arb_matches.audit_pairs(pairs)
    csv_path = tmp_path / "audit.csv"
    audit_arb_matches.write_csv(rows, csv_path)

    assert csv_path.exists()
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        loaded = list(reader)
    # Header check
    assert reader.fieldnames == audit_arb_matches.CSV_COLUMNS
    # Row-count parity
    assert len(loaded) == len(pairs)
    # Every row has the expected keys (DictReader guarantees this from header,
    # but we double-check no rogue values)
    for r in loaded:
        assert r["pair_id"]
        assert r["poly_title"]
        assert r["kalshi_title"]
        # score column round-trips as a float
        float(r["score"])
        # rejected as bool literal
        assert r["rejected"] in {"True", "False"}


def test_summarise_tallies_match_total(fixtures: dict) -> None:
    pairs = []
    for c in fixtures["bad"][:5] + fixtures["good"][:3]:
        pairs.append(
            {
                "pair_id": c["pair_id"],
                "poly_title": c["poly_title"],
                "kalshi_title": c["kalshi_title"],
                "poly_slug": "",
                "kalshi_ticker": "",
                "profit_pct": 2.5,
                "cost": 0.1,
                "source": "fixture",
            }
        )
    rows = audit_arb_matches.audit_pairs(pairs)
    summary = audit_arb_matches.summarise(rows)
    assert summary["total"] == len(pairs)
    # Buckets are mutually exclusive — must partition exactly
    assert (
        summary["high_confidence"] + summary["borderline"] + summary["rejected"] == summary["total"]
    )
    assert len(summary["top10_worst"]) <= 10


def test_blacklist_proposals_written_only_when_flag(tmp_path: Path, fixtures: dict) -> None:
    pairs = [
        {
            "pair_id": c["pair_id"],
            "poly_title": c["poly_title"],
            "kalshi_title": c["kalshi_title"],
            "poly_slug": "",
            "kalshi_ticker": "",
            "profit_pct": 0.5,
            "cost": 0.1,
            "source": "fixture",
        }
        for c in fixtures["bad"][:3]
    ]
    rows = audit_arb_matches.audit_pairs(pairs)
    out_path = tmp_path / "proposals.json"
    n = audit_arb_matches.write_blacklist_proposals(rows, out_path)
    assert out_path.exists()
    payload = json.loads(out_path.read_text())
    assert isinstance(payload, list)
    assert n == len(payload)
    # Every entry shape
    for entry in payload:
        assert {"pair_id", "reason", "score", "poly_title", "kalshi_title", "proposed_at"} <= set(
            entry.keys()
        )


# ---------------------------------------------------------------------------
# Part 4 — script CLI entry-point smoke test (doesn't hit network, uses
# the dashboard_state.json on disk if present, otherwise gracefully exits)
# ---------------------------------------------------------------------------


def test_load_pairs_from_dashboard_state_handles_missing(tmp_path: Path) -> None:
    pairs = audit_arb_matches.load_pairs_from_dashboard_state(tmp_path / "nope.json")
    assert pairs == []


def test_load_pairs_from_dashboard_state_reads_fixture(tmp_path: Path) -> None:
    fake = tmp_path / "dashboard_state.json"
    fake.write_text(
        json.dumps(
            {
                "opportunities": [
                    {
                        "name": "Will Trump win 2024?",
                        "arb_key": "ABC_yes_0x123",
                        "poly_slug": "trump-wins-2024",
                        "kalshi_ticker": "PRES-2024-T",
                        "profit_pct": 1.23,
                        "cost": 0.05,
                    }
                ]
            }
        )
    )
    pairs = audit_arb_matches.load_pairs_from_dashboard_state(fake)
    assert len(pairs) == 1
    p = pairs[0]
    assert p["pair_id"] == "ABC_yes_0x123"
    assert p["poly_slug"] == "trump-wins-2024"
    assert p["kalshi_ticker"] == "PRES-2024-T"
    assert p["source"] == "dashboard_state"
