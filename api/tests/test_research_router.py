"""Tests for ``pfm.research.router``.

The router lists ``docs/alpha-reports/alpha-report-v*.md`` as JSON cards and
returns individual bodies via ``GET /research/reports/{version}``. All tests
use a synthetic docs directory pointed to via the ``PFM_RESEARCH_DOCS_DIR``
environment variable so they are independent of the live repo state.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

REPORT_V18 = """# Alpha Report v18 — Wave-5 Demotions + 2026Q1 Robustness

**Generated**: 2026-05-12 overnight autopilot.
**Predecessors**: v17 (wave-5 stress reckoning).

This is the production-book lock. Three deployable strategies survive,
six are archived, and the watchlist holds three more. Honest numbers only.

## 2 · Deployable strategies (production book)

### 2.1 Calendar λ-ratio
Some prose.

### 2.2 Equity-coint tech basket
More prose.

## 4 · Anti-alpha graveyard

These all failed quarterly stability.

### 4.1 Recession-odds defensive sector long
Regime trade.
"""

REPORT_V17 = """# Alpha Report v17 — Honest Reckoning

**Generated**: 2026-05-02 wave-5 audit.

The headline: wave-5 lands 6 of 8 robustness verdicts.

## Deployable
One survivor.

## Graveyard
Six archived.
"""

REPORT_V2_WITH_FRONTMATTER = """---
version: 2
title: "Alpha Report v2 — Custom Frontmatter Title"
published_at: 2026-01-15
---
# Alpha Report v2 — Rigorous Statistical Validation

Three pairs survive all five stages of validation. The headline is brutal:
honest scrutiny eliminates most candidates.

## Deployable winners
Three pairs.

## Anti-alpha bucket
Several entries.
"""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_docs_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Create a temp docs dir with three synthetic reports + extra noise."""
    monkeypatch.setenv("PFM_RESEARCH_DOCS_DIR", str(tmp_path))
    (tmp_path / "alpha-report-v18.md").write_text(REPORT_V18, encoding="utf-8")
    (tmp_path / "alpha-report-v17.md").write_text(REPORT_V17, encoding="utf-8")
    (tmp_path / "alpha-report-v2.md").write_text(REPORT_V2_WITH_FRONTMATTER, encoding="utf-8")
    # Files that should be ignored by the glob.
    (tmp_path / "alpha-report.md").write_text("# Untagged report\n\nBody.\n")
    (tmp_path / "README.md").write_text("# README\n")
    (tmp_path / "alpha-tier-regen-report-2026-05-09.md").write_text("# Tier regen\n")
    return tmp_path


@pytest.fixture
def client(fake_docs_dir: Path) -> TestClient:
    """Reload the router module against the fake docs dir + clear its cache."""
    # Force a fresh import so the env-var-driven path resolver is rebuilt.
    if "pfm.research.router" in sys.modules:
        importlib.reload(sys.modules["pfm.research.router"])
    else:
        import pfm.research.router  # noqa: F401
    from pfm.research.router import _clear_cache, router

    _clear_cache()
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


# ---------------------------------------------------------------------------
# /research/reports — index endpoint
# ---------------------------------------------------------------------------


def test_list_reports_returns_only_versioned_files(client: TestClient) -> None:
    """The non-versioned ``alpha-report.md`` and unrelated files are skipped."""
    resp = client.get("/research/reports")
    assert resp.status_code == 200
    payload = resp.json()
    assert "reports" in payload
    versions = [r["version"] for r in payload["reports"]]
    assert versions == [18, 17, 2], "expected three parsed reports, version-desc"


def test_list_reports_sorted_by_version_desc(client: TestClient) -> None:
    """v18 must come before v17, which must come before v2."""
    resp = client.get("/research/reports")
    versions = [r["version"] for r in resp.json()["reports"]]
    assert versions == sorted(versions, reverse=True)


def test_card_has_required_fields(client: TestClient) -> None:
    """Each card carries the contract fields specified in T31."""
    card = client.get("/research/reports").json()["reports"][0]
    expected = {
        "id",
        "version",
        "title",
        "published_at",
        "summary",
        "deployable_count",
        "anti_alpha_count",
        "path",
    }
    assert expected.issubset(card.keys())
    assert card["id"] == "v18"
    assert card["version"] == 18
    assert card["path"] == "docs/alpha-reports/alpha-report-v18.md"
    assert card["title"].startswith("Alpha Report v18")


def test_card_published_at_from_generated_line(client: TestClient) -> None:
    """``**Generated**: 2026-05-12`` -> published_at='2026-05-12'."""
    cards = {c["version"]: c for c in client.get("/research/reports").json()["reports"]}
    assert cards[18]["published_at"] == "2026-05-12"
    assert cards[17]["published_at"] == "2026-05-02"


def test_card_published_at_from_frontmatter(client: TestClient) -> None:
    """YAML frontmatter ``published_at: 2026-01-15`` is honored when present."""
    cards = {c["version"]: c for c in client.get("/research/reports").json()["reports"]}
    assert cards[2]["published_at"] == "2026-01-15"


def test_card_title_from_frontmatter_wins(client: TestClient) -> None:
    """When YAML frontmatter sets ``title``, it overrides the H1 fallback."""
    cards = {c["version"]: c for c in client.get("/research/reports").json()["reports"]}
    assert cards[2]["title"] == "Alpha Report v2 — Custom Frontmatter Title"


def test_summary_derived_from_first_chars_after_h1(client: TestClient) -> None:
    """The card summary is the first ~200 chars of prose following the H1."""
    cards = {c["version"]: c for c in client.get("/research/reports").json()["reports"]}
    summary = cards[18]["summary"]
    assert summary.startswith("This is the production-book lock")
    # Cap is 200 chars (+ optional ellipsis).
    assert len(summary) <= 210
    # Meta lines like the bold **Predecessors** line must not appear.
    assert "Predecessors" not in summary


def test_deployable_and_anti_alpha_counts(client: TestClient) -> None:
    """Heading heuristics pick up deployable + anti-alpha sub-sections."""
    cards = {c["version"]: c for c in client.get("/research/reports").json()["reports"]}
    # v18 has one '## 2 · Deployable strategies' + 2 sub-deployable headings
    # (### 2.1 + ### 2.2 are not flagged — they don't contain 'deployable').
    # The '## 4 · Anti-alpha graveyard' heading is flagged for anti-alpha,
    # AND '### 4.1 Recession-odds defensive sector long' has no anti-alpha
    # keyword — so anti count is 1.
    assert cards[18]["deployable_count"] >= 1
    assert cards[18]["anti_alpha_count"] >= 1
    # v17 has '## Deployable' AND '## Graveyard'.
    assert cards[17]["deployable_count"] >= 1
    assert cards[17]["anti_alpha_count"] >= 1


# ---------------------------------------------------------------------------
# /research/reports/{version} — body endpoint
# ---------------------------------------------------------------------------


def test_get_report_returns_markdown_body(client: TestClient) -> None:
    """``GET /research/reports/v18`` returns the raw markdown."""
    resp = client.get("/research/reports/v18")
    assert resp.status_code == 200
    assert "text/markdown" in resp.headers["content-type"]
    assert "Alpha Report v18" in resp.text
    assert "Calendar λ-ratio" in resp.text


def test_get_report_accepts_numeric_version(client: TestClient) -> None:
    """Both ``18`` and ``v18`` are accepted as the version path param."""
    r1 = client.get("/research/reports/18")
    r2 = client.get("/research/reports/v18")
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r1.text == r2.text


def test_get_report_404_for_missing_version(client: TestClient) -> None:
    """A version with no backing file returns 404."""
    resp = client.get("/research/reports/v9999")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


def test_get_report_400_for_malformed_version(client: TestClient) -> None:
    """Garbage like ``vabc`` is rejected with HTTP 400."""
    resp = client.get("/research/reports/vabc")
    assert resp.status_code == 400


def test_get_report_html_format_graceful(client: TestClient) -> None:
    """``?format=html`` returns HTML when ``markdown`` is installed, else MD.

    Either outcome is acceptable — we just verify the response is 200 and
    the content-type matches one of the two supported branches.
    """
    resp = client.get("/research/reports/v18?format=html")
    assert resp.status_code == 200
    ctype = resp.headers["content-type"]
    assert "text/html" in ctype or "text/markdown" in ctype


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def test_index_is_cached_between_calls(client: TestClient, fake_docs_dir: Path) -> None:
    """Adding a new file after the first call does NOT appear until cache expiry.

    This proves the TTL cache is engaged — without it the new file would
    show up immediately.
    """
    first = client.get("/research/reports").json()
    assert len(first["reports"]) == 3

    # Drop a fresh report on disk.
    (fake_docs_dir / "alpha-report-v99.md").write_text(
        "# Alpha Report v99\n\n**Generated**: 2026-12-31\n\nBody.\n",
        encoding="utf-8",
    )

    second = client.get("/research/reports").json()
    # Cache hit: still 3, NOT 4.
    assert len(second["reports"]) == 3


def test_cache_clear_picks_up_new_files(client: TestClient, fake_docs_dir: Path) -> None:
    """After manually clearing the cache, new files appear immediately."""
    client.get("/research/reports")  # warm
    (fake_docs_dir / "alpha-report-v99.md").write_text(
        "# Alpha Report v99\n\n**Generated**: 2026-12-31\n\nBody.\n",
        encoding="utf-8",
    )
    from pfm.research.router import _clear_cache

    _clear_cache()

    after = client.get("/research/reports").json()
    versions = [r["version"] for r in after["reports"]]
    assert 99 in versions
    assert versions[0] == 99  # sort desc puts the new one first


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_missing_docs_dir_returns_empty_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-existent docs dir yields ``{"reports": []}`` (no crash)."""
    missing = tmp_path / "does-not-exist"
    monkeypatch.setenv("PFM_RESEARCH_DOCS_DIR", str(missing))
    if "pfm.research.router" in sys.modules:
        importlib.reload(sys.modules["pfm.research.router"])
    from pfm.research.router import _clear_cache, router

    _clear_cache()
    app = FastAPI()
    app.include_router(router)
    resp = TestClient(app).get("/research/reports")
    assert resp.status_code == 200
    assert resp.json() == {"reports": []}


def test_router_mounts_against_real_docs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sanity check: the router works against the live repo docs dir too.

    This guards against accidental import-time breakage from path resolution
    bugs. We don't assert on counts because reports are added over time —
    we just verify the call succeeds and at least one card comes back.
    """
    # Unset any override left over from other tests (pytest tmp fixtures
    # don't propagate, but be defensive).
    monkeypatch.delenv("PFM_RESEARCH_DOCS_DIR", raising=False)
    if "pfm.research.router" in sys.modules:
        importlib.reload(sys.modules["pfm.research.router"])
    from pfm.research.router import _clear_cache, router

    _clear_cache()
    app = FastAPI()
    app.include_router(router)
    resp = TestClient(app).get("/research/reports")
    assert resp.status_code == 200
    reports = resp.json()["reports"]
    # The repo ships docs/alpha-reports/alpha-report-v*.md, so at least one
    # card MUST come back. If this test fails on a stripped-down checkout,
    # set PFM_RESEARCH_DOCS_DIR to skip the live check.
    assert len(reports) >= 1
    assert all("version" in r for r in reports)
