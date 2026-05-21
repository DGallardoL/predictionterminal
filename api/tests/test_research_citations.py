"""Tests for ``pfm.research.citations_router`` (W12-22).

The router scrapes ``docs/alpha-reports/alpha-report-v*.md``, ``docs/adrs/*.md``,
and ``docs/regression-methodology-improvements.md`` into a deduplicated
bibliography. All tests build a synthetic ``docs/`` tree under ``tmp_path`` and
point the router at it via the ``PFM_CITATIONS_DOCS_ROOT`` env var.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Synthetic fixture content
# ---------------------------------------------------------------------------

ALPHA_REPORT_V18 = """# Alpha Report v18 — Wave-5 Demotions

**Generated**: 2026-05-12 overnight autopilot.

## Methodology
We apply the Bailey & López de Prado (2014) deflated-Sharpe correction.
The HAC standard errors follow Newey and West (1987). For the multiple-
testing scan we cite [Benjamini & Hochberg 1995] as the BH-FDR reference.

## References
- Bailey, D. H. and López de Prado, M. (2014). *The deflated Sharpe ratio.* J. Portfolio Management 40(5): 94–107.
"""

ALPHA_REPORT_V17 = """# Alpha Report v17 — Honest Reckoning

**Generated**: 2026-05-02 wave-5 audit.

Romano & Wolf (2005) extend the reality check. Hamilton (1989) is foundational.
Wave 5 (2026) brought stress harness X — should be filtered as noise.
"""

ALPHA_REPORT_OLD_V3 = """# Alpha Report v3 — Older

We mention Bailey & López de Prado (2014) again here.
"""

ADR_0010 = """# ADR-0010 Anti-alpha rule

Apply the Bailey & López de Prado (2014) deflated-Sharpe correction in
`pfm/quant/deflated_sharpe.py` using the empirical skewness and kurtosis.

We follow Newey and West (1987) HAC for variance.
"""

ADR_0003 = """# ADR-0003 HAC Newey-West

Newey and West (1987) provide the canonical heteroskedasticity-and-
autocorrelation consistent covariance estimator.
"""

REG_METHODS_IMPROVEMENTS = """# Regression methodology improvements (T79)

## 1. Background
Belloni, A. and Chernozhukov, V. (2013) propose post-LASSO inference.
Bailey & López de Prado (2014) deflated-Sharpe is the inspiration.

## 6. Selected references

- Belloni, A. and Chernozhukov, V. (2013). *Least squares after model selection in high-dimensional sparse models.* Bernoulli, 19(2): 521–547.
- Koenker, R. (2005). *Quantile Regression.* Cambridge University Press.
- Koenker, R. and Machado, J. A. F. (1999). *Goodness of fit and related inference processes for quantile regression.* JASA 94(448): 1296–1310.
- Bailey, D. H. and López de Prado, M. (2014). *The deflated Sharpe ratio.* Journal of Portfolio Management 40(5): 94–107.
- Zou, H. and Hastie, T. (2005). *Regularization and variable selection via the elastic net.* JRSS-B 67(2): 301–320.
- Hamilton, J. D. (1989). *A new approach to the economic analysis of nonstationary time series and the business cycle.* Econometrica 57(2): 357–384.
"""

ADR_WITH_BIBTEX = """# ADR-0011 Cache stampede

We adopt the singleflight approach.

```bibtex
@article{neweywest1987,
  author = {Newey, Whitney K. and West, Kenneth D.},
  year = {1987},
  title = {A Simple, Positive Semi-definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix}
}
```
"""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_docs_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Build a synthetic docs tree and point the router at it."""
    docs = tmp_path / "docs"
    (docs / "alpha-reports").mkdir(parents=True)
    (docs / "adrs").mkdir(parents=True)

    (docs / "alpha-reports" / "alpha-report-v18.md").write_text(ALPHA_REPORT_V18, encoding="utf-8")
    (docs / "alpha-reports" / "alpha-report-v17.md").write_text(ALPHA_REPORT_V17, encoding="utf-8")
    # Older versions live in docs/ root in the real repo too.
    (docs / "alpha-report-v3.md").write_text(ALPHA_REPORT_OLD_V3, encoding="utf-8")

    (docs / "adrs" / "ADR-0010-anti-alpha-rule.md").write_text(ADR_0010, encoding="utf-8")
    (docs / "adrs" / "0003-hac-newey-west.md").write_text(ADR_0003, encoding="utf-8")
    (docs / "adrs" / "ADR-0011-bibtex.md").write_text(ADR_WITH_BIBTEX, encoding="utf-8")
    (docs / "regression-methodology-improvements.md").write_text(
        REG_METHODS_IMPROVEMENTS, encoding="utf-8"
    )

    # Files that should be ignored by the scanner.
    (docs / "README.md").write_text("# Unrelated — should be skipped.\n")
    (docs / "alpha-report.md").write_text("# Untagged — no version, skip.\n")
    (docs / "alpha-reports" / "alpha-tier-regen-report-2026-05-09.md").write_text(
        "# Tier regen — wrong filename pattern.\n"
    )

    monkeypatch.setenv("PFM_CITATIONS_DOCS_ROOT", str(tmp_path))
    return tmp_path


@pytest.fixture
def client(fake_docs_root: Path) -> TestClient:
    """Reload the module so env-driven path resolver picks up the temp root."""
    if "pfm.research.citations_router" in sys.modules:
        importlib.reload(sys.modules["pfm.research.citations_router"])
    else:
        import pfm.research.citations_router  # noqa: F401
    from pfm.research.citations_router import _clear_cache, router

    _clear_cache()
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _by_key(payload: dict) -> dict:
    return {c["key"]: c for c in payload["citations"]}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_endpoint_returns_envelope(client: TestClient) -> None:
    """Top-level shape: checked_at + citations + count."""
    resp = client.get("/research/citations")
    assert resp.status_code == 200
    payload = resp.json()
    assert set(payload.keys()) == {"checked_at", "citations", "count"}
    assert isinstance(payload["citations"], list)
    assert payload["count"] == len(payload["citations"])
    # ISO8601 Zulu format
    assert payload["checked_at"].endswith("Z")
    assert "T" in payload["checked_at"]


def test_bailey_lopez_de_prado_merges_across_sources(client: TestClient) -> None:
    """A single paper cited in 4 docs collapses to one row with 4 sources."""
    payload = client.get("/research/citations").json()
    citations = _by_key(payload)
    assert "bailey-lopez-de-prado-2014" in citations
    entry = citations["bailey-lopez-de-prado-2014"]
    assert entry["year"] == 2014
    # Surnames present (López normalised to ASCII in the *key* but preserved
    # in the human-facing authors list).
    assert "Bailey" in entry["authors"]
    assert any("L" in a and "pez" in a.lower() for a in entry["authors"])
    referenced = set(entry["referenced_in"])
    # Bibliography line in regression-methodology + inline in v18 + ADR-0010
    # + v3 + inline in the same regression-methodology body = at least 4
    # distinct labels.
    assert "alpha-report-v18.md" in referenced
    assert "alpha-report-v3.md" in referenced
    assert "ADR-0010" in referenced
    assert "regression-methodology-improvements.md" in referenced
    # Bibliography line populates the title
    assert entry["title"] and "deflated sharpe" in entry["title"].lower()


def test_newey_west_inline_and_bibtex(client: TestClient) -> None:
    """Newey & West cited inline (ADR-0010, ADR-0003) and via bibtex."""
    citations = _by_key(client.get("/research/citations").json())
    assert "newey-west-1987" in citations
    entry = citations["newey-west-1987"]
    assert entry["year"] == 1987
    refs = set(entry["referenced_in"])
    assert "ADR-0010" in refs
    assert "ADR-0003" in refs
    assert "ADR-0011" in refs  # bibtex block lives here


def test_bibliography_line_provides_title(client: TestClient) -> None:
    """An italic ``*Title.*`` in the bullet list becomes ``citation.title``."""
    citations = _by_key(client.get("/research/citations").json())
    belloni = citations.get("belloni-chernozhukov-2013")
    assert belloni is not None
    assert belloni["title"] is not None
    assert "Least squares" in belloni["title"]
    assert belloni["year"] == 2013


def test_bracket_citation_recognised(client: TestClient) -> None:
    """``[Benjamini Hochberg 1995]`` becomes a citation row."""
    citations = _by_key(client.get("/research/citations").json())
    assert "benjamini-hochberg-1995" in citations
    bh = citations["benjamini-hochberg-1995"]
    assert bh["year"] == 1995
    assert "alpha-report-v18.md" in bh["referenced_in"]
    assert "Benjamini" in bh["authors"]


def test_false_positive_wave_year_is_filtered(client: TestClient) -> None:
    """``Wave 5 (2026)`` and similar prose should NOT become a citation."""
    citations = _by_key(client.get("/research/citations").json())
    # No citation should have a year of 2026 from the Wave-5 line.
    bad = [c for c in citations.values() if c["authors"] == ["Wave"]]
    assert bad == []


def test_count_matches_unique_keys(client: TestClient) -> None:
    """``count`` equals the number of unique citation keys."""
    payload = client.get("/research/citations").json()
    keys = [c["key"] for c in payload["citations"]]
    assert payload["count"] == len(keys) == len(set(keys))


def test_referenced_in_uses_short_labels(client: TestClient) -> None:
    """Alpha reports keep ``.md`` filename; ADRs collapse to ``ADR-####``."""
    citations = _by_key(client.get("/research/citations").json())
    nw = citations["newey-west-1987"]
    refs = set(nw["referenced_in"])
    # ADR labels look like 'ADR-0003' (no .md suffix), alpha-reports keep .md
    assert any(r.startswith("ADR-") and not r.endswith(".md") for r in refs)
    bald = citations["bailey-lopez-de-prado-2014"]
    assert any(r.startswith("alpha-report-v") and r.endswith(".md") for r in bald["referenced_in"])


def test_response_is_cached_for_one_hour(client: TestClient, fake_docs_root: Path) -> None:
    """Two successive GETs return the same ``checked_at`` thanks to the cache."""
    first = client.get("/research/citations").json()
    # Mutate the underlying docs — new ADR with an extra citation.
    new_adr = fake_docs_root / "docs" / "adrs" / "ADR-0099-extra.md"
    new_adr.write_text("# Extra\n\nWe cite Foobar (2020).\n", encoding="utf-8")
    second = client.get("/research/citations").json()
    # Cache means count is unchanged on the second hit.
    assert second["count"] == first["count"]
    assert second["checked_at"] == first["checked_at"]
    # After cache clear, the new citation is picked up.
    from pfm.research.citations_router import _clear_cache

    _clear_cache()
    third = client.get("/research/citations").json()
    assert third["count"] == first["count"] + 1
    third_keys = {c["key"] for c in third["citations"]}
    assert "foobar-2020" in third_keys


def test_alpha_report_pattern_filters_other_files(client: TestClient, fake_docs_root: Path) -> None:
    """The ``alpha-tier-regen-report-…`` file must be ignored."""
    payload = client.get("/research/citations").json()
    for c in payload["citations"]:
        for ref in c["referenced_in"]:
            assert "alpha-tier-regen-report" not in ref


def test_citations_sorted_by_first_author_then_year(client: TestClient) -> None:
    """Citations come back sorted by surname then by year."""
    citations = client.get("/research/citations").json()["citations"]
    sort_keys = [(c["authors"][0].lower() if c["authors"] else "z", c["year"]) for c in citations]
    assert sort_keys == sorted(sort_keys)


def test_empty_docs_root_returns_empty_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pointing at an empty directory yields ``count == 0``, no errors."""
    monkeypatch.setenv("PFM_CITATIONS_DOCS_ROOT", str(tmp_path))
    if "pfm.research.citations_router" in sys.modules:
        importlib.reload(sys.modules["pfm.research.citations_router"])
    else:
        import pfm.research.citations_router  # noqa: F401
    from pfm.research.citations_router import _clear_cache, router

    _clear_cache()
    app = FastAPI()
    app.include_router(router)
    payload = TestClient(app).get("/research/citations").json()
    assert payload["count"] == 0
    assert payload["citations"] == []
