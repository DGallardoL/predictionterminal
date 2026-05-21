"""Tests for ``scripts.aggregate_alpha_history``.

Uses mock report files written into a ``tmp_path`` to verify discovery,
section classification, bullet + table parsing, status inference, and the
end-to-end ``aggregate`` + ``build_output`` flow.

These tests never touch real ``docs/alpha-report-*.md`` files and never
write to ``docs/static/`` — every fixture is fully sandboxed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Make the scripts/ directory importable as a top-level module.
_API_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_API_ROOT))

from scripts import aggregate_alpha_history as agg

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_report(docs_root: Path, version: int, body: str, subdir: str | None = None) -> Path:
    base = docs_root / subdir if subdir else docs_root
    base.mkdir(parents=True, exist_ok=True)
    p = base / f"alpha-report-v{version}.md"
    p.write_text(body, encoding="utf-8")
    return p


_REPORT_V18_BODY = """# Alpha Deployment Report v18

## Currently Deployable (B_VALIDATED+)

- **Election-binary momentum** — long the leading binary contract. Net Sharpe ~1.4 (4Q stable). Allocation 8%.
- **Fed-decision straddle proxy** — VIX overlay using Polymarket FOMC odds. Net Sharpe ~1.1. Allocation 7%.
- **Calendar lambda-ratio** (`polymarket_calendar_lambda_v1`) — A_STRUCTURAL. Sharpe 1.19. Allocation 12%.

## Demoted / Anti-Alpha (this report)

- **Recession-odds defensive long** (was A_GOLD v15). Demoted to D_ARCHIVE.
- **Crypto-ETF approval drift** — Demoted to D_ARCHIVE.

## Methodology Notes

This section talks about Sharpe but should not produce strategy entries.
"""

_REPORT_V19_BODY = """# Alpha Deployment Report v19

## 3. Currently Deployable

| Strategy | Tier | Net Sharpe | Allocation |
|---|---|---|---|
| Calendar lambda-ratio | A_STRUCTURAL | 1.19 | 12% |
| Election-binary momentum | B_VALIDATED | 1.4 | 8% |
| Earnings-surprise odds vs IV | B_VALIDATED | 1.3 | 4% |

## 4. Anti-Alpha (additions)

None added this cycle. The v18 anti-alpha list carries forward.

## 5. Pending Stress (CONDITIONAL)

- **Binary pricing mispricing (T84)**: CONDITIONAL pending T81/T82/T83 verdicts.
- **Cross-sectional momentum (W12-25)**: CONDITIONAL until 4-quarter gate clears.
"""


# ---------------------------------------------------------------------------
# slugify
# ---------------------------------------------------------------------------


class TestSlugify:
    def test_basic_lowercase_dash(self) -> None:
        assert agg.slugify("Calendar Lambda-Ratio") == "calendar-lambda-ratio"

    def test_strip_markdown_emphasis(self) -> None:
        assert agg.slugify("**Election-binary momentum**") == "election-binary-momentum"

    def test_collapses_punctuation(self) -> None:
        assert agg.slugify("Fed/Decision: straddle (proxy)") == "fed-decision-straddle-proxy"

    def test_empty_and_none(self) -> None:
        assert agg.slugify("") == ""
        assert agg.slugify(None) == ""  # type: ignore[arg-type]

    def test_already_slug(self) -> None:
        assert agg.slugify("calendar-lambda-ratio") == "calendar-lambda-ratio"


# ---------------------------------------------------------------------------
# discover_reports
# ---------------------------------------------------------------------------


class TestDiscoverReports:
    def test_returns_empty_when_no_reports(self, tmp_path: Path) -> None:
        assert agg.discover_reports(tmp_path) == []

    def test_finds_top_level_and_subdir(self, tmp_path: Path) -> None:
        _write_report(tmp_path, 18, "# v18")
        _write_report(tmp_path, 17, "# v17", subdir="alpha-reports")
        _write_report(tmp_path, 16, "# v16", subdir="alpha-reports")
        reports = agg.discover_reports(tmp_path)
        labels = [label for label, _ in reports]
        assert labels == ["v16", "v17", "v18"]

    def test_top_level_overrides_subdir(self, tmp_path: Path) -> None:
        # Both paths exist; top-level should win.
        sub = _write_report(tmp_path, 18, "# subdir v18", subdir="alpha-reports")
        top = _write_report(tmp_path, 18, "# top v18")
        reports = agg.discover_reports(tmp_path)
        assert reports == [("v18", top)]
        assert sub.exists()  # not deleted; just not selected

    def test_sort_numeric_not_lex(self, tmp_path: Path) -> None:
        # Lexicographic sort would put v10 before v2.  We need numeric.
        for v in (2, 9, 10, 15, 100):
            _write_report(tmp_path, v, f"# v{v}", subdir="alpha-reports")
        reports = agg.discover_reports(tmp_path)
        labels = [label for label, _ in reports]
        assert labels == ["v2", "v9", "v10", "v15", "v100"]


# ---------------------------------------------------------------------------
# parse_report — section classification + entry extraction
# ---------------------------------------------------------------------------


class TestParseReport:
    def test_parses_deployable_bullets(self) -> None:
        entries = agg.parse_report(_REPORT_V18_BODY, "v18")
        slugs = {agg.slugify(e.raw_name): e for e in entries if e.section == "deployable"}
        assert "election-binary-momentum" in slugs
        e = slugs["election-binary-momentum"]
        assert e.status == "B_VALIDATED"
        assert e.sharpe == 1.4
        # 8% allocation -> 0.08 (with float rounding tolerance).
        assert e.allocation is not None and abs(e.allocation - 0.08) < 1e-9
        assert e.report == "v18"

    def test_parses_demoted_bullets_assigns_d_archive(self) -> None:
        entries = agg.parse_report(_REPORT_V18_BODY, "v18")
        demoted = [e for e in entries if e.section == "demoted"]
        names = {agg.slugify(e.raw_name) for e in demoted}
        assert "recession-odds-defensive-long" in names
        assert "crypto-etf-approval-drift" in names
        for e in demoted:
            assert e.status == "D_ARCHIVE"

    def test_parses_table_rows(self) -> None:
        entries = agg.parse_report(_REPORT_V19_BODY, "v19")
        deploy = {agg.slugify(e.raw_name): e for e in entries if e.section == "deployable"}
        # Header row "Strategy" must be filtered.
        assert "strategy" not in deploy
        assert "calendar-lambda-ratio" in deploy
        e = deploy["calendar-lambda-ratio"]
        assert e.status == "A_STRUCTURAL"
        assert e.sharpe == 1.19
        assert e.allocation is not None and abs(e.allocation - 0.12) < 1e-9

    def test_explicit_tier_word_overrides_section_default(self) -> None:
        body = (
            "## Currently Deployable\n\n"
            "- **Test strat A** — explicit A_STRUCTURAL, Sharpe 2.0, allocation 10%.\n"
        )
        entries = agg.parse_report(body, "v18")
        assert len(entries) == 1
        assert entries[0].status == "A_STRUCTURAL"
        assert entries[0].sharpe == 2.0

    def test_conditional_section_classification(self) -> None:
        entries = agg.parse_report(_REPORT_V19_BODY, "v19")
        cond = [e for e in entries if e.section == "conditional"]
        assert len(cond) == 2
        names = {agg.slugify(e.raw_name) for e in cond}
        assert "binary-pricing-mispricing-t84" in names

    def test_no_entries_in_methodology_section(self) -> None:
        entries = agg.parse_report(_REPORT_V18_BODY, "v18")
        # No entry's raw_name should be the methodology paragraph text.
        for e in entries:
            assert "methodology" not in agg.slugify(e.raw_name)

    def test_empty_body_returns_empty_list(self) -> None:
        assert agg.parse_report("", "v0") == []

    def test_negative_sharpe_parses(self) -> None:
        body = "## Currently Deployable\n\n- **Test neg** — Sharpe -0.89 was bad; allocation 0%.\n"
        entries = agg.parse_report(body, "v17")
        assert len(entries) == 1
        assert entries[0].sharpe == -0.89


# ---------------------------------------------------------------------------
# aggregate — multi-report time series
# ---------------------------------------------------------------------------


class TestAggregate:
    def test_builds_time_series_across_reports(self, tmp_path: Path) -> None:
        _write_report(tmp_path, 18, _REPORT_V18_BODY)
        _write_report(tmp_path, 19, _REPORT_V19_BODY)
        reports = agg.discover_reports(tmp_path)
        payload = agg.aggregate(reports)

        assert "strategies" in payload
        strategies = payload["strategies"]
        assert isinstance(strategies, dict)

        # Calendar lambda-ratio appears in both reports.
        cal = strategies.get("calendar-lambda-ratio")
        assert cal is not None
        # Should have entries for v18 AND v19, sorted.
        versions = [entry["report"] for entry in cal]
        assert versions == ["v18", "v19"]
        # Status should match — v19 explicitly tags A_STRUCTURAL in the table.
        assert cal[-1]["status"] == "A_STRUCTURAL"

    def test_chronological_order_across_double_digit_versions(self, tmp_path: Path) -> None:
        # Reports v9 and v10 should sort numerically.
        body = "## Currently Deployable\n- **Same-name strat** — Sharpe 1.0, allocation 1%.\n"
        _write_report(tmp_path, 9, body, subdir="alpha-reports")
        _write_report(tmp_path, 10, body, subdir="alpha-reports")
        payload = agg.aggregate(agg.discover_reports(tmp_path))
        history = payload["strategies"]["same-name-strat"]
        assert [h["report"] for h in history] == ["v9", "v10"]

    def test_dedup_within_single_report(self, tmp_path: Path) -> None:
        # Same strategy mentioned in both deployable and conditional → only
        # the first hit (deployable) is kept for that report.
        body = (
            "## Currently Deployable\n\n"
            "- **Dup strat** — Sharpe 1.0.\n\n"
            "## Pending Stress\n\n"
            "- **Dup strat** — CONDITIONAL.\n"
        )
        _write_report(tmp_path, 11, body, subdir="alpha-reports")
        payload = agg.aggregate(agg.discover_reports(tmp_path))
        history = payload["strategies"]["dup-strat"]
        assert len(history) == 1
        assert history[0]["section"] == "deployable"

    def test_source_reports_only_lists_reports_with_entries(self, tmp_path: Path) -> None:
        _write_report(tmp_path, 1, "# empty\n\nNo deployable section here.\n")
        _write_report(tmp_path, 18, _REPORT_V18_BODY)
        payload = agg.aggregate(agg.discover_reports(tmp_path))
        # v1 had no parseable section → should NOT appear.
        assert "v1" not in payload["source_reports"]
        assert "v18" in payload["source_reports"]


# ---------------------------------------------------------------------------
# build_output + main (end-to-end)
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_build_output_includes_generated_at_iso(self, tmp_path: Path) -> None:
        _write_report(tmp_path, 18, _REPORT_V18_BODY)
        payload = agg.build_output(tmp_path)
        ts = payload["generated_at"]
        assert isinstance(ts, str)
        assert ts.endswith("Z")
        assert "T" in ts  # ISO-8601 datetime form

    def test_main_writes_output_file(self, tmp_path: Path) -> None:
        _write_report(tmp_path, 18, _REPORT_V18_BODY)
        out = tmp_path / "alpha-history.json"
        rc = agg.main(
            [
                "--docs-root",
                str(tmp_path),
                "--output",
                str(out),
            ]
        )
        assert rc == 0
        assert out.exists()
        data = json.loads(out.read_text())
        assert "strategies" in data
        assert "generated_at" in data
        assert "source_reports" in data
        assert "v18" in data["source_reports"]

    def test_main_dry_run_does_not_write(self, tmp_path: Path) -> None:
        _write_report(tmp_path, 18, _REPORT_V18_BODY)
        out = tmp_path / "should-not-exist.json"
        rc = agg.main(
            [
                "--docs-root",
                str(tmp_path),
                "--output",
                str(out),
                "--dry-run",
            ]
        )
        assert rc == 0
        assert not out.exists()

    def test_main_print_flag_outputs_json(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _write_report(tmp_path, 18, _REPORT_V18_BODY)
        out = tmp_path / "alpha-history.json"
        agg.main(
            [
                "--docs-root",
                str(tmp_path),
                "--output",
                str(out),
                "--print",
            ]
        )
        captured = capsys.readouterr().out
        assert '"strategies"' in captured
        # Must be valid JSON.
        parsed = json.loads(captured)
        assert "v18" in parsed["source_reports"]


# ---------------------------------------------------------------------------
# Status inference table
# ---------------------------------------------------------------------------


class TestStatusInference:
    @pytest.mark.parametrize(
        ("text", "section", "expected"),
        [
            ("explicit A_GOLD here", "deployable", "A_GOLD"),
            ("This is B_VALIDATED, allocation 5%", "deployable", "B_VALIDATED"),
            ("Demoted to D_ARCHIVE", "demoted", "D_ARCHIVE"),
            ("CONDITIONAL pending verdict", "conditional", "CONDITIONAL"),
            ("no tier mentioned at all", "deployable", "B_VALIDATED"),  # section default
            ("no tier mentioned at all", "demoted", "D_ARCHIVE"),
            ("no tier mentioned at all", "conditional", "CONDITIONAL"),
            ("no tier mentioned at all", "weird", "UNKNOWN"),
        ],
    )
    def test_infer_status_table(self, text: str, section: str, expected: str) -> None:
        assert agg._infer_status(text, section) == expected
