"""Aggregate alpha-deployability history across all ``docs/alpha-report-vN.md``.

Walks the repo's ``docs/`` tree (the canonical reports live in
``docs/alpha-reports/`` plus newer ones, v18+, at the top of ``docs/``),
parses the "Currently Deployable" and "Demoted / Anti-Alpha" sections of
each report, and emits a time-series JSON consumable by the frontend.

Schema::

    {
      "generated_at": "<ISO-8601 UTC>",
      "source_reports": ["v15", "v16", ..., "v20"],
      "strategies": {
        "<slug>": [
          {
            "report": "v18",
            "status": "B_VALIDATED",
            "sharpe": 1.4,
            "allocation": 0.08,
            "raw_name": "Election-binary momentum",
            "section": "deployable"
          },
          ...
        ],
        ...
      }
    }

Status values are restricted to the project's tier vocabulary
(``A_GOLD``, ``A_STRUCTURAL``, ``B_VALIDATED``, ``C_TENTATIVE``,
``D_ARCHIVE``, ``CONDITIONAL``, ``UNKNOWN``) so the frontend can render
consistent tier pills without doing its own normalisation.

Usage::

    python -m scripts.aggregate_alpha_history          # write docs/static/alpha-history.json
    python -m scripts.aggregate_alpha_history --print  # also dump to stdout
    python -m scripts.aggregate_alpha_history --output /tmp/out.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import re
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

# -----------------------------------------------------------------------------
# Repo paths
# -----------------------------------------------------------------------------

_API_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _API_ROOT.parent
DEFAULT_DOCS_ROOT = _REPO_ROOT / "docs"
DEFAULT_OUTPUT = _REPO_ROOT / "docs" / "static" / "alpha-history.json"

# -----------------------------------------------------------------------------
# Tier vocabulary
# -----------------------------------------------------------------------------

KNOWN_TIERS: tuple[str, ...] = (
    "A_GOLD",
    "A_STRUCTURAL",
    "B_VALIDATED",
    "C_TENTATIVE",
    "D_ARCHIVE",
    "CONDITIONAL",
)

# Mapping from common free-form verdict words to canonical tiers.
_VERDICT_ALIASES: dict[str, str] = {
    "deploy": "A_GOLD",
    "deployable": "B_VALIDATED",
    "watchlist": "C_TENTATIVE",
    "archive": "D_ARCHIVE",
    "demoted": "D_ARCHIVE",
    "anti-alpha": "D_ARCHIVE",
    "anti_alpha": "D_ARCHIVE",
    "pending": "CONDITIONAL",
    "conditional": "CONDITIONAL",
    "tentative": "C_TENTATIVE",
    "validated": "B_VALIDATED",
    "structural": "A_STRUCTURAL",
    "gold": "A_GOLD",
}


# -----------------------------------------------------------------------------
# Dataclasses
# -----------------------------------------------------------------------------


@dataclass
class StrategyEntry:
    """One report-row for one strategy."""

    report: str
    status: str
    sharpe: float | None = None
    allocation: float | None = None
    raw_name: str = ""
    section: str = ""  # "deployable" | "demoted" | "conditional"
    note: str = ""

    def to_jsonable(self) -> dict[str, object]:
        out: dict[str, object] = {
            "report": self.report,
            "status": self.status,
        }
        if self.sharpe is not None:
            out["sharpe"] = round(self.sharpe, 4)
        if self.allocation is not None:
            out["allocation"] = round(self.allocation, 6)
        if self.raw_name:
            out["raw_name"] = self.raw_name
        if self.section:
            out["section"] = self.section
        if self.note:
            out["note"] = self.note
        return out


@dataclass
class ReportParse:
    """Everything we extracted from a single report file."""

    report: str
    path: Path
    entries: list[StrategyEntry] = field(default_factory=list)


# -----------------------------------------------------------------------------
# Slug-ification — turn "Election-binary momentum" into "election-binary-momentum"
# -----------------------------------------------------------------------------

_SLUG_KEEP = re.compile(r"[^a-z0-9]+")


def slugify(name: str) -> str:
    """Best-effort kebab-case slug for a strategy name.

    Rules:
    - lower-case
    - strip leading/trailing markdown emphasis (``*``, ``_``)
    - keep only ``[a-z0-9-]``; runs of other chars collapse to a single ``-``
    - trim leading/trailing ``-``
    - empty input → ``""``
    """
    if name is None:
        return ""
    s = name.strip()
    # Drop wrapping markdown emphasis.
    s = s.strip("*_`")
    s = s.lower()
    s = _SLUG_KEEP.sub("-", s)
    s = s.strip("-")
    return s


# -----------------------------------------------------------------------------
# File discovery
# -----------------------------------------------------------------------------

_REPORT_RE = re.compile(r"alpha-report-v(\d+)\.md$")


def discover_reports(docs_root: Path) -> list[tuple[str, Path]]:
    """Return ``[(version_label, path), ...]`` sorted by version number.

    Searches both ``docs/`` and ``docs/alpha-reports/`` non-recursively.  If
    the same version exists in both locations, the version at the top of
    ``docs/`` wins — that mirrors the codebase where v18+ live at the top
    and v2-v17 live in ``alpha-reports/``.
    """
    candidates: dict[int, tuple[int, Path]] = {}

    def _scan(dirpath: Path, priority: int) -> None:
        if not dirpath.is_dir():
            return
        for p in sorted(dirpath.iterdir()):
            m = _REPORT_RE.search(p.name)
            if not m:
                continue
            n = int(m.group(1))
            existing = candidates.get(n)
            # Strictly higher priority overrides; equal priority does NOT.
            if existing is None or priority > existing[0]:
                candidates[n] = (priority, p)

    _scan(docs_root / "alpha-reports", priority=0)
    _scan(docs_root, priority=1)

    return [(f"v{n}", candidates[n][1]) for n in sorted(candidates)]


# -----------------------------------------------------------------------------
# Markdown section extraction
# -----------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def _split_sections(md: str) -> list[tuple[int, str, list[str]]]:
    """Split markdown into ``(heading_level, heading_text, body_lines)`` triples."""
    sections: list[tuple[int, str, list[str]]] = []
    current_level = 0
    current_heading = ""
    current_body: list[str] = []
    for line in md.splitlines():
        m = _HEADING_RE.match(line)
        if m:
            if current_heading or current_body:
                sections.append((current_level, current_heading, current_body))
            current_level = len(m.group(1))
            current_heading = m.group(2).strip()
            current_body = []
        else:
            current_body.append(line)
    if current_heading or current_body:
        sections.append((current_level, current_heading, current_body))
    return sections


_DEPLOYABLE_RE = re.compile(r"currently\s+deployable|deployable", re.IGNORECASE)
_DEMOTED_RE = re.compile(r"demoted|anti[-\s]?alpha|archived|graveyard", re.IGNORECASE)
_CONDITIONAL_RE = re.compile(r"pending\s+stress|conditional|watchlist|pending", re.IGNORECASE)


def _classify_section(heading: str) -> str | None:
    if _DEPLOYABLE_RE.search(heading) and not _DEMOTED_RE.search(heading):
        return "deployable"
    if _DEMOTED_RE.search(heading):
        return "demoted"
    if _CONDITIONAL_RE.search(heading):
        return "conditional"
    return None


# -----------------------------------------------------------------------------
# Entry extraction
# -----------------------------------------------------------------------------

# Match "Sharpe ~1.4", "Sharpe 1.19", "Net Sharpe ~1.6", "Sharpe: -0.89"
_SHARPE_RE = re.compile(
    r"(?:net\s+)?sharpe\s*(?:est(?:imate)?\.?)?\s*[~≈]?\s*([+-]?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
# Match "allocation 5%", "5% allocation", "12%" (when in allocation column)
_PERCENT_RE = re.compile(r"([+-]?\d+(?:\.\d+)?)\s*%")
# Match an explicit tier mention.
_TIER_RE = re.compile(r"\b(A_GOLD|A_STRUCTURAL|B_VALIDATED|C_TENTATIVE|D_ARCHIVE|CONDITIONAL)\b")
# Strategy name pattern at start of bullet "- **<name>** — ..."
_BULLET_NAME_RE = re.compile(r"^\s*[-*]\s+\*\*([^*]+?)\*\*")
# Strategy name as the LAST quoted code-span "- ... (`name`)" — fallback.
_CODE_NAME_RE = re.compile(r"`([^`]+)`")


def _parse_sharpe(text: str) -> float | None:
    m = _SHARPE_RE.search(text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _parse_allocation(text: str) -> float | None:
    """Return allocation as a fraction (0-1).

    The text typically reads ``"allocation halved to 5%"`` or
    ``"12%"`` in a table cell.  We pick the FIRST percentage we find that
    appears near the word ``allocation`` if possible, otherwise the first.
    """
    # Prefer percentages adjacent to "allocation".
    near = re.search(
        r"alloc(?:ation)?[^%]{0,30}?([+-]?\d+(?:\.\d+)?)\s*%",
        text,
        re.IGNORECASE,
    )
    if near:
        try:
            return float(near.group(1)) / 100.0
        except ValueError:
            return None
    # Fallback: first percentage.
    m = _PERCENT_RE.search(text)
    if not m:
        return None
    try:
        return float(m.group(1)) / 100.0
    except ValueError:
        return None


def _infer_status(text: str, section_kind: str) -> str:
    """Choose a canonical tier from explicit mention or section context.

    Rules:
    1. If the text contains a *transition* phrase ("Demoted to X",
       "Promoted to X", "→ X") then X wins regardless of any earlier
       tier mention (which is typically a historical reference such as
       "(was A_GOLD v15)").
    2. Otherwise, if a section_kind == 'demoted' AND multiple tiers are
       mentioned, prefer the LAST one (the new tier, not the old one).
    3. Otherwise, return the first tier mentioned, or fall back to the
       section default, or 'UNKNOWN'.
    """
    # 1) Explicit transition.
    trans = re.search(
        r"(?:demoted|promoted|downgraded|upgraded|moved)\s+to\s+"
        r"(A_GOLD|A_STRUCTURAL|B_VALIDATED|C_TENTATIVE|D_ARCHIVE|CONDITIONAL)",
        text,
        re.IGNORECASE,
    )
    if trans:
        return trans.group(1).upper()
    arrow = re.search(
        r"(?:→|->)\s*(A_GOLD|A_STRUCTURAL|B_VALIDATED|C_TENTATIVE|D_ARCHIVE|CONDITIONAL)",
        text,
    )
    if arrow:
        return arrow.group(1).upper()
    # 2) Demoted-section policy: last tier wins.
    matches = list(_TIER_RE.finditer(text))
    if matches:
        if section_kind == "demoted":
            return matches[-1].group(1)
        return matches[0].group(1)
    # Free-form aliases.
    lower = text.lower()
    for word, tier in _VERDICT_ALIASES.items():
        if word in lower:
            return tier
    # Fall back to section default.
    if section_kind == "deployable":
        return "B_VALIDATED"
    if section_kind == "demoted":
        return "D_ARCHIVE"
    if section_kind == "conditional":
        return "CONDITIONAL"
    return "UNKNOWN"


def _extract_name(line: str) -> str:
    m = _BULLET_NAME_RE.search(line)
    if m:
        return m.group(1).strip()
    # Table row fallback: "| Strategy | ... |"
    if line.lstrip().startswith("|"):
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if cells:
            first = cells[0]
            # Strip markdown emphasis.
            first = re.sub(r"^[*_`]+|[*_`]+$", "", first).strip()
            # If the first cell has a (`slug`) suffix, drop it for display.
            return first
    # Code-span fallback.
    m = _CODE_NAME_RE.search(line)
    if m:
        return m.group(1)
    return ""


def _is_table_separator(line: str) -> bool:
    return bool(re.match(r"^\s*\|?\s*:?-{2,}", line))


def _table_rows(body: list[str]) -> list[tuple[list[str], list[str]]]:
    """Extract markdown-table data rows from ``body``.

    Returns a list of ``(headers, cells)`` tuples — one entry per data row.
    Both ``headers`` and ``cells`` are lower-cased / trimmed where helpful;
    ``headers`` carries the column titles of the table the row belongs to.
    """
    rows: list[tuple[list[str], list[str]]] = []
    in_table = False
    seen_header = False
    current_headers: list[str] = []
    for line in body:
        stripped = line.strip()
        if stripped.startswith("|") and "|" in stripped[1:]:
            if not in_table:
                in_table = True
                seen_header = False
                current_headers = []
            if _is_table_separator(stripped):
                seen_header = True
                continue
            cells = [c.strip() for c in stripped.strip("|").split("|")]
            if not seen_header:
                # First row of a table is the header.
                current_headers = [c.strip().lower() for c in cells]
                continue
            if cells:
                rows.append((current_headers, cells))
        else:
            in_table = False
            seen_header = False
            current_headers = []
    return rows


def _cell_to_float(cell: str) -> float | None:
    """Pull the first signed number out of a cell, return as float or None."""
    m = re.search(r"([+-]?\d+(?:\.\d+)?)", cell)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _parse_row_sharpe(headers: list[str], cells: list[str]) -> float | None:
    """Sharpe column lookup, then text-anywhere fallback."""
    for idx, h in enumerate(headers):
        if "sharpe" in h and idx < len(cells):
            v = _cell_to_float(cells[idx])
            if v is not None:
                return v
    return _parse_sharpe(" | ".join(cells))


def _parse_row_allocation(headers: list[str], cells: list[str]) -> float | None:
    """Allocation column lookup, then text-anywhere fallback."""
    for idx, h in enumerate(headers):
        if ("alloc" in h or h == "weight" or "%" in h) and idx < len(cells):
            v = _cell_to_float(cells[idx])
            if v is not None:
                return v / 100.0
    return _parse_allocation(" | ".join(cells))


def _entries_from_section(
    section_kind: str, body_lines: list[str], report: str
) -> list[StrategyEntry]:
    """Pull every strategy mention out of one classified section."""
    entries: list[StrategyEntry] = []

    # 1) Bullet lines
    for line in body_lines:
        if not _BULLET_NAME_RE.search(line):
            continue
        name = _extract_name(line)
        if not name:
            continue
        entries.append(
            StrategyEntry(
                report=report,
                status=_infer_status(line, section_kind),
                sharpe=_parse_sharpe(line),
                allocation=_parse_allocation(line),
                raw_name=name,
                section=section_kind,
            )
        )

    # 2) Table rows: first cell is the strategy name; scan whole row for
    #    tier / sharpe / allocation.
    for headers, cells in _table_rows(body_lines):
        if not cells:
            continue
        name = cells[0]
        # Skip empty / separator-ish first cells.
        if not name or set(name) <= {"-", " ", ":"}:
            continue
        # Skip rows whose first cell is obviously not a strategy (e.g.
        # "Method", "Allocation", "Caveats").
        if name.lower() in {
            "method",
            "allocation",
            "caveats",
            "test",
            "metric",
            "tier",
            "strategy",
        }:
            continue
        whole = " | ".join(cells)
        name_clean = re.sub(r"^[*_`]+|[*_`]+$", "", name).strip()
        if not name_clean:
            continue
        entries.append(
            StrategyEntry(
                report=report,
                status=_infer_status(whole, section_kind),
                sharpe=_parse_row_sharpe(headers, cells) or _parse_sharpe(whole),
                allocation=_parse_row_allocation(headers, cells) or _parse_allocation(whole),
                raw_name=name_clean,
                section=section_kind,
            )
        )

    return entries


# -----------------------------------------------------------------------------
# Public parsing API
# -----------------------------------------------------------------------------


def parse_report(text: str, report_label: str) -> list[StrategyEntry]:
    """Parse one report and return a (possibly empty) list of strategy entries."""
    sections = _split_sections(text)
    all_entries: list[StrategyEntry] = []
    for _level, heading, body in sections:
        kind = _classify_section(heading)
        if kind is None:
            continue
        all_entries.extend(_entries_from_section(kind, body, report_label))
    return all_entries


def aggregate(reports: Iterable[tuple[str, Path]]) -> dict[str, object]:
    """Aggregate parsed entries across all reports.

    Returns the JSON-ready output dict (without ``generated_at``).
    """
    by_slug: dict[str, list[StrategyEntry]] = {}
    source_reports: list[str] = []

    for label, path in reports:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:  # pragma: no cover — defensive
            logging.warning("could not read %s: %s", path, exc)
            continue
        entries = parse_report(text, label)
        if entries:
            source_reports.append(label)
        # Within one report, keep the FIRST occurrence of each slug
        # (typically the deployable mention wins over a later mention in
        # methodology/notes).
        seen_in_report: set[str] = set()
        for e in entries:
            slug = slugify(e.raw_name)
            if not slug:
                continue
            if slug in seen_in_report:
                continue
            seen_in_report.add(slug)
            by_slug.setdefault(slug, []).append(e)

    # Sort each strategy's history chronologically by version number.
    def _vnum(entry: StrategyEntry) -> int:
        m = re.match(r"v(\d+)", entry.report)
        return int(m.group(1)) if m else -1

    strategies_out: dict[str, list[dict[str, object]]] = {}
    for slug in sorted(by_slug):
        history = sorted(by_slug[slug], key=_vnum)
        strategies_out[slug] = [e.to_jsonable() for e in history]

    return {
        "source_reports": source_reports,
        "strategies": strategies_out,
    }


def build_output(docs_root: Path) -> dict[str, object]:
    reports = discover_reports(docs_root)
    payload = aggregate(reports)
    payload["generated_at"] = (
        dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )
    return payload


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Aggregate alpha-deployability history across all docs/alpha-report-vN.md"
            " into a single JSON file for the frontend."
        )
    )
    p.add_argument(
        "--docs-root",
        type=Path,
        default=DEFAULT_DOCS_ROOT,
        help=f"Root of the docs directory (default: {DEFAULT_DOCS_ROOT})",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT})",
    )
    p.add_argument(
        "--print",
        action="store_true",
        help="Also print the JSON payload to stdout.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute but do not write the output file.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args(argv)
    payload = build_output(args.docs_root)
    if args.print:
        print(json.dumps(payload, indent=2))
    if not args.dry_run:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        logging.info(
            "wrote %s — %d strategies across %d reports",
            args.output,
            len(payload.get("strategies", {})),
            len(payload.get("source_reports", [])),
        )
    return 0


# Public re-export for tests.
__all__ = [
    "KNOWN_TIERS",
    "ReportParse",
    "StrategyEntry",
    "aggregate",
    "build_output",
    "discover_reports",
    "main",
    "parse_report",
    "slugify",
]


# Silence "asdict imported but unused" if someone trims the module.
_ = asdict


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
