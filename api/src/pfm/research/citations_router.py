"""FastAPI router for the cross-document citations bibliography (T W12-22).

Endpoint
--------
* ``GET /research/citations`` — bibliography of papers referenced across all
  alpha reports + ADRs + ``docs/regression-methodology-improvements.md``.

The router scans:

* ``docs/alpha-reports/alpha-report-v*.md`` (also ``docs/alpha-report-v*.md`` —
  some older versions live in the docs root)
* ``docs/adrs/*.md``
* ``docs/regression-methodology-improvements.md``

and extracts citations in three styles:

1. **Bibliography-list entries** — bullet-style ``- Author, X. and Author, Y.
   (2014). *Title.* Journal …``  (the canonical form in
   ``regression-methodology-improvements.md`` and the alpha-report references
   sections).
2. **Inline ``Author (Year)`` / ``Author & Author (Year)`` / ``Author and Author
   (Year)``** — used inside prose ("the Bailey & López de Prado (2014)
   deflated-Sharpe correction in …").
3. **Bracket markdown citations** — ``[Author Year]`` or ``[Author, Year]`` —
   used by a small number of ADR drafts.

Each unique paper is keyed by ``<lowercased-first-author>[-<lowercased-second
-author>]-<year>`` so the same reference cited in multiple files collapses to
a single bibliography row with ``referenced_in`` listing every source.

The response is cached for 1 hour (per the task spec).

Integration note
----------------
This router is **not** auto-mounted from ``pfm.main``. To enable, claim the
``main.py:routes`` section (see ``.coordination/PROTOCOL-V2.md``) and add::

    from pfm.research.citations_router import router as _citations_router
    app.include_router(_citations_router)
"""

from __future__ import annotations

import os
import re
import threading
import time
import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Cache TTL for the citation index — 1 hour per spec.
_CACHE_TTL_SECONDS = 3600

# Repo-root anchored docs sub-paths the scanner reads from. All are resolved
# relative to the repo root computed in :func:`_docs_root`.
_DOCS_ALPHA_REPORT_DIRS: tuple[tuple[str, ...], ...] = (
    ("docs", "alpha-reports"),  # current home (v10+)
    ("docs",),  # older v2..v9 still here in this repo
)
_DOCS_ADR_DIRS: tuple[tuple[str, ...], ...] = (
    ("docs", "adrs"),
    ("docs", "ADRs"),  # spec mentions ADRs/ but actual dir is lowercase
)
_DOCS_EXTRA_FILES: tuple[tuple[str, ...], ...] = (
    ("docs", "regression-methodology-improvements.md"),
)

_ALPHA_REPORT_RE = re.compile(r"^alpha-report-v\d+\.md$", re.IGNORECASE)


# Year range we accept (anything outside is almost certainly not a citation
# year — e.g. ``(0.05)`` significance levels or ``(1)`` footnote markers).
_MIN_YEAR = 1900
_MAX_YEAR = 2100

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# 1. Inline ``Author (Year)`` / ``Author & Author (Year)`` / ``Author and Author
#    (Year)`` / ``Author, Author, & Author (Year)``. We allow letters,
#    apostrophes, hyphens, and multi-word last names like "López de Prado".
#
# Examples matched:
#   - Bailey & López de Prado (2014)
#   - Newey and West (1987)
#   - Hamilton (1989)
#   - MacKinnon, Haug & Michelis (1999)
#   - Romano & Wolf (2005)
_NAME_TOKEN = r"[A-ZÁÉÍÓÚÑÜ][A-Za-zÁÉÍÓÚÑÜáéíóúñü'’\-]+"
_MULTI_WORD_NAME = rf"{_NAME_TOKEN}(?:\s+(?:de|van|von|der|del|la|le|du)\s+{_NAME_TOKEN})?"
_INLINE_RE = re.compile(
    rf"(?<![A-Za-z])(?P<authors>{_MULTI_WORD_NAME}(?:(?:,?\s+(?:&|and|y)\s+|,\s+){_MULTI_WORD_NAME})*)"
    rf"\s+\((?P<year>\d{{4}})\)"
)

# 2. Bracket citations: ``[Author Year]`` or ``[Author, Year]`` or
#    ``[Author & Author Year]`` or ``[Author Author Year]``. We avoid eating
#    real markdown links like ``[text](url)`` by requiring the closing ``]``
#    not to be followed by ``(``. Author separator is "&"/"and"/comma/whitespace.
_BRACKET_RE = re.compile(
    rf"\[(?P<authors>{_MULTI_WORD_NAME}(?:(?:,?\s+(?:&|and)\s+|,?\s+){_MULTI_WORD_NAME})*),?\s+(?P<year>\d{{4}})\](?!\()"
)

# 3. Bibliography list rows: ``- Author, X. and Author, Y. (year). *Title.*``.
#    We extract the year and the *italic title* once we've matched a bullet.
_BIB_LINE_RE = re.compile(
    r"^\s*[-*]\s+(?P<authors>[^()]+?)\s+\((?P<year>\d{4})\)\.\s*"
    r"(?:\*(?P<title_star>[^*]+?)\*|_(?P<title_under>[^_]+?)_|\"(?P<title_quote>[^\"]+?)\")?"
)

# 4. Bibtex-like blocks (``@article{key, author = {...}, year = {...},
#    title = {...}, ...}``). The opener is matched by regex; the body is
#    sliced out by brace-counting in :func:`_iter_bibtex_blocks` so inner
#    ``{...}`` field values don't truncate the entry.
_BIBTEX_OPENER_RE = re.compile(
    r"@(?P<type>article|book|inproceedings|misc|techreport|phdthesis)\s*\{\s*"
    r"(?P<key>[^,\s]+)\s*,",
    re.IGNORECASE,
)
_BIBTEX_FIELD_RE = re.compile(
    r"(?P<field>author|year|title)\s*=\s*[{\"](?P<value>[^}\"]+)[}\"]",
    re.IGNORECASE,
)


def _iter_bibtex_blocks(text: str) -> list[str]:
    """Slice out every ``@type{ ... }`` block, handling nested braces."""
    blocks: list[str] = []
    for m in _BIBTEX_OPENER_RE.finditer(text):
        # ``m.end()`` points just after the comma; we need to find the
        # matching closing ``}`` of the outer block via brace counting.
        # Re-find the opening ``{`` first.
        open_pos = text.find("{", m.start())
        if open_pos < 0:
            continue
        depth = 1
        i = open_pos + 1
        while i < len(text) and depth > 0:
            ch = text[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            i += 1
        if depth != 0:
            continue
        blocks.append(text[open_pos + 1 : i - 1])
    return blocks


# ---------------------------------------------------------------------------
# TTL cache (thread-safe)
# ---------------------------------------------------------------------------


class _TTLCache:
    """Single-slot TTL cache keyed by the resolved repo root."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[str, tuple[float, dict[str, Any]]] = {}

    def get(self, key: str, ttl: float) -> dict[str, Any] | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            ts, value = entry
            if time.monotonic() - ts > ttl:
                self._entries.pop(key, None)
                return None
            return value

    def set(self, key: str, value: dict[str, Any]) -> None:
        with self._lock:
            self._entries[key] = (time.monotonic(), value)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


_CACHE = _TTLCache()


def _clear_cache() -> None:
    """Test hook — drop the cached index."""
    _CACHE.clear()


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def _docs_root() -> Path:
    """Return the repo root (or override) holding the ``docs/`` tree.

    Resolution order:

    1. ``PFM_CITATIONS_DOCS_ROOT`` env var — used by tests to point at a
       temp directory.
    2. Repo root computed relative to this file:
       ``api/src/pfm/research/citations_router.py`` is four levels deep,
       so ``parents[4]`` is the repo root.
    """
    override = os.environ.get("PFM_CITATIONS_DOCS_ROOT")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[4]


def _collect_source_files(root: Path) -> list[Path]:
    """Return every markdown file we will scrape, deduplicated."""
    seen: set[Path] = set()
    out: list[Path] = []

    def _add(p: Path) -> None:
        try:
            resolved = p.resolve()
        except OSError:
            return
        if resolved in seen:
            return
        seen.add(resolved)
        out.append(p)

    # Alpha reports
    for sub in _DOCS_ALPHA_REPORT_DIRS:
        d = root.joinpath(*sub)
        if d.is_dir():
            for entry in sorted(d.iterdir()):
                if entry.is_file() and _ALPHA_REPORT_RE.match(entry.name):
                    _add(entry)

    # ADRs
    for sub in _DOCS_ADR_DIRS:
        d = root.joinpath(*sub)
        if d.is_dir():
            for entry in sorted(d.iterdir()):
                if entry.is_file() and entry.suffix.lower() == ".md":
                    _add(entry)

    # Extra single files
    for sub in _DOCS_EXTRA_FILES:
        f = root.joinpath(*sub)
        if f.is_file():
            _add(f)

    return out


def _label_for(path: Path, root: Path) -> str:
    """Compact display label for the ``referenced_in`` list.

    * Alpha reports → ``"alpha-report-v18.md"``
    * ADRs        → ``"ADR-0010"`` (or ``"<filename-stem>"`` when not numbered)
    * Other       → ``"<filename>"``
    """
    name = path.name
    if _ALPHA_REPORT_RE.match(name):
        return name
    try:
        rel = path.resolve().relative_to(root.resolve())
        parts = rel.parts
    except ValueError:
        parts = (name,)
    if len(parts) >= 2 and parts[-2].lower() == "adrs":
        stem = path.stem
        m = re.match(r"^(?:ADR-)?(\d{4})", stem, re.IGNORECASE)
        if m:
            return f"ADR-{m.group(1)}"
        # Already starts with "ADR-" prefix?
        if stem.lower().startswith("adr-"):
            return stem
        return stem
    return name


# ---------------------------------------------------------------------------
# Citation extraction
# ---------------------------------------------------------------------------


def _normalise_author(token: str) -> str:
    """Strip initials, periods, and trailing commas to recover the surname.

    Examples
    --------
    >>> _normalise_author("Bailey, D. H.")
    'Bailey'
    >>> _normalise_author("López de Prado, M.")
    'López de Prado'
    >>> _normalise_author("van de Geer, S.")
    'van de Geer'
    >>> _normalise_author("Bailey")
    'Bailey'
    """
    s = token.strip()
    # Strip leading/trailing markdown emphasis and quotes/brackets.
    s = s.strip("*_`\"'[]()")
    s = s.rstrip(",.;:")
    # Drop trailing possessive ``'s`` so "Bailey & López de Prado's" → "Bailey & López de Prado"
    if s.endswith("'s") or s.endswith("’s"):
        s = s[:-2]
    # Drop trailing initials/given-names: "Bailey, D. H." → "Bailey",
    # "Newey, Whitney K." → "Newey". We treat anything after the FIRST comma
    # as "given names" if every token there looks like a name-piece or an
    # initial. Surnames with embedded commas are not used in this codebase.
    if "," in s:
        head, _, tail = s.partition(",")
        tail_clean = tail.strip().rstrip(".")
        tokens = [p for p in re.split(r"\s+", tail_clean) if p]

        def _is_given_name_piece(p: str) -> bool:
            core = p.rstrip(".")
            if not core:
                return True
            # Initial: "D" or "D."
            if len(core) <= 2 and core.isalpha():
                return True
            # Full given name starting with capital letter: "Whitney"
            return core[:1].isupper() and core.isalpha()

        if tokens and all(_is_given_name_piece(p) for p in tokens):
            s = head.strip()
    # Strip stray ``et al.`` markers
    s = re.sub(r"\s+et\s+al\.?$", "", s, flags=re.IGNORECASE).strip()
    return s


def _split_authors(raw: str) -> list[str]:
    """Split an author group like ``"Bailey & López de Prado"`` into surnames.

    Handles ``&``, ``and``, ``y`` (Spanish), commas, and Oxford-comma plus
    ampersand mixes. Initials are stripped via :func:`_normalise_author`.
    """
    if not raw:
        return []
    # Normalise separators to a single token we can split on.
    s = raw.strip()
    # ``Author A. and Author B.`` → ``Author A.|Author B.``
    s = re.sub(r"\s+(?:&|and|y)\s+", "|", s, flags=re.IGNORECASE)
    # Now split on commas too — but keep "Surname, F." together (those
    # have already been collapsed by the regex if they came via _BIB_LINE_RE,
    # and inline-style citations don't carry initials).
    parts: list[str] = []
    for chunk in s.split("|"):
        # If the chunk still contains a comma followed by uppercase initials,
        # keep it (e.g. "Bailey, D. H."). Otherwise split on commas — these
        # are "Surname, Surname, & Surname" Oxford forms.
        chunk = chunk.strip()
        if not chunk:
            continue
        # If the chunk is in "Surname, GivenName(s)" form, _normalise_author
        # will strip the trailing piece. Detect by checking whether the
        # comma-tail looks like a sequence of given-name tokens (initials
        # or capitalised words). When so, treat the chunk as a single
        # author rather than splitting on the comma.
        if "," in chunk:
            _head, _, tail = chunk.partition(",")
            tail_tokens = [p for p in re.split(r"\s+", tail.strip().rstrip(".")) if p]

            def _is_given(p: str) -> bool:
                core = p.rstrip(".")
                if not core:
                    return True
                if len(core) <= 2 and core.isalpha():
                    return True
                return core[:1].isupper() and core.isalpha()

            if tail_tokens and all(_is_given(p) for p in tail_tokens):
                parts.append(_normalise_author(chunk))
                continue
        if "," in chunk:
            for sub in chunk.split(","):
                sub = sub.strip()
                if sub:
                    parts.append(_normalise_author(sub))
        else:
            parts.append(_normalise_author(chunk))
    # Drop empties and dedupe preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for p in parts:
        if not p or p.lower() in {"and", "y", "the", "et al"}:
            continue
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _slug(text: str) -> str:
    """Lowercase ASCII slug suitable for citation keys."""
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_ = nfkd.encode("ascii", "ignore").decode("ascii").lower()
    ascii_ = re.sub(r"[^a-z0-9]+", "-", ascii_).strip("-")
    return ascii_


def _make_key(authors: list[str], year: int) -> str:
    """Compose the bibliography key from up to the first two surnames."""
    pieces = [_slug(a) for a in authors[:2] if _slug(a)]
    if not pieces:
        return f"unknown-{year}"
    return "-".join(pieces) + f"-{year}"


def _valid_year(y: int) -> bool:
    return _MIN_YEAR <= y <= _MAX_YEAR


# A blocklist of inline-match "author" phrases that are common false positives
# in this codebase. (E.g. "Wave 5 (2026)" is not a citation.)
_FALSE_POSITIVE_AUTHORS = {
    "wave",
    "fig",
    "figure",
    "table",
    "section",
    "appendix",
    "eq",
    "equation",
    "chapter",
    "see",
    "note",
    "step",
}


def _is_plausible_author_group(authors: list[str]) -> bool:
    """Reject inline matches whose first token is obviously not a surname."""
    if not authors:
        return False
    first = authors[0]
    if first.lower() in _FALSE_POSITIVE_AUTHORS:
        return False
    # Surnames are at least 2 chars
    return not len(first) < 2


def _extract_from_bibtex(text: str, label: str, sink: dict[str, dict[str, Any]]) -> None:
    """Pull citations from ``@article{...}`` style blocks, if any."""
    for body in _iter_bibtex_blocks(text):
        fields: dict[str, str] = {}
        for fm in _BIBTEX_FIELD_RE.finditer(body):
            fields[fm.group("field").lower()] = fm.group("value").strip()
        if "year" not in fields:
            continue
        try:
            year = int(fields["year"])
        except ValueError:
            continue
        if not _valid_year(year):
            continue
        authors_raw = fields.get("author", "")
        authors = _split_authors(authors_raw)
        if not _is_plausible_author_group(authors):
            continue
        key = _make_key(authors, year)
        entry = sink.setdefault(
            key,
            {
                "key": key,
                "authors": authors,
                "year": year,
                "title": fields.get("title"),
                "referenced_in": [],
            },
        )
        if entry.get("title") is None and fields.get("title"):
            entry["title"] = fields["title"].strip()
        if label not in entry["referenced_in"]:
            entry["referenced_in"].append(label)


def _extract_from_bibliography_bullets(
    text: str, label: str, sink: dict[str, dict[str, Any]]
) -> None:
    """Pull citations from bullet bibliography lines.

    Format examples (from ``regression-methodology-improvements.md``)::

        - Belloni, A. and Chernozhukov, V. (2013). *Least squares ...*
        - Bailey, D. H. and López de Prado, M. (2014). *The deflated ...*
    """
    for raw_line in text.splitlines():
        m = _BIB_LINE_RE.match(raw_line)
        if not m:
            continue
        year = int(m.group("year"))
        if not _valid_year(year):
            continue
        authors = _split_authors(m.group("authors"))
        if not _is_plausible_author_group(authors):
            continue
        title = m.group("title_star") or m.group("title_under") or m.group("title_quote")
        if title:
            title = title.strip().rstrip(".")
        key = _make_key(authors, year)
        entry = sink.setdefault(
            key,
            {
                "key": key,
                "authors": authors,
                "year": year,
                "title": title,
                "referenced_in": [],
            },
        )
        # Prefer the bibliography-line title (richest signal) once we see one.
        if title and not entry.get("title"):
            entry["title"] = title
        if label not in entry["referenced_in"]:
            entry["referenced_in"].append(label)


def _extract_inline(text: str, label: str, sink: dict[str, dict[str, Any]]) -> None:
    """Pull inline ``Author (Year)`` / ``Author & Author (Year)`` citations."""
    for m in _INLINE_RE.finditer(text):
        try:
            year = int(m.group("year"))
        except ValueError:
            continue
        if not _valid_year(year):
            continue
        authors = _split_authors(m.group("authors"))
        if not _is_plausible_author_group(authors):
            continue
        # If the previous character is a single capital initial like "D." we're
        # likely catching the second half of a bibliography line — let the
        # bibliography extractor handle that.
        start = m.start("authors")
        if start > 0 and text[start - 1] in {"."}:
            # e.g. "...D. and López de Prado, M. (2014)" — already in bib pass
            continue
        key = _make_key(authors, year)
        entry = sink.setdefault(
            key,
            {
                "key": key,
                "authors": authors,
                "year": year,
                "title": None,
                "referenced_in": [],
            },
        )
        if label not in entry["referenced_in"]:
            entry["referenced_in"].append(label)


def _extract_brackets(text: str, label: str, sink: dict[str, dict[str, Any]]) -> None:
    """Pull ``[Author Year]`` / ``[Author, Year]`` style markdown citations."""
    for m in _BRACKET_RE.finditer(text):
        try:
            year = int(m.group("year"))
        except ValueError:
            continue
        if not _valid_year(year):
            continue
        authors = _split_authors(m.group("authors"))
        if not _is_plausible_author_group(authors):
            continue
        key = _make_key(authors, year)
        entry = sink.setdefault(
            key,
            {
                "key": key,
                "authors": authors,
                "year": year,
                "title": None,
                "referenced_in": [],
            },
        )
        if label not in entry["referenced_in"]:
            entry["referenced_in"].append(label)


def _scrape_file(path: Path, root: Path, sink: dict[str, dict[str, Any]]) -> None:
    """Scrape one markdown file into the citation sink."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return
    label = _label_for(path, root)
    # Bibliography bullets are highest signal — extract first so titles win.
    _extract_from_bibliography_bullets(text, label, sink)
    _extract_from_bibtex(text, label, sink)
    _extract_inline(text, label, sink)
    _extract_brackets(text, label, sink)


def _build_index(root: Path) -> dict[str, Any]:
    """Build the citation bibliography from every scraped file."""
    sink: dict[str, dict[str, Any]] = {}
    for path in _collect_source_files(root):
        _scrape_file(path, root, sink)
    citations = sorted(
        sink.values(),
        key=lambda c: (c["authors"][0].lower() if c["authors"] else "z", c["year"]),
    )
    return {
        "checked_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "citations": citations,
        "count": len(citations),
    }


def _get_index() -> dict[str, Any]:
    """Return the cached citations index (TTL = 1 h)."""
    root = _docs_root()
    key = str(root)
    cached = _CACHE.get(key, _CACHE_TTL_SECONDS)
    if cached is not None:
        return cached
    fresh = _build_index(root)
    _CACHE.set(key, fresh)
    return fresh


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/research", tags=["research"])


@router.get(
    "/citations",
    summary="Cross-document bibliography of papers referenced across alpha reports, ADRs and the regression-methodology proposal.",
)
def list_citations() -> dict[str, Any]:
    """Return the merged bibliography across all scraped docs.

    Shape::

        {
          "checked_at": "<UTC ISO8601>",
          "citations": [
            {
              "key": "bailey-lopez-de-prado-2014",
              "authors": ["Bailey", "López de Prado"],
              "year": 2014,
              "title": "The deflated Sharpe ratio",
              "referenced_in": ["alpha-report-v18.md", "ADR-0010", ...]
            },
            ...
          ],
          "count": 47
        }
    """
    return _get_index()
