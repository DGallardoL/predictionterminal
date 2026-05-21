"""FastAPI router for the Research sub-tab of the alpha Hub.

Endpoints
---------
* ``GET /research/reports`` — list every ``docs/alpha-reports/alpha-report-v*.md``
  file as a JSON card (version, title, published_at, summary, deployable_count,
  anti_alpha_count, path).
* ``GET /research/reports/{version}`` — return the markdown body of one report;
  when ``?format=html`` is given AND the optional ``markdown`` library is
  installed, the body is server-rendered to HTML.

The router is consumed by the Track-G Research sub-tab inside α Hub (T57).

Integration note
----------------
This router is **not** auto-mounted from ``pfm.main``. To enable, claim the
``main.py:routes`` section (see ``.coordination/PROTOCOL-V2.md``) and add::

    from pfm.research.router import router as _research_router
    app.include_router(_research_router)

next to the other ``app.include_router(...)`` calls at the bottom of
``api/src/pfm/main.py``.
"""

from __future__ import annotations

import os
import re
import threading
import time
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse, Response

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Cache TTL for the parsed report index (seconds). 5 minutes per spec.
_CACHE_TTL_SECONDS = 300

# Repo-root anchored docs location. Resolved lazily so tests can override via
# the ``PFM_RESEARCH_DOCS_DIR`` environment variable without import-time
# side effects.
_DEFAULT_DOCS_SUBPATH = ("docs", "alpha-reports")

# How many characters of the body (after the H1) get sliced into the summary.
_SUMMARY_CHARS = 200

# Filename pattern: alpha-report-v<N>.md. ``N`` must be an integer.
_FILENAME_RE = re.compile(r"^alpha-report-v(\d+)\.md$")

# Frontmatter delimiter for YAML-style metadata (rare in this repo).
_FRONTMATTER_RE = re.compile(
    r"\A---\s*\n(?P<body>.*?)\n---\s*\n",
    re.DOTALL,
)

# Bold "Generated" line used by virtually every report:
#   **Generated**: 2026-05-02 overnight autopilot.
_GENERATED_RE = re.compile(
    r"\*\*Generated\*\*\s*:\s*(?P<date>\d{4}-\d{2}-\d{2})",
)

# Headings we treat as "deployable" or "anti-alpha" sections. The check is
# intentionally heuristic — substring match against the heading text, lowercased.
_DEPLOYABLE_KEYWORDS = (
    "deployable",
    "production book",
    "production three",
    "production-book",
    "validated alpha",
)
_ANTI_KEYWORDS = (
    "anti-alpha",
    "anti alpha",
    "graveyard",
    "archived",
    "do not redeploy",
    "dead",
    "demoted",
)


# ---------------------------------------------------------------------------
# TTL cache (thread-safe)
# ---------------------------------------------------------------------------


class _TTLCache:
    """A trivial single-slot TTL cache for the parsed report index.

    The cache key is the resolved docs directory path (so tests that point at
    a temp directory get their own slot). Eviction is implicit via the TTL.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[str, tuple[float, list[dict[str, Any]]]] = {}

    def get(self, key: str, ttl: float) -> list[dict[str, Any]] | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            ts, value = entry
            if time.monotonic() - ts > ttl:
                # Expired — drop and miss.
                self._entries.pop(key, None)
                return None
            return value

    def set(self, key: str, value: list[dict[str, Any]]) -> None:
        with self._lock:
            self._entries[key] = (time.monotonic(), value)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


_CACHE = _TTLCache()


def _clear_cache() -> None:
    """Public hook for tests — drops every cached index entry."""
    _CACHE.clear()


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def _docs_dir() -> Path:
    """Return the directory holding ``alpha-report-v*.md`` files.

    Resolution order:

    1. ``PFM_RESEARCH_DOCS_DIR`` env var (used by tests).
    2. ``<repo-root>/docs/alpha-reports``.

    The repo root is computed relative to this file: ``api/src/pfm/research``
    is four levels deep, so ``parents[4]`` is the repo root.
    """
    override = os.environ.get("PFM_RESEARCH_DOCS_DIR")
    if override:
        return Path(override)
    here = Path(__file__).resolve()
    # parents: [0]=research, [1]=pfm, [2]=src, [3]=api, [4]=repo root
    return here.parents[4].joinpath(*_DEFAULT_DOCS_SUBPATH)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _parse_frontmatter(text: str) -> dict[str, str]:
    """Return a flat ``key: value`` map from optional YAML frontmatter.

    Only handles the trivial ``key: value`` form (one per line). Lists,
    dicts, and quoted strings are left as plain text. Reports in this repo
    currently have no frontmatter; this is forward-compat plumbing.
    """
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}
    out: dict[str, str] = {}
    for line in m.group("body").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        out[key.strip().lower()] = value.strip().strip('"').strip("'")
    return out


def _body_after_frontmatter(text: str) -> str:
    """Return ``text`` with any leading YAML frontmatter block stripped."""
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return text
    return text[m.end() :]


def _extract_h1(body: str) -> str | None:
    """Return the first ``# Heading`` line, without the leading ``#``."""
    for line in body.splitlines():
        stripped = line.strip()
        if stripped.startswith("# ") and not stripped.startswith("## "):
            return stripped[2:].strip()
    return None


def _extract_summary(body: str) -> str:
    """Return the first ~200 chars of prose after the H1.

    Skips blank lines, sub-headings, horizontal rules, and bold-only meta
    lines (e.g. ``**Generated**: 2026-05-02``). Joins remaining prose lines
    on single spaces, then slices to ``_SUMMARY_CHARS``.
    """
    lines = body.splitlines()
    # Find index of first H1
    start = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("# ") and not line.strip().startswith("## "):
            start = i + 1
            break
    prose: list[str] = []
    for line in lines[start:]:
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            # Hit the next section — stop.
            break
        if s.startswith("---"):
            continue
        if s.startswith("**") and s.endswith("**") and ":" not in s:
            continue
        # Skip pure-meta bold prefix lines like "**Generated**: ...".
        if s.startswith("**Generated**") or s.startswith("**Predecessors**"):
            continue
        prose.append(s)
        joined = " ".join(prose)
        if len(joined) >= _SUMMARY_CHARS:
            break
    summary = " ".join(prose).strip()
    if len(summary) > _SUMMARY_CHARS:
        summary = summary[:_SUMMARY_CHARS].rstrip() + "..."
    return summary


def _count_sections(body: str, keywords: tuple[str, ...]) -> int:
    """Return the count of markdown headings (any level) matching keywords.

    The match is a case-insensitive substring check against the heading text.
    Duplicate matches on the same line are counted once. Sub-headings count
    independently of their parent, which is the intended heuristic.
    """
    count = 0
    for line in body.splitlines():
        s = line.strip()
        if not s.startswith("#"):
            continue
        # Strip leading '#' chars + whitespace.
        text = s.lstrip("#").strip().lower()
        if not text:
            continue
        for kw in keywords:
            if kw in text:
                count += 1
                break
    return count


def _parse_report(path: Path) -> dict[str, Any] | None:
    """Parse a single ``alpha-report-v*.md`` file into a card dict.

    Returns ``None`` when the filename does not match the expected pattern
    (so e.g. ``alpha-report.md`` without a version is silently skipped).
    """
    m = _FILENAME_RE.match(path.name)
    if not m:
        return None
    version = int(m.group(1))
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None

    fm = _parse_frontmatter(text)
    body = _body_after_frontmatter(text)
    h1 = _extract_h1(body) or fm.get("title") or f"Alpha Report v{version}"

    published_at = fm.get("published_at") or fm.get("date") or ""
    if not published_at:
        gen = _GENERATED_RE.search(body)
        if gen:
            published_at = gen.group("date")

    summary = _extract_summary(body)
    deployable_count = _count_sections(body, _DEPLOYABLE_KEYWORDS)
    anti_alpha_count = _count_sections(body, _ANTI_KEYWORDS)

    # The ``path`` field is a repo-relative POSIX string suitable for showing
    # in the UI / linking from elsewhere.
    rel_path = f"docs/alpha-reports/{path.name}"

    return {
        "id": f"v{version}",
        "version": version,
        "title": fm.get("title") or h1,
        "published_at": published_at,
        "summary": summary,
        "deployable_count": deployable_count,
        "anti_alpha_count": anti_alpha_count,
        "path": rel_path,
    }


def _load_index(docs_dir: Path) -> list[dict[str, Any]]:
    """Glob the docs dir, parse every report, sort by version desc."""
    if not docs_dir.is_dir():
        return []
    cards: list[dict[str, Any]] = []
    for entry in sorted(docs_dir.iterdir()):
        if not entry.is_file():
            continue
        card = _parse_report(entry)
        if card is not None:
            cards.append(card)
    cards.sort(key=lambda c: c["version"], reverse=True)
    return cards


def _get_index() -> list[dict[str, Any]]:
    """Return the cached report index (TTL = 5 min) for the resolved docs dir."""
    docs_dir = _docs_dir()
    key = str(docs_dir)
    cached = _CACHE.get(key, _CACHE_TTL_SECONDS)
    if cached is not None:
        return cached
    fresh = _load_index(docs_dir)
    _CACHE.set(key, fresh)
    return fresh


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/research", tags=["research"])


@router.get(
    "/reports",
    summary="List all alpha-report versions as JSON cards",
)
def list_reports() -> dict[str, list[dict[str, Any]]]:
    """Return every parsed ``alpha-report-v*.md`` sorted by version desc.

    The response envelope intentionally mirrors the spec in T31::

        {"reports": [{"id": "v18", "version": 18, ...}, ...]}
    """
    return {"reports": _get_index()}


def _resolve_version(version: str) -> int:
    """Coerce ``"v18"`` / ``"18"`` / ``"V18"`` to the integer ``18``.

    Raises ``HTTPException(400)`` when the input cannot be parsed.
    """
    s = version.strip().lower()
    if s.startswith("v"):
        s = s[1:]
    if not s.isdigit():
        raise HTTPException(status_code=400, detail=f"invalid version: {version!r}")
    return int(s)


@router.get(
    "/reports/{version}",
    summary="Return the markdown body (or HTML) of a single alpha report",
)
def get_report(
    version: str,
    format: Annotated[
        str,
        Query(description="Response format: 'markdown' (default) or 'html'."),
    ] = "markdown",
) -> Response:
    """Return the raw markdown body of one report, or rendered HTML.

    HTML rendering is best-effort — when the optional ``markdown`` library
    is not installed, the request gracefully falls back to ``text/markdown``
    so the front-end can render client-side.
    """
    n = _resolve_version(version)
    docs_dir = _docs_dir()
    target = docs_dir / f"alpha-report-v{n}.md"
    if not target.is_file():
        raise HTTPException(status_code=404, detail=f"report v{n} not found")

    try:
        text = target.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - filesystem failure
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    fmt = format.strip().lower()
    if fmt == "html":
        try:
            import markdown as _md  # type: ignore[import-not-found]
        except ImportError:
            # Graceful degradation: still return markdown so the UI can render
            # it client-side with marked.js / showdown.
            return PlainTextResponse(content=text, media_type="text/markdown")
        body = _body_after_frontmatter(text)
        html = _md.markdown(body, extensions=["tables", "fenced_code"])
        return Response(content=html, media_type="text/html")

    return PlainTextResponse(content=text, media_type="text/markdown")
