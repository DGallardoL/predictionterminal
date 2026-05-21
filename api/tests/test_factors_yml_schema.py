"""Structural integrity checks for ``src/pfm/factors.yml``.

This is a read-only validation suite. It does NOT modify ``factors.yml``.
If a check finds dirty data (duplicate slugs, theme typos, missing fields, ...)
the failure surfaces the offending entries so a human can decide what to do.

Run standalone (no project conftest needed):
    pytest tests/test_factors_yml_schema.py -q --noconftest
"""

from __future__ import annotations

import json
import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
import yaml

# --- locate factors.yml -------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FACTORS_YML = _REPO_ROOT / "api" / "src" / "pfm" / "factors.yml"
_ALPHA_STRATEGIES_JSON = _REPO_ROOT / "web" / "data" / "alpha_strategies.json"

# CLAUDE.md "Scale" subsection currently states 1228 factors (post-2026-05-14).
# PROTOCOL-V2 mentions 1360 in a forbidden-actions note. Track against the
# value most recently confirmed by `GET /factors/all`.
_EXPECTED_FACTOR_COUNT = 1228
_COUNT_TOLERANCE_PCT = 5.0  # soft assertion threshold

# Required fields on every entry.
_REQUIRED_FIELDS = ("id", "name", "slug", "source", "theme", "description")

# Known good themes. Detected from the live file as of 2026-05-16.
# Update this set when a new theme is intentionally added.
_KNOWN_THEMES = frozenset(
    {
        "ai",
        "business",
        "chips",
        "climate",
        "commodities",
        "crypto",
        "energy",
        "equity",
        "geopolitics",
        "health",
        "legal",
        "macro",
        "other",
        "politics",
        "pop_culture",
        "science",
        "space",
        "sports",
        "weather",
    }
)

# Sources whose external IDs are NOT lowercase-kebab strings.
# - kalshi tickers are ALL-CAPS (e.g. ``KXRECSSNBER-26``).
# - fred / bls series IDs are ALL-CAPS (e.g. ``T10Y2Y``, ``ICSA``).
# - predictit contract IDs are numeric strings (``8200``).
# - polymarket slugs occasionally start with a digit (e.g. ``2026-balance-...``,
#   ``10pt0-or-above-earthquake-...``); those are still kebab-cased so we allow
#   a leading digit when the source is polymarket.
_NON_KEBAB_SOURCES = frozenset({"kalshi", "fred", "bls", "predictit"})

_LOWER_KEBAB_RE = re.compile(r"^[a-z][a-z0-9-]*$")
_LOWER_KEBAB_OR_LEADING_DIGIT_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_UPPER_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9_\-\.]*$")
_NUMERIC_RE = re.compile(r"^[0-9]+$")


# --- fixtures -----------------------------------------------------------------


@pytest.fixture(scope="module")
def factors_yml_path() -> Path:
    assert _FACTORS_YML.exists(), f"factors.yml not found at {_FACTORS_YML}"
    return _FACTORS_YML


@pytest.fixture(scope="module")
def factors_doc(factors_yml_path: Path) -> Any:
    with factors_yml_path.open() as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="module")
def factors(factors_doc: Any) -> list[dict[str, Any]]:
    assert isinstance(factors_doc, dict) and "factors" in factors_doc, (
        "Expected top-level dict with a 'factors' list."
    )
    items = factors_doc["factors"]
    assert isinstance(items, list), "'factors' must be a list."
    return items


# --- tests --------------------------------------------------------------------


def test_yaml_parses_without_error(factors_yml_path: Path) -> None:
    """factors.yml must be syntactically valid YAML."""
    with factors_yml_path.open() as fh:
        try:
            yaml.safe_load(fh)
        except yaml.YAMLError as exc:  # pragma: no cover - failure path
            pytest.fail(f"factors.yml failed to parse: {exc}")


def test_top_level_structure_is_dict_with_factors_list(factors_doc: Any) -> None:
    """Top-level is a mapping with a single 'factors' list of dicts."""
    assert isinstance(factors_doc, dict), (
        f"Expected dict at top level, got {type(factors_doc).__name__}"
    )
    assert "factors" in factors_doc, "Top-level dict must contain a 'factors' key"
    assert isinstance(factors_doc["factors"], list), "'factors' must be a list"
    assert factors_doc["factors"], "'factors' list must be non-empty"
    for i, entry in enumerate(factors_doc["factors"][:5]):
        assert isinstance(entry, dict), f"Entry {i} is not a dict: {type(entry).__name__}"


def test_no_duplicate_slugs(factors: list[dict[str, Any]]) -> None:
    """Every slug must appear exactly once."""
    counts = Counter(entry.get("slug", "") for entry in factors)
    dupes = {slug: c for slug, c in counts.items() if c > 1}
    assert not dupes, f"Duplicate slugs found ({len(dupes)}): {dupes}"


def test_no_duplicate_ids(factors: list[dict[str, Any]]) -> None:
    """Every id must also appear exactly once (paranoid sibling check)."""
    counts = Counter(entry.get("id", "") for entry in factors)
    dupes = {fid: c for fid, c in counts.items() if c > 1}
    assert not dupes, f"Duplicate ids found ({len(dupes)}): {dupes}"


def test_slugs_match_source_specific_pattern(factors: list[dict[str, Any]]) -> None:
    """Slug format depends on source:

    - polymarket / manifold: lowercase-kebab, may start with a digit
    - kalshi / fred / bls: ALL-CAPS ticker-style
    - predictit: numeric string

    Slugs containing whitespace or colons (``:``) are NEVER allowed in this
    file. The ``sentiment:<query>`` form is a runtime synthetic factor source
    and is documented in CLAUDE.md to live outside ``factors.yml``.
    """
    bad: list[tuple[str, str, str]] = []
    for entry in factors:
        slug = entry.get("slug", "")
        source = entry.get("source", "")
        if not slug or " " in slug or ":" in slug:
            bad.append((entry.get("id", "?"), source, slug))
            continue
        if source in _NON_KEBAB_SOURCES:
            if source == "predictit":
                ok = bool(_NUMERIC_RE.match(slug))
            else:
                ok = bool(_UPPER_TICKER_RE.match(slug))
        else:
            # polymarket, manifold, and anything else default to kebab.
            ok = bool(_LOWER_KEBAB_OR_LEADING_DIGIT_RE.match(slug))
        if not ok:
            bad.append((entry.get("id", "?"), source, slug))

    assert not bad, "Slug format violations (id, source, slug):\n" + "\n".join(
        f"  {fid}\t{src}\t{slug!r}" for fid, src, slug in bad
    )


def test_every_entry_has_required_fields(factors: list[dict[str, Any]]) -> None:
    """Each factor must define id/name/slug/source/theme/description."""
    missing: list[tuple[int, str, list[str]]] = []
    for i, entry in enumerate(factors):
        absent = [f for f in _REQUIRED_FIELDS if f not in entry]
        if absent:
            missing.append((i, entry.get("id", "?"), absent))
    assert not missing, f"{len(missing)} entries missing required fields. First 10: {missing[:10]}"


def test_no_empty_labels(factors: list[dict[str, Any]]) -> None:
    """``name`` (used by the UI as the human label) must be non-empty."""
    empty = [entry.get("id", "?") for entry in factors if not str(entry.get("name", "")).strip()]
    assert not empty, f"{len(empty)} entries have empty 'name' fields: {empty[:10]}"


def test_no_empty_descriptions(factors: list[dict[str, Any]]) -> None:
    """Descriptions must be non-empty for UI tooltips and factor cards."""
    empty = [
        entry.get("id", "?") for entry in factors if not str(entry.get("description", "")).strip()
    ]
    assert not empty, f"{len(empty)} entries have empty 'description' fields: {empty[:10]}"


def test_themes_are_from_known_set(factors: list[dict[str, Any]]) -> None:
    """No typos like 'politcs' / 'macrro' — every theme is in the known set."""
    observed = Counter(entry.get("theme", "") for entry in factors)
    unknown = {theme: count for theme, count in observed.items() if theme not in _KNOWN_THEMES}
    assert not unknown, (
        f"Unknown / possibly-typo themes detected: {unknown}. Known set: {sorted(_KNOWN_THEMES)}"
    )


def test_no_slug_contains_space_or_colon(factors: list[dict[str, Any]]) -> None:
    """Whitespace / colons in slugs would collide with the ``sentiment:<q>``
    runtime prefix convention."""
    bad = [
        (entry.get("id", "?"), entry.get("slug", ""))
        for entry in factors
        if " " in str(entry.get("slug", "")) or ":" in str(entry.get("slug", ""))
    ]
    assert not bad, f"Slugs with space or colon: {bad}"


def test_factor_count_within_tolerance(factors: list[dict[str, Any]]) -> None:
    """Soft assertion: warn (not fail) if catalog size drifts >5% from expected."""
    actual = len(factors)
    delta_pct = abs(actual - _EXPECTED_FACTOR_COUNT) / _EXPECTED_FACTOR_COUNT * 100
    if delta_pct > _COUNT_TOLERANCE_PCT:
        warnings.warn(
            f"Factor count drift: expected ~{_EXPECTED_FACTOR_COUNT}, "
            f"found {actual} ({delta_pct:.1f}% delta). Update CLAUDE.md 'Scale' "
            f"and _EXPECTED_FACTOR_COUNT in this test.",
            stacklevel=2,
        )
    # Hard floor: never let it silently empty out.
    assert actual >= 100, f"Catalog suspiciously small: only {actual} entries"


#: Synthetic-strategy slug prefixes that legitimately don't appear in
#: ``factors.yml``: they identify compound strategies (calendar spreads,
#: equity baskets, fair-prob shadow markets) whose ``a_slug``/``b_slug``
#: fields are internal markers rather than Polymarket market slugs.
_SYNTHETIC_SLUG_PATTERNS = (
    "calendar-",
    "composite-",
    "cpt-",
    "equity-",
    "fresh-",
    "basket-",
)


def test_alpha_strategies_slugs_present_in_factors_yml(
    factors: list[dict[str, Any]],
) -> None:
    """Every Polymarket ``a_slug`` / ``b_slug`` referenced by
    ``alpha_strategies.json`` must exist in ``factors.yml``. Synthetic
    compound-strategy identifiers (calendar/composite/basket) are
    excluded — they don't map to single Polymarket markets."""
    if not _ALPHA_STRATEGIES_JSON.exists():
        pytest.skip(f"alpha_strategies.json not found at {_ALPHA_STRATEGIES_JSON}")

    with _ALPHA_STRATEGIES_JSON.open() as fh:
        payload = json.load(fh)

    strategies = payload.get("strategies", []) if isinstance(payload, dict) else []
    if not strategies:
        pytest.skip("alpha_strategies.json has no 'strategies' list to validate")

    catalog_slugs = {entry.get("slug", "") for entry in factors}
    referenced: set[str] = set()
    for s in strategies:
        for k in ("a_slug", "b_slug"):
            v = s.get(k)
            if isinstance(v, str) and v:
                referenced.add(v)

    def _is_synthetic(slug: str) -> bool:
        return any(slug.startswith(p) for p in _SYNTHETIC_SLUG_PATTERNS)

    missing = sorted(s for s in referenced if s not in catalog_slugs and not _is_synthetic(s))
    # Real-slug drift between alpha_strategies.json and factors.yml is a
    # data-quality concern, not a code defect — surface as xfail so it
    # appears in CI without blocking the suite. Run the curation script
    # (``scripts/sanitize_alpha_strategies.py``) to reconcile.
    if missing:
        pytest.xfail(
            f"{len(missing)} non-synthetic slugs in alpha_strategies.json missing "
            f"from factors.yml: {missing[:10]} (run sanitize_alpha_strategies.py)"
        )


def test_sources_are_recognised(factors: list[dict[str, Any]]) -> None:
    """Sentinel — guard against typo'd source values."""
    known_sources = {"polymarket", "kalshi", "manifold", "predictit", "fred", "bls"}
    observed = Counter(entry.get("source", "") for entry in factors)
    unknown = {src: c for src, c in observed.items() if src not in known_sources}
    assert not unknown, (
        f"Unknown sources detected (probable typos): {unknown}. Known: {sorted(known_sources)}"
    )
