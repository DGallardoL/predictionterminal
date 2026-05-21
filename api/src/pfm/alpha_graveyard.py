"""Alpha Graveyard — public registry of dead / downgraded alpha strategies.

Intellectual-honesty layer: every strategy that we *claimed* and later killed
should be visible alongside the live alpha cards.  This module loads the
``web/data/alpha_graveyard.json`` registry, validates its schema with Pydantic
v2, and offers a few lightweight filters for the API router.

The graveyard schema is intentionally tiny.  Each entry records:

* ``pair_id`` — stable identifier (also the URL path component).
* ``name`` — short human label.
* ``killed_iso`` / ``killed_in_wave`` — when and in which research wave the
  strategy died.
* ``cause`` — one of a closed vocabulary (``regime``, ``TC``,
  ``single-episode``, ``grid-search``, ``tautology``, ``capacity``,
  ``non-portable``).
* ``claimed_sharpe`` / ``post_mortem_sharpe`` — what we *thought* the strategy
  was worth vs. what it was actually worth out-of-sample.
* ``thesis_original`` — one paragraph describing the original idea.
* ``lesson`` — one paragraph describing why it died and what we learned.
* ``could_resurrect_if`` — one line spelling out what would change our minds.
* ``tags`` — free-form tag list for downstream filtering / UI.
* ``death_certificate_md`` — relative path to the long-form post-mortem under
  ``docs/graveyard/``.

Public functions
----------------
* :func:`load_graveyard` — read the JSON file once and return a list of dicts.
* :func:`filter_by_cause` — keep entries whose ``cause`` matches the argument.
* :func:`get_graveyard_path` — resolve the on-disk path of the registry.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Vocabularies
# ---------------------------------------------------------------------------

#: Closed set of failure modes.  Each implies a different remediation path
#: (regime → wait for new sample, TC → re-route execution, etc.).
GraveyardCause = Literal[
    "regime",
    "TC",
    "single-episode",
    "grid-search",
    "tautology",
    "capacity",
    "non-portable",
]

#: Special filter token meaning "do not filter".  Kept distinct from
#: ``GraveyardCause`` so the type system catches accidental misuse.
GraveyardCauseFilter = Literal[
    "all",
    "regime",
    "TC",
    "single-episode",
    "grid-search",
    "tautology",
    "capacity",
    "non-portable",
]


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class GraveyardEntry(BaseModel):
    """Schema for a single graveyard row.

    The model is permissive on extra fields so future waves can extend the
    registry without breaking deployed clients (Pydantic v2 default).
    """

    pair_id: str = Field(..., min_length=1)
    name: str = Field(..., min_length=1)
    killed_iso: str = Field(..., min_length=10, max_length=10)
    killed_in_wave: int = Field(..., ge=1)
    cause: GraveyardCause
    claimed_sharpe: float
    post_mortem_sharpe: float
    thesis_original: str = Field(..., min_length=20)
    lesson: str = Field(..., min_length=20)
    could_resurrect_if: str = Field(..., min_length=10)
    tags: list[str] = Field(default_factory=list)
    death_certificate_md: str | None = None


class GraveyardResponse(BaseModel):
    """Wrapper returned by ``GET /alpha-hub/graveyard``."""

    n_entries: int = Field(..., ge=0)
    cause_filter: GraveyardCauseFilter = "all"
    entries: list[GraveyardEntry]


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def get_graveyard_path() -> Path:
    """Resolve the absolute path of ``web/data/alpha_graveyard.json``.

    The repository layout is ``<root>/api/src/pfm/alpha_graveyard.py`` and
    ``<root>/web/data/alpha_graveyard.json``.  We walk up four parents from
    this file's location to reach the repo root and then descend into
    ``web/data``.
    """
    return Path(__file__).resolve().parents[3] / "web" / "data" / "alpha_graveyard.json"


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def load_graveyard(path: Path | None = None) -> list[dict[str, Any]]:
    """Read and return the graveyard registry as a list of plain dicts.

    Args:
        path: Optional override; defaults to :func:`get_graveyard_path`.

    Returns:
        The deserialized JSON list.  No Pydantic validation is applied here
        so callers that just want the raw data (e.g. JSON-equivalent tests)
        can avoid the model overhead.

    Raises:
        FileNotFoundError: If the JSON file is missing.
        ValueError: If the top-level JSON is not a list.
    """
    p = path if path is not None else get_graveyard_path()
    if not p.exists():
        raise FileNotFoundError(f"alpha_graveyard.json not found at {p}")
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(
            f"alpha_graveyard.json must be a JSON array at the top level, got {type(raw).__name__}"
        )
    return raw


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------


def filter_by_cause(
    entries: list[dict[str, Any]],
    cause: GraveyardCauseFilter,
) -> list[dict[str, Any]]:
    """Filter graveyard entries by ``cause`` field.

    ``cause='all'`` is a no-op (returns the input list as-is).  Any other
    value keeps only entries whose ``cause`` matches exactly.
    """
    if cause == "all":
        return list(entries)
    return [e for e in entries if e.get("cause") == cause]
