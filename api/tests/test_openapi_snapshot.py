"""Snapshot tests for ``/openapi.json`` paths.

These tests guard against unintended path additions/removals or silent
summary regressions in the FastAPI app exposed by ``pfm.main``.

How it works
------------
A canonical snapshot is stored at ``tests/fixtures/openapi_snapshot.json``
containing:
- ``paths``: sorted list of every path string in the OpenAPI schema
- ``path_count``: integer length of ``paths``
- ``summaries``: ``{path: summary}`` mapping (first non-empty summary
  encountered across HTTP methods for that path)

On each test run the current app is introspected and compared against the
snapshot. Differences are reported as added/removed path lists.

Update mode
-----------
To regenerate the snapshot (e.g. after a deliberate API change) run::

    PYTEST_UPDATE_SNAPSHOTS=1 pytest tests/test_openapi_snapshot.py

When that env var is set the snapshot file is rewritten and every test is
marked as skipped (because comparison is meaningless against just-written
data). Commit the regenerated fixture and re-run pytest without the env
var to verify.

Strict mode
-----------
``test_no_unexpected_path_added`` is lenient by default — it warns on new
paths but does not fail. Set ``PYTEST_OPENAPI_STRICT=1`` to make it fail
on any addition (useful in CI to gate API surface changes).

Bounds
------
``test_path_count_within_bounds`` enforces a deletion tolerance of 5 and
an addition tolerance of 30 relative to the snapshot count. Outside this
band the test fails — meaning either many paths vanished (likely a bug)
or a wave grew the surface enough to warrant refreshing the snapshot.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import pytest

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "openapi_snapshot.json"
UPDATE_ENV = "PYTEST_UPDATE_SNAPSHOTS"
STRICT_ENV = "PYTEST_OPENAPI_STRICT"

DELETION_TOLERANCE = 5
ADDITION_TOLERANCE = 30


def _is_update_mode() -> bool:
    return os.environ.get(UPDATE_ENV, "").strip() in {"1", "true", "yes"}


def _is_strict_mode() -> bool:
    return os.environ.get(STRICT_ENV, "").strip() in {"1", "true", "yes"}


def _load_snapshot() -> dict:
    if not FIXTURE_PATH.exists():
        pytest.skip(
            f"Snapshot fixture missing at {FIXTURE_PATH}. Run with {UPDATE_ENV}=1 to create it."
        )
    with FIXTURE_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _build_current() -> dict:
    """Introspect the live app and produce a snapshot-shaped dict."""
    # Import inside the helper so module import does not require app load.
    from pfm.main import app  # type: ignore

    openapi = app.openapi()
    paths_data = openapi.get("paths", {}) or {}

    summaries: dict[str, str] = {}
    for path, methods in paths_data.items():
        if not isinstance(methods, dict):
            continue
        for spec in methods.values():
            if not isinstance(spec, dict):
                continue
            summary = spec.get("summary")
            if summary:
                summaries[path] = summary
                break

    return {
        "paths": sorted(paths_data.keys()),
        "path_count": len(paths_data),
        "summaries": summaries,
    }


def _write_snapshot(snapshot: dict) -> None:
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with FIXTURE_PATH.open("w", encoding="utf-8") as fh:
        json.dump(snapshot, fh, indent=2, sort_keys=True)
        fh.write("\n")


@pytest.fixture(scope="module")
def current_snapshot() -> dict:
    return _build_current()


@pytest.fixture(scope="module")
def stored_snapshot() -> dict:
    return _load_snapshot()


def _maybe_update(current: dict) -> None:
    if _is_update_mode():
        _write_snapshot(current)
        pytest.skip(
            f"Snapshot regenerated at {FIXTURE_PATH} "
            f"({current['path_count']} paths). "
            "Unset PYTEST_UPDATE_SNAPSHOTS and re-run to verify."
        )


def test_no_paths_removed(current_snapshot: dict, stored_snapshot: dict) -> None:
    """Every path in the snapshot must still exist in the live app."""
    _maybe_update(current_snapshot)

    stored_paths = set(stored_snapshot.get("paths", []))
    current_paths = set(current_snapshot["paths"])

    removed = sorted(stored_paths - current_paths)
    assert not removed, (
        f"{len(removed)} path(s) removed since snapshot:\n  - "
        + "\n  - ".join(removed)
        + f"\nIf removal is intentional, regenerate the snapshot with "
        f"{UPDATE_ENV}=1 and commit the updated fixture."
    )


def test_no_unexpected_path_added(current_snapshot: dict, stored_snapshot: dict) -> None:
    """New paths warn by default; fail only in strict mode."""
    _maybe_update(current_snapshot)

    stored_paths = set(stored_snapshot.get("paths", []))
    current_paths = set(current_snapshot["paths"])

    added = sorted(current_paths - stored_paths)
    if not added:
        return

    msg = (
        f"{len(added)} new path(s) detected since snapshot:\n  + "
        + "\n  + ".join(added)
        + f"\nIf intentional, regenerate the snapshot with {UPDATE_ENV}=1."
    )
    if _is_strict_mode():
        pytest.fail(msg)
    else:
        warnings.warn(msg, stacklevel=2)


def test_path_count_within_bounds(current_snapshot: dict, stored_snapshot: dict) -> None:
    """Path count must fall within [snapshot - 5, snapshot + 30]."""
    _maybe_update(current_snapshot)

    stored_count = int(stored_snapshot.get("path_count", 0))
    current_count = int(current_snapshot["path_count"])

    lower = stored_count - DELETION_TOLERANCE
    upper = stored_count + ADDITION_TOLERANCE

    assert lower <= current_count <= upper, (
        f"OpenAPI path count {current_count} outside tolerance band "
        f"[{lower}, {upper}] (snapshot baseline = {stored_count}). "
        f"Either a large deletion happened (investigate) or many paths "
        f"were added (regenerate snapshot with {UPDATE_ENV}=1)."
    )


def test_summaries_didnt_regress(current_snapshot: dict, stored_snapshot: dict) -> None:
    """For every path still present, a summary must still exist.

    We only enforce *existence* of a summary, not exact equality —
    summaries are allowed to be edited for clarity, but silently
    removing them counts as a documentation regression.
    """
    _maybe_update(current_snapshot)

    stored_summaries: dict[str, str] = stored_snapshot.get("summaries", {})
    current_summaries: dict[str, str] = current_snapshot["summaries"]
    current_paths = set(current_snapshot["paths"])

    regressed: list[str] = []
    for path in stored_summaries:
        if path not in current_paths:
            # Removal is policed by test_no_paths_removed; skip here.
            continue
        if not current_summaries.get(path):
            regressed.append(path)

    assert not regressed, (
        f"{len(regressed)} path(s) lost their summary since snapshot:\n  - "
        + "\n  - ".join(sorted(regressed))
        + "\nRestore the OpenAPI summary or regenerate the snapshot with "
        f"{UPDATE_ENV}=1 if removal is intentional."
    )
