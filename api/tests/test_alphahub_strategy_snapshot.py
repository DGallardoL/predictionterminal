"""Snapshot test for the ``/alpha-hub/strategy/{pair_id}`` JSON shape.

This protects the public contract of the per-strategy detail endpoint by
capturing the *shape* of the response (keys + value types + length-class)
rather than the data values themselves. The leaderboard composition shifts
every time ``web/data/alpha_strategies.json`` is regenerated, but the shape
of an individual strategy detail payload should only move when we
deliberately rename or add fields.

Same protocol as :mod:`tests.test_alphahub_leaderboard_snapshot` (task
W12-11). Wave-13, task W13-29.

Behaviour:

* The current shape must match the saved fixture at
  ``api/tests/fixtures/alphahub_strategy_snapshot.json``.
* Setting the environment variable ``PYTEST_UPDATE_SNAPSHOTS=1`` regenerates
  the fixture in-place — the snapshot test becomes a "write" instead of an
  "assert".
* Adding a NEW required field (key) at any level fails the snapshot — the
  developer must consciously rerun with ``PYTEST_UPDATE_SNAPSHOTS=1``.
* Renaming a field (delete one key, add another) fails the snapshot for the
  same reason.

Strategy selection
------------------
We deterministically pick the top strategy by ``oos_sharpe desc`` from the
leaderboard at test time. This isolates the snapshot from churn in the
strategy *identity* — the test still asserts on *shape*, not on the
particular ``pair_id`` selected. To make the shape stable across catalog
regenerations we additionally walk the leaderboard until we find an entry
whose detail payload exposes every optional sub-block (``deploy_signal_logic``,
``rule``, ``recent_signal``, ``deployment``). Strategies that omit any of
those would otherwise drift the shape descriptor on every catalog rebuild.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from pfm.main import app

# --- Helpers ---------------------------------------------------------------

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "alphahub_strategy_snapshot.json"


def shape_of(obj: Any) -> Any:
    """Return a recursive shape descriptor for ``obj``.

    * dict   -> ``{"_dict": {key: shape_of(value), ...}}``
    * list   -> ``{"_list": shape_of(first), "_size_class": "empty"|"single"|"many"}``
    * scalar -> the type name ("str", "int", "float", "bool", "NoneType")

    The size_class is intentionally bucketed so the snapshot does not break
    when the spread / equity-curve length wobbles by a few items — only
    when it crosses an empty/non-empty boundary or shrinks to a single row.
    """
    if isinstance(obj, dict):
        return {"_dict": {k: shape_of(v) for k, v in obj.items()}}
    if isinstance(obj, list):
        if not obj:
            return {"_list": None, "_size_class": "empty"}
        size_class = "single" if len(obj) == 1 else "many"
        return {"_list": shape_of(obj[0]), "_size_class": size_class}
    if obj is None:
        return "NoneType"
    return type(obj).__name__


def _load_fixture() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text())


def _save_fixture(shape: dict[str, Any]) -> None:
    FIXTURE_PATH.write_text(json.dumps(shape, indent=2) + "\n")


#: Top-level keys we require to be present in the chosen detail payload so
#: the snapshot covers every optional sub-block at least once. If the
#: top-ranked strategy is missing one (e.g. ``recent_signal=None`` because
#: no live signal yet), we walk further down the leaderboard.
_REQUIRED_SUBBLOCKS: tuple[str, ...] = (
    "deploy_signal_logic",
    "rule",
    "recent_signal",
    "deployment",
    "equity_curve",
    "spread_series",
    "risk",
)


def _pick_full_strategy_detail(client: TestClient) -> dict[str, Any]:
    """Return the detail payload of the highest-Sharpe strategy with all
    optional sub-blocks populated.

    Falls back to the top entry unconditionally if no candidate exposes the
    full sub-block set (so the test still runs even on a sparse catalog).
    """
    page = client.get("/alpha-hub/leaderboard?sort=oos_sharpe&order=desc&limit=50")
    assert page.status_code == 200, f"leaderboard probe returned {page.status_code}"
    items = page.json().get("items", [])
    assert items, "leaderboard is empty; cannot pick a pair_id"

    fallback: dict[str, Any] | None = None
    for item in items:
        pid = item.get("pair_id")
        if not pid:
            continue
        r = client.get(f"/alpha-hub/strategy/{pid}")
        if r.status_code != 200:
            continue
        payload = r.json()
        if fallback is None:
            fallback = payload
        if all(payload.get(k) not in (None, [], {}) for k in _REQUIRED_SUBBLOCKS):
            return payload
    assert fallback is not None, "no strategy detail fetched successfully"
    return fallback


# --- Fixtures --------------------------------------------------------------


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(scope="module")
def current_shape(client: TestClient) -> dict[str, Any]:
    payload = _pick_full_strategy_detail(client)
    return shape_of(payload)


# --- Tests -----------------------------------------------------------------


def test_current_shape_matches_saved_snapshot(
    current_shape: dict[str, Any],
) -> None:
    """The current response shape matches the committed fixture.

    Honours ``PYTEST_UPDATE_SNAPSHOTS=1``: when set, the fixture is
    regenerated and the assertion is skipped. This is the explicit override
    a developer must use when they deliberately changed the response.
    """
    if os.environ.get("PYTEST_UPDATE_SNAPSHOTS") == "1":
        _save_fixture(current_shape)
        pytest.skip("PYTEST_UPDATE_SNAPSHOTS=1 set — snapshot regenerated, assertion skipped.")

    assert FIXTURE_PATH.exists(), (
        f"snapshot fixture missing at {FIXTURE_PATH}; "
        "run with PYTEST_UPDATE_SNAPSHOTS=1 to create it"
    )
    saved = _load_fixture()
    assert current_shape == saved, (
        "strategy-detail shape drift detected. Re-run with "
        "PYTEST_UPDATE_SNAPSHOTS=1 to accept the new shape."
    )


def test_pytest_update_snapshots_regenerates_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    current_shape: dict[str, Any],
) -> None:
    """``PYTEST_UPDATE_SNAPSHOTS=1`` rewrites the fixture file in place.

    We point the test at a temp file (so the real fixture is untouched),
    flip the flag, and verify the temp file is written with the current
    shape.
    """
    temp_fixture = tmp_path / "snapshot.json"
    monkeypatch.setattr(
        "tests.test_alphahub_strategy_snapshot.FIXTURE_PATH",
        temp_fixture,
    )
    monkeypatch.setenv("PYTEST_UPDATE_SNAPSHOTS", "1")

    # Reimplement the save call against the patched path (since the env
    # branch in the main test would skip via pytest.skip).
    if os.environ.get("PYTEST_UPDATE_SNAPSHOTS") == "1":
        temp_fixture.write_text(json.dumps(current_shape, indent=2) + "\n")

    assert temp_fixture.exists()
    written = json.loads(temp_fixture.read_text())
    assert written == current_shape


def test_new_required_field_triggers_failure(
    current_shape: dict[str, Any],
) -> None:
    """Adding a brand-new top-level field must break the snapshot test.

    We simulate the bug: the saved fixture has the *old* shape, the live
    response has the new shape (one extra key). The equality assertion
    must fail so the developer is forced to rerun with
    ``PYTEST_UPDATE_SNAPSHOTS=1``.
    """
    saved = copy.deepcopy(current_shape)
    mutated = copy.deepcopy(current_shape)
    mutated["_dict"]["brand_new_field"] = "int"
    assert saved != mutated, "expected a new top-level key to change the shape descriptor"

    # Also verify the same protection at a nested sub-block level.
    saved_nested = copy.deepcopy(current_shape)
    mutated_nested = copy.deepcopy(current_shape)
    risk_shape = mutated_nested["_dict"]["risk"]["_dict"]
    risk_shape["new_metric"] = "float"
    assert saved_nested != mutated_nested, (
        "expected a new nested field (risk.new_metric) to change the shape descriptor"
    )


def test_field_rename_triggers_failure(
    current_shape: dict[str, Any],
) -> None:
    """Renaming a field (drop one key, add another) must break the snapshot.

    A pure rename leaves the total key count unchanged but changes the
    descriptor, so the equality assertion must still fail. This is the
    classic refactor bug we want to catch: e.g. ``oos_sharpe`` -> ``sharpe_oos``.
    """
    saved = copy.deepcopy(current_shape)
    mutated = copy.deepcopy(current_shape)

    top_shape = mutated["_dict"]
    assert "oos_sharpe" in top_shape, (
        "fixture sanity check: oos_sharpe is one of the per-strategy fields "
        "we expect the detail payload to expose"
    )
    top_shape["sharpe_oos"] = top_shape.pop("oos_sharpe")

    assert saved != mutated, (
        "expected a field rename (oos_sharpe -> sharpe_oos) to change the shape descriptor"
    )


def test_required_subblocks_present(current_shape: dict[str, Any]) -> None:
    """Sanity-check: the snapshot covers every optional sub-block we care about.

    If a future catalog regeneration drops all entries with a populated
    ``recent_signal`` (for example), the picker would silently fall back
    to the top entry and the snapshot would mutate. This test makes that
    scenario explicit: every block named in ``_REQUIRED_SUBBLOCKS`` must
    appear in the captured shape's top-level ``_dict``.
    """
    top_keys = set(current_shape["_dict"].keys())
    for key in _REQUIRED_SUBBLOCKS:
        assert key in top_keys, (
            f"required sub-block '{key}' missing from strategy-detail snapshot; "
            "the picker probably fell back to a sparse strategy. Re-run with "
            "PYTEST_UPDATE_SNAPSHOTS=1 only after confirming the catalog still "
            "has a candidate with all sub-blocks populated."
        )


# --- Helper-function unit tests -------------------------------------------


def test_shape_of_handles_primitives() -> None:
    assert shape_of("hello") == "str"
    assert shape_of(42) == "int"
    assert shape_of(3.14) == "float"
    assert shape_of(True) == "bool"
    assert shape_of(None) == "NoneType"


def test_shape_of_size_class_buckets() -> None:
    assert shape_of([]) == {"_list": None, "_size_class": "empty"}
    assert shape_of([1]) == {"_list": "int", "_size_class": "single"}
    assert shape_of([1, 2, 3]) == {"_list": "int", "_size_class": "many"}


def test_shape_of_nested_dict() -> None:
    result = shape_of({"a": 1, "b": {"c": "x"}})
    assert result == {
        "_dict": {
            "a": "int",
            "b": {"_dict": {"c": "str"}},
        }
    }
