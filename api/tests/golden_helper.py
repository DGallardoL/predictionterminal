"""Golden-file regression helper.

Goal
----
Pin the *exact JSON shape* of a small set of important endpoints so any
unintended schema change is caught by CI rather than discovered by a
front-end consumer at demo time.  This is the same idea as ``pytest-snapshot``
or ``jest --updateSnapshot``, kept dependency-free.

How it works
------------
* On the first run with no golden file present, ``assert_matches_golden``
  writes ``tests/golden/<name>.json`` and *fails* the test (so the new
  golden is committed deliberately, not silently accepted).
* On every subsequent run it diffs the actual response against the saved
  file after stripping any keys the test marked as volatile (timestamps,
  uptime counters, git SHAs, absolute paths).
* The comparison is deep-equal on the cleaned dicts.  Mismatches print the
  first divergent path so you know what changed.

Regenerating a golden after an *intentional* schema change
----------------------------------------------------------
Run ``scripts/regenerate_golden.sh`` (or simply delete the offending file
under ``tests/golden/`` and re-run pytest twice — once to write, once to
verify).  Always inspect the resulting ``git diff`` before committing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

GOLDEN_DIR: Path = Path(__file__).parent / "golden"


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------


def assert_matches_golden(
    name: str,
    actual: Any,
    *,
    ignore_keys: list[str] | None = None,
) -> None:
    """Compare ``actual`` against ``tests/golden/<name>.json``.

    Args:
        name: Stem of the golden file (no ``.json`` extension).  Subdirs
            are allowed (``"terminal/quote_dummy"`` is fine).
        actual: Anything JSON-serialisable — typically the parsed body of
            a ``TestClient`` response.
        ignore_keys: Top-level *or nested* dict keys to drop before
            comparison.  Use this for timestamps, uptime counters,
            git SHAs, file paths, and other values that legitimately vary
            between runs.
    """
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    path = GOLDEN_DIR / f"{name}.json"
    cleaned_actual = _strip_keys(actual, set(ignore_keys or []))

    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(cleaned_actual, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        pytest.fail(
            f"golden file '{name}.json' did not exist — wrote it from this "
            "run's output. Re-run pytest to verify, then commit the new file."
        )

    expected = json.loads(path.read_text(encoding="utf-8"))
    cleaned_expected = _strip_keys(expected, set(ignore_keys or []))

    if not _tolerant_equal(cleaned_actual, cleaned_expected):
        diff_path = _first_diff(cleaned_expected, cleaned_actual)
        raise AssertionError(
            f"golden mismatch for '{name}'.\n"
            f"first divergence at: {diff_path}\n"
            f"to regenerate: delete tests/golden/{name}.json and re-run."
        )


# ---------------------------------------------------------------------------
# internals
# ---------------------------------------------------------------------------


def _strip_keys(obj: Any, keys: set[str]) -> Any:
    """Recursively remove any dict entries whose key is in ``keys``."""
    if isinstance(obj, dict):
        return {k: _strip_keys(v, keys) for k, v in obj.items() if k not in keys}
    if isinstance(obj, list):
        return [_strip_keys(x, keys) for x in obj]
    return obj


def _tolerant_equal(a: Any, b: Any, *, rtol: float = 1e-3, atol: float = 1e-9) -> bool:
    """Deep-equal, but compare floats within a tolerance.

    Numeric leaves (e.g. optimizer weights, ratios) can differ in the last
    digits between platforms (Mac vs the Linux CI runner) due to BLAS/float
    rounding — that's not a schema change, so we treat near-equal floats as
    equal. Everything non-numeric is still compared exactly, so a real shape
    or value change is still caught.
    """
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= atol + rtol * abs(float(b))
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return False
        return all(_tolerant_equal(a[k], b[k], rtol=rtol, atol=atol) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return False
        return all(_tolerant_equal(x, y, rtol=rtol, atol=atol) for x, y in zip(a, b, strict=True))
    return a == b


def _first_diff(expected: Any, actual: Any, path: str = "$") -> str:
    """Return a JSONPath-ish description of the first place two structures
    diverge.  Used to make assertion failures useful in CI logs."""
    if type(expected) is not type(actual):
        return f"{path} (type {type(expected).__name__} vs {type(actual).__name__})"
    if isinstance(expected, dict):
        for k in sorted(set(expected) | set(actual)):
            if k not in expected:
                return f"{path}.{k} (extra in actual)"
            if k not in actual:
                return f"{path}.{k} (missing in actual)"
            sub = _first_diff(expected[k], actual[k], f"{path}.{k}")
            if sub:
                return sub
        return ""
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return f"{path} (len {len(expected)} vs {len(actual)})"
        for i, (e, a) in enumerate(zip(expected, actual, strict=True)):
            sub = _first_diff(e, a, f"{path}[{i}]")
            if sub:
                return sub
        return ""
    return path if expected != actual else ""
