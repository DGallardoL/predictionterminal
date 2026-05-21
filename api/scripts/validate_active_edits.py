#!/usr/bin/env python3
"""Validate `.coordination/active-edits.json` structure.

Used as a pre-commit hook so that no commit can land a malformed
coordination ledger. Exit 1 on any error (prints a diagnostic to stderr).

Required schema per entry (see `.coordination/PROTOCOL-V2.md`):
- session_id: str
- files: list[str]
- scope: str
- started_at: ISO8601 UTC str
- expires_at: ISO8601 UTC str
- task_id: str  (recommended; warned if missing, not fatal)
- wave: str     (recommended; warned if missing, not fatal)
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

REQUIRED_FIELDS = ("session_id", "files", "scope", "started_at", "expires_at")
RECOMMENDED_FIELDS = ("task_id", "wave")


def _find_repo_root(start: Path) -> Path:
    """Walk upward from `start` to find a directory containing `.coordination`."""
    cur = start.resolve()
    for _ in range(8):
        if (cur / ".coordination" / "active-edits.json").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    # Fallback: assume two levels up from api/scripts/
    return Path(__file__).resolve().parents[2]


def _parse_iso(value: str) -> datetime | None:
    """Parse an ISO8601 timestamp; return None if unparseable."""
    if not isinstance(value, str):
        return None
    text = value.rstrip("Z")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def main() -> int:
    repo_root = _find_repo_root(Path.cwd())
    ledger = repo_root / ".coordination" / "active-edits.json"

    if not ledger.exists():
        print(f"ERROR: missing {ledger}", file=sys.stderr)
        return 1

    try:
        raw = ledger.read_text(encoding="utf-8")
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"ERROR: {ledger} is not valid JSON: {exc}", file=sys.stderr)
        return 1

    if not isinstance(data, list):
        print(
            f"ERROR: {ledger} must be a JSON array (got {type(data).__name__})",
            file=sys.stderr,
        )
        return 1

    errors: list[str] = []
    warnings: list[str] = []

    for idx, entry in enumerate(data):
        prefix = f"entry[{idx}]"
        if not isinstance(entry, dict):
            errors.append(f"{prefix}: must be an object (got {type(entry).__name__})")
            continue

        for field in REQUIRED_FIELDS:
            if field not in entry:
                errors.append(f"{prefix}: missing required field '{field}'")

        files = entry.get("files")
        if files is not None and not (
            isinstance(files, list) and all(isinstance(f, str) for f in files)
        ):
            errors.append(f"{prefix}: 'files' must be a list[str]")

        for ts_field in ("started_at", "expires_at"):
            ts = entry.get(ts_field)
            if ts is None:
                continue
            if _parse_iso(ts) is None:
                errors.append(f"{prefix}: '{ts_field}' is not valid ISO8601 ({ts!r})")

        # NOTE: PROTOCOL-V2 explicitly allows `expires_at < started_at` as a
        # "released" marker. We only fail on missing/unparseable timestamps,
        # not on inverted ordering.

        for rec_field in RECOMMENDED_FIELDS:
            if rec_field not in entry:
                warnings.append(f"{prefix}: missing recommended field '{rec_field}'")

    for w in warnings:
        print(f"WARN: {w}", file=sys.stderr)

    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        print(
            f"FAIL: {len(errors)} error(s) in {ledger.relative_to(repo_root)}",
            file=sys.stderr,
        )
        return 1

    print(f"OK: {len(data)} active-edits entries valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
