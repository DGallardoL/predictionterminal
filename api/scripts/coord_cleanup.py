#!/usr/bin/env python3
"""Sweep expired/completed entries from ``.coordination/active-edits.json``.

Operational maintenance utility for the multi-session coordination ledger
described in ``.coordination/PROTOCOL-V2.md``. The ledger is APPEND-ONLY
during normal use, so it grows unbounded; this script archives stale rows
to keep the live file fast to parse.

Archival policy (an entry is archived when ANY rule matches):
    * ``status == "COMPLETED"`` (case-insensitive) AND ``expires_at`` is older
      than 1 hour ago.
    * ``expires_at`` is older than ``keep_completed_hours`` hours ago (stale
      active claims that were never released).

Archived rows are appended (one JSON object per line) to
``.coordination/active-edits-archive.jsonl``. The cleaned array is written
back atomically (write to a sibling tempfile in the same directory, then
``os.replace`` over the original).

CLI::

    python scripts/coord_cleanup.py [--dry-run] [--keep-hours 24] \\
        [--path PATH] [--archive PATH]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

COMPLETED_GRACE_HOURS = 1
"""How long after expiry a COMPLETED claim is kept before archiving."""

DEFAULT_KEEP_HOURS = 24
"""Stale active claims older than this many hours past expiry are archived."""


def _parse_iso(value: Any) -> datetime | None:
    """Parse an ISO8601 timestamp into a timezone-aware UTC ``datetime``.

    Returns ``None`` for values that cannot be parsed; callers treat
    unparseable timestamps as "do not archive" (safer to keep than drop).
    """
    if not isinstance(value, str) or not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _should_archive(
    entry: dict[str, Any],
    *,
    now: datetime,
    keep_completed_hours: int,
) -> bool:
    """Return True if ``entry`` should be moved to the archive."""
    expires = _parse_iso(entry.get("expires_at"))
    if expires is None:
        # Unparseable timestamp: keep it in the live file so a human can fix.
        return False

    status = entry.get("status")
    is_completed = isinstance(status, str) and status.upper() == "COMPLETED"

    if is_completed and expires < now - timedelta(hours=COMPLETED_GRACE_HOURS):
        return True

    if expires < now - timedelta(hours=keep_completed_hours):
        return True

    return False


def _atomic_write_json(path: Path, payload: list[dict[str, Any]]) -> None:
    """Write ``payload`` to ``path`` atomically via tempfile + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        os.replace(tmp_name, path)
    except Exception:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def cleanup_active_edits(
    path: str | Path = ".coordination/active-edits.json",
    *,
    archive_to: str | Path | None = ".coordination/active-edits-archive.jsonl",
    keep_completed_hours: int = DEFAULT_KEEP_HOURS,
    dry_run: bool = False,
) -> dict:
    """Sweep stale/completed claims from the coordination ledger.

    Args:
        path: Path to ``active-edits.json``.
        archive_to: Path to the JSONL archive. ``None`` skips archival
            (entries are dropped from the live file but not persisted).
        keep_completed_hours: Stale active claims older than this many hours
            past ``expires_at`` are archived.
        dry_run: If True, do not write either file; only report counts.

    Returns:
        Dict with keys ``kept``, ``archived``, ``total_before``,
        ``total_after``.
    """
    src = Path(path)
    raw = src.read_text(encoding="utf-8")
    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array at {src}, got {type(data).__name__}")

    now = datetime.now(UTC)
    kept: list[dict[str, Any]] = []
    archived: list[dict[str, Any]] = []

    for entry in data:
        if not isinstance(entry, dict):
            # Malformed row: keep it so a human can inspect.
            kept.append(entry)
            continue
        if _should_archive(
            entry,
            now=now,
            keep_completed_hours=keep_completed_hours,
        ):
            archived.append(entry)
        else:
            kept.append(entry)

    summary = {
        "kept": len(kept),
        "archived": len(archived),
        "total_before": len(data),
        "total_after": len(kept),
    }

    if dry_run or not archived:
        # Nothing to write when dry-run, or when no rows were archived.
        return summary

    if archive_to is not None:
        archive_path = Path(archive_to)
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        with archive_path.open("a", encoding="utf-8") as handle:
            for entry in archived:
                handle.write(json.dumps(entry, sort_keys=True))
                handle.write("\n")

    _atomic_write_json(src, kept)
    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Sweep stale/completed entries from .coordination/active-edits.json"),
    )
    parser.add_argument(
        "--path",
        default=".coordination/active-edits.json",
        help="Path to active-edits.json (default: %(default)s)",
    )
    parser.add_argument(
        "--archive",
        default=".coordination/active-edits-archive.jsonl",
        help="JSONL archive file (default: %(default)s)",
    )
    parser.add_argument(
        "--keep-hours",
        type=int,
        default=DEFAULT_KEEP_HOURS,
        help=(
            "Stale active claims older than this many hours past expires_at "
            "are archived (default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would happen without writing any files.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    summary = cleanup_active_edits(
        path=args.path,
        archive_to=args.archive,
        keep_completed_hours=args.keep_hours,
        dry_run=args.dry_run,
    )
    mode = "DRY-RUN" if args.dry_run else "APPLIED"
    print(
        f"[coord_cleanup] {mode} "
        f"before={summary['total_before']} "
        f"after={summary['total_after']} "
        f"archived={summary['archived']} "
        f"kept={summary['kept']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
