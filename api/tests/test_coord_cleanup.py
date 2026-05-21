"""Tests for ``api/scripts/coord_cleanup.py`` (W12-57)."""

from __future__ import annotations

import importlib.util
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Load the script as a module (it lives under api/scripts, not the package).
# ---------------------------------------------------------------------------

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "coord_cleanup.py"
_SPEC = importlib.util.spec_from_file_location("coord_cleanup", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
coord_cleanup = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(coord_cleanup)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _iso(dt: datetime) -> str:
    """Format a datetime as the protocol's ISO8601 ``...Z`` form."""
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _entry(
    session_id: str,
    *,
    started_at: datetime,
    expires_at: datetime,
    status: str | None = None,
    files: list[str] | None = None,
) -> dict:
    e = {
        "session_id": session_id,
        "files": files or [f"web/{session_id}.js"],
        "scope": f"scope-{session_id}",
        "started_at": _iso(started_at),
        "expires_at": _iso(expires_at),
        "task_id": f"T-{session_id}",
        "wave": "wave-test",
    }
    if status is not None:
        e["status"] = status
    return e


@pytest.fixture()
def now() -> datetime:
    # Fixed reference time to make deltas explicit.
    return datetime(2026, 5, 16, 12, 0, 0, tzinfo=UTC)


@pytest.fixture()
def ledger(tmp_path: Path) -> Path:
    p = tmp_path / "active-edits.json"
    p.write_text("[]", encoding="utf-8")
    return p


@pytest.fixture()
def archive(tmp_path: Path) -> Path:
    return tmp_path / "active-edits-archive.jsonl"


# ---------------------------------------------------------------------------
# Tests (≥10)
# ---------------------------------------------------------------------------


def test_empty_ledger_returns_zero_counts(ledger: Path, archive: Path) -> None:
    out = coord_cleanup.cleanup_active_edits(ledger, archive_to=archive)
    assert out == {
        "kept": 0,
        "archived": 0,
        "total_before": 0,
        "total_after": 0,
    }
    # Archive file is not created when nothing was archived.
    assert not archive.exists()


def test_active_unexpired_entries_are_kept(
    ledger: Path, archive: Path, now: datetime, monkeypatch: pytest.MonkeyPatch
) -> None:
    entries = [
        _entry(
            "fresh",
            started_at=now - timedelta(minutes=5),
            expires_at=now + timedelta(minutes=25),
        ),
        _entry(
            "soon",
            started_at=now - timedelta(minutes=20),
            expires_at=now + timedelta(minutes=10),
        ),
    ]
    ledger.write_text(json.dumps(entries), encoding="utf-8")

    # Pin ``datetime.now`` to the fixture's reference time — without this
    # the test ages out (entries with ``expires_at = now + 25min`` from
    # 2026-05-16 are real-time-expired by the time the suite runs).
    class _Fixed(datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
            return now

    monkeypatch.setattr(coord_cleanup, "datetime", _Fixed)
    out = coord_cleanup.cleanup_active_edits(ledger, archive_to=archive)
    assert out["archived"] == 0
    assert out["kept"] == 2
    assert json.loads(ledger.read_text()) == entries


def test_completed_recently_kept_within_grace(
    ledger: Path, archive: Path, now: datetime, monkeypatch: pytest.MonkeyPatch
) -> None:
    # COMPLETED entry whose expires_at is only 30 minutes ago — within
    # the 1h grace window, so it should be kept.
    e = _entry(
        "recent-done",
        started_at=now - timedelta(hours=2),
        expires_at=now - timedelta(minutes=30),
        status="COMPLETED",
    )
    ledger.write_text(json.dumps([e]), encoding="utf-8")

    class _Fixed(datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
            return now

    monkeypatch.setattr(coord_cleanup, "datetime", _Fixed)
    out = coord_cleanup.cleanup_active_edits(ledger, archive_to=archive)
    assert out["archived"] == 0
    assert out["kept"] == 1


def test_completed_past_grace_is_archived(
    ledger: Path, archive: Path, now: datetime, monkeypatch: pytest.MonkeyPatch
) -> None:
    e = _entry(
        "old-done",
        started_at=now - timedelta(hours=5),
        expires_at=now - timedelta(hours=2),
        status="COMPLETED",
    )
    ledger.write_text(json.dumps([e]), encoding="utf-8")

    class _Fixed(datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
            return now

    monkeypatch.setattr(coord_cleanup, "datetime", _Fixed)
    out = coord_cleanup.cleanup_active_edits(ledger, archive_to=archive)
    assert out == {
        "kept": 0,
        "archived": 1,
        "total_before": 1,
        "total_after": 0,
    }
    assert json.loads(ledger.read_text()) == []
    lines = archive.read_text().strip().splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["session_id"] == "old-done"


def test_stale_active_claim_archived_after_24h(
    ledger: Path, archive: Path, now: datetime, monkeypatch: pytest.MonkeyPatch
) -> None:
    stale = _entry(
        "stale-active",
        started_at=now - timedelta(hours=30),
        expires_at=now - timedelta(hours=25),
        status="ACTIVE",
    )
    fresh = _entry(
        "fresh-active",
        started_at=now - timedelta(minutes=10),
        expires_at=now + timedelta(minutes=20),
        status="ACTIVE",
    )
    ledger.write_text(json.dumps([stale, fresh]), encoding="utf-8")

    class _Fixed(datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
            return now

    monkeypatch.setattr(coord_cleanup, "datetime", _Fixed)
    out = coord_cleanup.cleanup_active_edits(ledger, archive_to=archive)
    assert out["archived"] == 1
    assert out["kept"] == 1
    assert json.loads(ledger.read_text())[0]["session_id"] == "fresh-active"


def test_custom_keep_hours_threshold(
    ledger: Path, archive: Path, now: datetime, monkeypatch: pytest.MonkeyPatch
) -> None:
    """keep_completed_hours=2 → an active claim 3h past expiry is archived."""
    e = _entry(
        "two-hour-old",
        started_at=now - timedelta(hours=4),
        expires_at=now - timedelta(hours=3),
    )
    ledger.write_text(json.dumps([e]), encoding="utf-8")

    class _Fixed(datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
            return now

    monkeypatch.setattr(coord_cleanup, "datetime", _Fixed)
    out = coord_cleanup.cleanup_active_edits(ledger, archive_to=archive, keep_completed_hours=2)
    assert out["archived"] == 1


def test_dry_run_does_not_modify_files(
    ledger: Path, archive: Path, now: datetime, monkeypatch: pytest.MonkeyPatch
) -> None:
    e = _entry(
        "old-done",
        started_at=now - timedelta(hours=5),
        expires_at=now - timedelta(hours=2),
        status="COMPLETED",
    )
    original = json.dumps([e])
    ledger.write_text(original, encoding="utf-8")

    class _Fixed(datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
            return now

    monkeypatch.setattr(coord_cleanup, "datetime", _Fixed)
    out = coord_cleanup.cleanup_active_edits(ledger, archive_to=archive, dry_run=True)
    assert out["archived"] == 1
    assert out["total_after"] == 0
    # Files are untouched.
    assert ledger.read_text() == original
    assert not archive.exists()


def test_archive_jsonl_appends_one_per_line(
    ledger: Path, archive: Path, now: datetime, monkeypatch: pytest.MonkeyPatch
) -> None:
    a = _entry(
        "a",
        started_at=now - timedelta(hours=10),
        expires_at=now - timedelta(hours=5),
        status="COMPLETED",
    )
    b = _entry(
        "b",
        started_at=now - timedelta(hours=30),
        expires_at=now - timedelta(hours=26),
        status="ACTIVE",
    )
    ledger.write_text(json.dumps([a, b]), encoding="utf-8")

    class _Fixed(datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
            return now

    monkeypatch.setattr(coord_cleanup, "datetime", _Fixed)

    # First call archives both.
    coord_cleanup.cleanup_active_edits(ledger, archive_to=archive)
    lines = archive.read_text().splitlines()
    assert len(lines) == 2
    sessions = {json.loads(line)["session_id"] for line in lines}
    assert sessions == {"a", "b"}

    # Second call (now empty) is a no-op and doesn't blow away the archive.
    coord_cleanup.cleanup_active_edits(ledger, archive_to=archive)
    assert archive.read_text().splitlines() == lines


def test_unparseable_expires_at_is_preserved(ledger: Path, archive: Path) -> None:
    """Malformed timestamps must be kept so a human can inspect."""
    bad = {
        "session_id": "bad-ts",
        "files": ["x.py"],
        "scope": "scope",
        "started_at": "yesterday",
        "expires_at": "not-a-timestamp",
        "status": "COMPLETED",
        "task_id": "T-bad",
        "wave": "wave-test",
    }
    ledger.write_text(json.dumps([bad]), encoding="utf-8")
    out = coord_cleanup.cleanup_active_edits(ledger, archive_to=archive)
    assert out["archived"] == 0
    assert out["kept"] == 1


def test_non_dict_entries_are_preserved(ledger: Path, archive: Path) -> None:
    """Stray non-dict rows must round-trip rather than crash the sweeper."""
    ledger.write_text(json.dumps([{"x": 1}, "stray-string", 42]), "utf-8")
    out = coord_cleanup.cleanup_active_edits(ledger, archive_to=archive)
    assert out["kept"] == 3
    assert out["archived"] == 0
    data = json.loads(ledger.read_text())
    assert data == [{"x": 1}, "stray-string", 42]


def test_non_array_root_raises(ledger: Path, archive: Path) -> None:
    ledger.write_text(json.dumps({"not": "a list"}), "utf-8")
    with pytest.raises(ValueError, match="Expected a JSON array"):
        coord_cleanup.cleanup_active_edits(ledger, archive_to=archive)


def test_atomic_write_no_partial_file_on_crash(
    ledger: Path, archive: Path, now: datetime, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If write fails mid-flight, the live ledger must remain intact."""
    keep = _entry(
        "fresh",
        started_at=now - timedelta(minutes=5),
        expires_at=now + timedelta(minutes=25),
    )
    drop = _entry(
        "old",
        started_at=now - timedelta(hours=30),
        expires_at=now - timedelta(hours=26),
        status="COMPLETED",
    )
    payload = json.dumps([keep, drop], indent=2)
    ledger.write_text(payload, encoding="utf-8")

    class _Fixed(datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
            return now

    monkeypatch.setattr(coord_cleanup, "datetime", _Fixed)

    # Force os.replace to blow up — atomic write must leave original intact.
    import os as _os

    real_replace = _os.replace

    def _boom(src, dst):  # type: ignore[no-untyped-def]
        raise OSError("disk on fire")

    monkeypatch.setattr(coord_cleanup.os, "replace", _boom)

    with pytest.raises(OSError, match="disk on fire"):
        coord_cleanup.cleanup_active_edits(ledger, archive_to=archive)

    # Original ledger is still readable and unchanged.
    assert ledger.read_text() == payload
    # No leftover ".tmp" siblings remain in the directory.
    leftovers = [
        p
        for p in ledger.parent.iterdir()
        if p.name.startswith(ledger.name + ".") and p.name.endswith(".tmp")
    ]
    assert leftovers == [], f"leftover tempfiles: {leftovers}"

    # Restore so other tests aren't affected (monkeypatch undoes this anyway).
    monkeypatch.setattr(coord_cleanup.os, "replace", real_replace)


def test_archive_to_none_skips_archive_file(
    ledger: Path, now: datetime, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    e = _entry(
        "old-done",
        started_at=now - timedelta(hours=5),
        expires_at=now - timedelta(hours=2),
        status="COMPLETED",
    )
    ledger.write_text(json.dumps([e]), encoding="utf-8")

    class _Fixed(datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
            return now

    monkeypatch.setattr(coord_cleanup, "datetime", _Fixed)
    out = coord_cleanup.cleanup_active_edits(ledger, archive_to=None)
    assert out["archived"] == 1
    # Confirm no stray archive file was created in the tmp dir.
    assert list(tmp_path.glob("*.jsonl")) == []


def test_status_completed_is_case_insensitive(
    ledger: Path, archive: Path, now: datetime, monkeypatch: pytest.MonkeyPatch
) -> None:
    e = _entry(
        "old-done-lower",
        started_at=now - timedelta(hours=5),
        expires_at=now - timedelta(hours=2),
        status="completed",
    )
    ledger.write_text(json.dumps([e]), encoding="utf-8")

    class _Fixed(datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
            return now

    monkeypatch.setattr(coord_cleanup, "datetime", _Fixed)
    out = coord_cleanup.cleanup_active_edits(ledger, archive_to=archive)
    assert out["archived"] == 1


def test_cli_dry_run(
    ledger: Path,
    archive: Path,
    now: datetime,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    e = _entry(
        "old-done",
        started_at=now - timedelta(hours=5),
        expires_at=now - timedelta(hours=2),
        status="COMPLETED",
    )
    original = json.dumps([e])
    ledger.write_text(original, encoding="utf-8")

    class _Fixed(datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[override]
            return now

    monkeypatch.setattr(coord_cleanup, "datetime", _Fixed)

    rc = coord_cleanup.main(
        [
            "--path",
            str(ledger),
            "--archive",
            str(archive),
            "--dry-run",
        ]
    )
    assert rc == 0
    captured = capsys.readouterr().out
    assert "DRY-RUN" in captured
    assert "archived=1" in captured
    # Dry-run leaves ledger unchanged.
    assert ledger.read_text() == original
