"""Tests for ``scripts/detect_dead_slugs.py`` (W12-09).

All tests inject a mocked ``fetch_history`` callable — none hit the network.
The fixture-style ``_yml_with_factors`` helper writes a fresh YAML per test so
runs are hermetic.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml

# --- Import the script as a module (it lives outside the ``pfm`` package) ---
_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "detect_dead_slugs.py"
_spec = importlib.util.spec_from_file_location("_detect_dead_slugs", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
detect_mod = importlib.util.module_from_spec(_spec)
sys.modules["_detect_dead_slugs"] = detect_mod
_spec.loader.exec_module(detect_mod)

detect_dead_slugs = detect_mod.detect_dead_slugs
apply_prune = detect_mod.apply_prune
write_report = detect_mod.write_report
REASON_NO_DATA = detect_mod.REASON_NO_DATA
REASON_FETCH_ERROR = detect_mod.REASON_FETCH_ERROR
REASON_INSUFFICIENT_OBS = detect_mod.REASON_INSUFFICIENT_OBS


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _yml_with_factors(tmp_path: Path, factors: list[dict], *, name: str = "factors.yml") -> Path:
    p = tmp_path / name
    p.write_text(yaml.safe_dump({"factors": factors}, sort_keys=False))
    return p


def _factor(
    fid: str,
    *,
    slug: str | None = None,
    source: str = "polymarket",
    theme: str = "macro",
) -> dict:
    return {
        "id": fid,
        "name": fid.replace("_", " ").title(),
        "slug": slug or f"slug-{fid}",
        "source": source,
        "theme": theme,
        "description": f"desc for {fid}",
    }


# ---------------------------------------------------------------------------
# 1. All-healthy → no dead records
# ---------------------------------------------------------------------------


def test_all_healthy_returns_empty(tmp_path: Path) -> None:
    yml = _yml_with_factors(tmp_path, [_factor("a"), _factor("b"), _factor("c")])

    def fetch(_factor: dict, _cutoff: datetime) -> list[int]:
        return [1] * 40  # comfortably above min_obs=30

    dead = detect_dead_slugs(yml, fetch_history=fetch)
    assert dead == []


# ---------------------------------------------------------------------------
# 2. No-data → flagged with no_data_returned reason
# ---------------------------------------------------------------------------


def test_empty_fetch_flags_as_no_data(tmp_path: Path) -> None:
    yml = _yml_with_factors(tmp_path, [_factor("ghost", slug="ghost-slug")])

    def fetch(_factor: dict, _cutoff: datetime) -> list[int]:
        return []

    dead = detect_dead_slugs(yml, fetch_history=fetch, min_obs=30, since_days=90)
    assert len(dead) == 1
    rec = dead[0]
    assert rec["slug"] == "ghost-slug"
    assert rec["obs_count"] == 0
    assert rec["reason"] == REASON_NO_DATA
    assert rec["id"] == "ghost"


# ---------------------------------------------------------------------------
# 3. Insufficient observations → distinct reason tag
# ---------------------------------------------------------------------------


def test_insufficient_obs_reason(tmp_path: Path) -> None:
    yml = _yml_with_factors(tmp_path, [_factor("thin", slug="thin-slug")])

    def fetch(_factor: dict, _cutoff: datetime) -> list[int]:
        return [1] * 12  # below default min_obs=30

    dead = detect_dead_slugs(yml, fetch_history=fetch)
    assert len(dead) == 1
    rec = dead[0]
    assert rec["obs_count"] == 12
    assert rec["reason"] == REASON_INSUFFICIENT_OBS


# ---------------------------------------------------------------------------
# 4. Boundary: exactly min_obs is healthy; min_obs - 1 is dead
# ---------------------------------------------------------------------------


def test_boundary_at_min_obs_is_healthy(tmp_path: Path) -> None:
    yml = _yml_with_factors(tmp_path, [_factor("edge")])

    def fetch_exact(_factor: dict, _cutoff: datetime) -> list[int]:
        return [1] * 30

    def fetch_below(_factor: dict, _cutoff: datetime) -> list[int]:
        return [1] * 29

    assert detect_dead_slugs(yml, fetch_history=fetch_exact, min_obs=30) == []
    below = detect_dead_slugs(yml, fetch_history=fetch_below, min_obs=30)
    assert len(below) == 1
    assert below[0]["obs_count"] == 29


# ---------------------------------------------------------------------------
# 5. Fetch errors are captured under a stable reason tag
# ---------------------------------------------------------------------------


def test_fetch_exception_marks_dead_with_fetch_error(tmp_path: Path) -> None:
    yml = _yml_with_factors(tmp_path, [_factor("explodes", slug="boom")])

    def fetch(_factor: dict, _cutoff: datetime) -> list[int]:
        raise RuntimeError("upstream HTTP 500: gateway timeout")

    dead = detect_dead_slugs(yml, fetch_history=fetch)
    assert len(dead) == 1
    rec = dead[0]
    assert rec["reason"] == REASON_FETCH_ERROR
    assert "upstream HTTP 500" in rec["error"]
    assert rec["obs_count"] == 0


# ---------------------------------------------------------------------------
# 6. Source filter narrows the scan
# ---------------------------------------------------------------------------


def test_source_filter_restricts_scope(tmp_path: Path) -> None:
    yml = _yml_with_factors(
        tmp_path,
        [
            _factor("pm1", source="polymarket"),
            _factor("k1", source="kalshi"),
            _factor("f1", source="fred"),
        ],
    )
    seen: list[str] = []

    def fetch(factor: dict, _cutoff: datetime) -> list[int]:
        seen.append(factor["source"])
        return []

    dead = detect_dead_slugs(yml, fetch_history=fetch, sources=frozenset({"polymarket"}))
    assert set(seen) == {"polymarket"}
    assert len(dead) == 1
    assert dead[0]["source"] == "polymarket"


# ---------------------------------------------------------------------------
# 7. Cutoff passed to fetcher matches now - since_days
# ---------------------------------------------------------------------------


def test_cutoff_respects_since_days(tmp_path: Path) -> None:
    yml = _yml_with_factors(tmp_path, [_factor("a")])
    fixed_now = datetime(2026, 5, 16, 12, 0, tzinfo=UTC)
    captured: list[datetime] = []

    def fetch(_factor: dict, cutoff: datetime) -> list[int]:
        captured.append(cutoff)
        return [1] * 100

    detect_dead_slugs(yml, fetch_history=fetch, since_days=45, now=fixed_now)
    assert len(captured) == 1
    assert captured[0] == fixed_now - timedelta(days=45)


# ---------------------------------------------------------------------------
# 8. Output JSON shape (via write_report)
# ---------------------------------------------------------------------------


def test_write_report_emits_expected_schema(tmp_path: Path) -> None:
    out = tmp_path / "dead.json"
    dead = [
        {
            "id": "x",
            "slug": "x-slug",
            "theme": "macro",
            "source": "polymarket",
            "obs_count": 4,
            "reason": REASON_INSUFFICIENT_OBS,
        }
    ]
    write_report(dead, output=out, min_obs=30, since_days=90)
    payload = json.loads(out.read_text())
    assert set(payload.keys()) >= {
        "checked_at",
        "min_obs",
        "since_days",
        "dead_count",
        "dead_slugs",
    }
    assert payload["min_obs"] == 30
    assert payload["since_days"] == 90
    assert payload["dead_count"] == 1
    assert payload["dead_slugs"][0]["slug"] == "x-slug"


# ---------------------------------------------------------------------------
# 9. apply_prune rewrites factors.yml and creates a backup
# ---------------------------------------------------------------------------


def test_apply_prune_removes_dead_ids_and_writes_backup(tmp_path: Path) -> None:
    yml = _yml_with_factors(
        tmp_path,
        [_factor("keep_me"), _factor("drop_me"), _factor("keep_too")],
    )
    dead = [
        {
            "id": "drop_me",
            "slug": "slug-drop_me",
            "theme": "macro",
            "source": "polymarket",
            "obs_count": 0,
            "reason": REASON_NO_DATA,
        }
    ]
    backup = apply_prune(yml, dead, backup_suffix=".bak.test")
    assert backup.exists(), "backup must exist"
    # Backup retains the original three entries
    pre = yaml.safe_load(backup.read_text())
    assert sorted(f["id"] for f in pre["factors"]) == ["drop_me", "keep_me", "keep_too"]
    # File now contains only the two kept entries
    post = yaml.safe_load(yml.read_text())
    ids = sorted(f["id"] for f in post["factors"])
    assert ids == ["keep_me", "keep_too"]


# ---------------------------------------------------------------------------
# 10. apply_prune with empty list is a no-op for content, still backs up
# ---------------------------------------------------------------------------


def test_apply_prune_noop_with_empty_dead(tmp_path: Path) -> None:
    yml = _yml_with_factors(tmp_path, [_factor("a"), _factor("b")])
    before = yaml.safe_load(yml.read_text())
    backup = apply_prune(yml, [], backup_suffix=".bak.noop")
    assert backup.exists()
    after = yaml.safe_load(yml.read_text())
    assert sorted(f["id"] for f in after["factors"]) == sorted(f["id"] for f in before["factors"])


# ---------------------------------------------------------------------------
# 11. Missing factors.yml raises FileNotFoundError
# ---------------------------------------------------------------------------


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        detect_dead_slugs(tmp_path / "nope.yml", fetch_history=lambda f, c: [])


# ---------------------------------------------------------------------------
# 12. Validation: min_obs and since_days bounds
# ---------------------------------------------------------------------------


def test_invalid_thresholds_rejected(tmp_path: Path) -> None:
    yml = _yml_with_factors(tmp_path, [_factor("a")])
    with pytest.raises(ValueError):
        detect_dead_slugs(yml, min_obs=-1, fetch_history=lambda f, c: [])
    with pytest.raises(ValueError):
        detect_dead_slugs(yml, since_days=0, fetch_history=lambda f, c: [])


# ---------------------------------------------------------------------------
# 13. Mixed catalog: some healthy, some dead — only dead are returned
# ---------------------------------------------------------------------------


def test_mixed_catalog_partitions_correctly(tmp_path: Path) -> None:
    yml = _yml_with_factors(
        tmp_path,
        [
            _factor("healthy_1"),
            _factor("dead_a"),
            _factor("healthy_2"),
            _factor("dead_b"),
            _factor("dead_c"),
        ],
    )
    obs_table = {
        "healthy_1": 50,
        "dead_a": 5,
        "healthy_2": 31,
        "dead_b": 0,
        "dead_c": 29,
    }

    def fetch(factor: dict, _cutoff: datetime) -> list[int]:
        return [1] * obs_table[factor["id"]]

    dead = detect_dead_slugs(yml, fetch_history=fetch, min_obs=30)
    dead_ids = {r["id"] for r in dead}
    assert dead_ids == {"dead_a", "dead_b", "dead_c"}
    # Only dead_b should carry the no_data reason
    reasons = {r["id"]: r["reason"] for r in dead}
    assert reasons["dead_b"] == REASON_NO_DATA
    assert reasons["dead_a"] == REASON_INSUFFICIENT_OBS
    assert reasons["dead_c"] == REASON_INSUFFICIENT_OBS


# ---------------------------------------------------------------------------
# 14. CLI main() runs end-to-end in dry-run, writes the report file
# ---------------------------------------------------------------------------


def test_cli_main_dry_run_writes_report(tmp_path: Path, monkeypatch) -> None:
    yml = _yml_with_factors(
        tmp_path,
        [_factor("good"), _factor("bad", slug="bad-slug")],
    )
    out = tmp_path / "out.json"

    # Force the CLI fetcher to consider 'bad-slug' dead by stubbing the helper.
    def fake_fetcher_factory():
        def _fetch(factor, cutoff):
            del cutoff
            return [] if factor["slug"] == "bad-slug" else [1] * 40

        return _fetch

    monkeypatch.setattr(detect_mod, "_make_cli_fetcher", fake_fetcher_factory)

    rc = detect_mod.main(
        [
            "--factors-yml",
            str(yml),
            "--output",
            str(out),
            "--min-obs",
            "30",
            "--since-days",
            "60",
        ]
    )
    assert rc == 0
    payload = json.loads(out.read_text())
    assert payload["dead_count"] == 1
    assert payload["dead_slugs"][0]["slug"] == "bad-slug"
    # No --apply flag → factors.yml untouched
    assert {f["id"] for f in yaml.safe_load(yml.read_text())["factors"]} == {"good", "bad"}


# ---------------------------------------------------------------------------
# 15. CLI main() with --apply prunes the YAML and leaves a backup
# ---------------------------------------------------------------------------


def test_cli_main_apply_prunes_yaml(tmp_path: Path, monkeypatch) -> None:
    yml = _yml_with_factors(
        tmp_path,
        [_factor("keep"), _factor("drop", slug="drop-slug")],
    )
    out = tmp_path / "out.json"

    def fake_fetcher_factory():
        def _fetch(factor, cutoff):
            del cutoff
            return [] if factor["slug"] == "drop-slug" else [1] * 40

        return _fetch

    monkeypatch.setattr(detect_mod, "_make_cli_fetcher", fake_fetcher_factory)

    rc = detect_mod.main(
        [
            "--factors-yml",
            str(yml),
            "--output",
            str(out),
            "--apply",
        ]
    )
    assert rc == 0
    surviving = {f["id"] for f in yaml.safe_load(yml.read_text())["factors"]}
    assert surviving == {"keep"}
    # At least one backup file should exist next to factors.yml
    backups = list(yml.parent.glob("factors.yml.bak.dead_slugs.*"))
    assert backups, "expected timestamped backup"


# ---------------------------------------------------------------------------
# 16. _count_observations handles list / dict / DataFrame-like inputs
# ---------------------------------------------------------------------------


def test_count_observations_handles_varied_shapes() -> None:
    count = detect_mod._count_observations
    assert count([1, 2, 3]) == 3
    assert count({"a": 1, "b": 2}) == 2
    assert count(None) == 0
    assert count(0) == 0  # ints have no len → 0

    class _FakeFrame:
        shape = (17, 4)

    assert count(_FakeFrame()) == 17


# ---------------------------------------------------------------------------
# 17. Progress callback is invoked once per factor
# ---------------------------------------------------------------------------


def test_progress_callback_called_per_factor(tmp_path: Path) -> None:
    yml = _yml_with_factors(tmp_path, [_factor("a"), _factor("b"), _factor("c")])
    calls: list[tuple[int, int]] = []

    def progress(i: int, n: int, _factor: dict) -> None:
        calls.append((i, n))

    detect_dead_slugs(
        yml,
        fetch_history=lambda f, c: [1] * 100,
        progress=progress,
    )
    assert calls == [(1, 3), (2, 3), (3, 3)]
