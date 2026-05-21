"""Tests for ``scripts/nightly_dead_slug_sweep.py`` (W12-53).

All tests inject a mocked detector + fetcher — none hit the network or the
real ``factors.yml``. The sweep writes to tmp directories so the suite stays
hermetic across parallel runs.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import date
from pathlib import Path

import yaml

# Import the script as a module — ``scripts/`` is not a package.
_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "nightly_dead_slug_sweep.py"
_spec = importlib.util.spec_from_file_location("_nightly_sweep", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
sweep_mod = importlib.util.module_from_spec(_spec)
sys.modules["_nightly_sweep"] = sweep_mod
_spec.loader.exec_module(sweep_mod)

run_sweep = sweep_mod.run_sweep
load_history = sweep_mod.load_history
compute_trend = sweep_mod.compute_trend
render_markdown = sweep_mod.render_markdown
maybe_notify = sweep_mod.maybe_notify


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_factors_yml(tmp_path: Path) -> Path:
    p = tmp_path / "factors.yml"
    p.write_text(
        yaml.safe_dump(
            {
                "factors": [
                    {"id": "a", "slug": "slug-a", "source": "polymarket", "theme": "macro"},
                    {"id": "b", "slug": "slug-b", "source": "kalshi", "theme": "crypto"},
                ]
            }
        )
    )
    return p


def _dead_record(
    fid: str,
    *,
    source: str = "polymarket",
    theme: str = "macro",
    obs: int = 0,
    reason: str = "no_data_returned",
) -> dict:
    return {
        "id": fid,
        "slug": f"slug-{fid}",
        "source": source,
        "theme": theme,
        "obs_count": obs,
        "reason": reason,
    }


def _write_history(report_dir: Path, entries: list[tuple[str, int]]) -> None:
    """Drop dead-slug JSON files named per the date-stamp convention."""
    report_dir.mkdir(parents=True, exist_ok=True)
    for date_str, count in entries:
        (report_dir / f"dead-slugs-{date_str}.json").write_text(
            json.dumps(
                {
                    "checked_at": f"{date_str}T02:00:00+00:00",
                    "report_date": date_str,
                    "dead_count": count,
                    "dead_slugs": [],
                }
            )
        )


# ---------------------------------------------------------------------------
# 1. run_sweep — happy path with mocked detector
# ---------------------------------------------------------------------------


def test_run_sweep_writes_json_and_markdown(tmp_path):
    factors_yml = _fake_factors_yml(tmp_path)
    report_dir = tmp_path / "out"
    history_dir = tmp_path / "history"

    def fake_detect(path, *, min_obs, since_days, fetch_history=None):
        assert Path(path) == factors_yml
        assert min_obs == 30
        assert since_days == 90
        return [_dead_record("a"), _dead_record("b", source="kalshi")]

    result = run_sweep(
        factors_yml=factors_yml,
        report_dir=report_dir,
        history_dir=history_dir,
        today=date(2026, 5, 16),
        detect_fn=fake_detect,
        env={},
    )

    assert result["dead_count"] == 2
    json_path = Path(result["json_path"])
    md_path = Path(result["md_path"])
    assert json_path.exists()
    assert md_path.exists()
    assert json_path.name == "dead-slugs-2026-05-16.json"
    assert md_path.name == "dead-slugs-2026-05-16.md"

    payload = json.loads(json_path.read_text())
    assert payload["dead_count"] == 2
    assert payload["report_date"] == "2026-05-16"
    assert {r["id"] for r in payload["dead_slugs"]} == {"a", "b"}

    md_body = md_path.read_text()
    assert "Dead-slug sweep — 2026-05-16" in md_body
    assert "Dead slugs detected: **2**" in md_body
    # Markdown table contains both rows.
    assert "`slug-a`" in md_body
    assert "`slug-b`" in md_body


# ---------------------------------------------------------------------------
# 2. run_sweep — clean catalog
# ---------------------------------------------------------------------------


def test_run_sweep_clean_catalog_no_dead(tmp_path):
    factors_yml = _fake_factors_yml(tmp_path)

    def fake_detect(path, *, min_obs, since_days, fetch_history=None):
        return []

    result = run_sweep(
        factors_yml=factors_yml,
        report_dir=tmp_path / "out",
        history_dir=tmp_path / "history",
        today=date(2026, 5, 16),
        detect_fn=fake_detect,
        env={},
    )
    assert result["dead_count"] == 0
    md = Path(result["md_path"]).read_text()
    assert "_None — catalog clean._" in md
    assert "Dead slugs detected: **0**" in md
    assert result["trend"]["alert"] is False


# ---------------------------------------------------------------------------
# 3. compute_trend — within threshold
# ---------------------------------------------------------------------------


def test_compute_trend_within_threshold():
    history = [(date(2026, 5, 10), 100), (date(2026, 5, 11), 110), (date(2026, 5, 12), 105)]
    trend = compute_trend(108, history, threshold=0.20)
    assert trend["alert"] is False
    assert trend["reason"] == "within_threshold"
    assert trend["samples"] == 3
    assert trend["baseline"] == 105.0


# ---------------------------------------------------------------------------
# 4. compute_trend — spike alert
# ---------------------------------------------------------------------------


def test_compute_trend_spike_above_20pct():
    history = [(date(2026, 5, 10), 10), (date(2026, 5, 11), 10), (date(2026, 5, 12), 10)]
    trend = compute_trend(20, history, threshold=0.20)
    assert trend["alert"] is True
    assert trend["reason"] == "spike"
    assert trend["pct_change"] == 1.0  # 100% rise
    assert trend["delta"] == 10


# ---------------------------------------------------------------------------
# 5. compute_trend — drop alert + no history + zero baseline edge
# ---------------------------------------------------------------------------


def test_compute_trend_drop_no_history_and_zero_baseline():
    # Drop case
    history = [(date(2026, 5, 10), 100)]
    trend = compute_trend(50, history, threshold=0.20)
    assert trend["alert"] is True
    assert trend["reason"] == "drop"

    # No history
    trend2 = compute_trend(10, [], threshold=0.20)
    assert trend2["alert"] is False
    assert trend2["reason"] == "no_history"
    assert trend2["baseline"] is None
    assert trend2["samples"] == 0

    # Zero baseline + nonzero current → infinity guard kicks in
    trend3 = compute_trend(5, [(date(2026, 5, 10), 0)], threshold=0.20)
    assert trend3["alert"] is True
    assert trend3["pct_change_inf"] is True
    # And zero -> zero is not an alert
    trend4 = compute_trend(0, [(date(2026, 5, 10), 0)], threshold=0.20)
    assert trend4["alert"] is False


# ---------------------------------------------------------------------------
# 6. load_history — date filtering + malformed file robustness
# ---------------------------------------------------------------------------


def test_load_history_filters_window_and_ignores_garbage(tmp_path):
    report_dir = tmp_path / "out"
    _write_history(
        report_dir,
        [
            ("2026-05-09", 10),  # inside window (today - 7 = 2026-05-09)
            ("2026-05-13", 12),
            ("2026-05-15", 18),
            ("2026-05-16", 20),  # today — must be excluded
            ("2026-04-01", 99),  # outside window — excluded
        ],
    )
    # Stray garbage files
    (report_dir / "dead-slugs-not-a-date.json").write_text("garbage")
    (report_dir / "dead-slugs-2026-05-14.json").write_text("{not valid json")
    (report_dir / "unrelated.txt").write_text("ignore me")

    history = load_history(report_dir, today=date(2026, 5, 16), window_days=7)
    dates = [d for d, _ in history]
    counts = [c for _, c in history]
    assert dates == [date(2026, 5, 9), date(2026, 5, 13), date(2026, 5, 15)]
    assert counts == [10, 12, 18]


# ---------------------------------------------------------------------------
# 7. run_sweep — trend integration uses prior reports
# ---------------------------------------------------------------------------


def test_run_sweep_trend_alert_fires_when_spike(tmp_path):
    factors_yml = _fake_factors_yml(tmp_path)
    report_dir = tmp_path / "out"
    _write_history(
        report_dir,
        [
            ("2026-05-10", 10),
            ("2026-05-11", 10),
            ("2026-05-12", 10),
        ],
    )

    def fake_detect(path, *, min_obs, since_days, fetch_history=None):
        return [_dead_record(f"a{i}") for i in range(30)]

    result = run_sweep(
        factors_yml=factors_yml,
        report_dir=report_dir,
        history_dir=tmp_path / "history",
        today=date(2026, 5, 16),
        detect_fn=fake_detect,
        env={},
    )
    assert result["dead_count"] == 30
    assert result["trend"]["alert"] is True
    assert result["trend"]["reason"] == "spike"
    md = Path(result["md_path"]).read_text()
    assert "TREND_ALERT" in md
    assert "+200.0%" in md  # 30 vs baseline 10


# ---------------------------------------------------------------------------
# 8. maybe_notify — Slack webhook path with mocked urlopen
# ---------------------------------------------------------------------------


def test_maybe_notify_slack_success_on_alert():
    calls: list[dict] = []

    def fake_urlopen(req, timeout=None):
        calls.append(
            {
                "url": req.full_url,
                "data": req.data.decode("utf-8") if req.data else "",
                "method": req.get_method(),
                "headers": dict(req.headers),
            }
        )

        class _Resp:
            status = 200

            def close(self):
                pass

        return _Resp()

    result = maybe_notify(
        report_date=date(2026, 5, 16),
        dead_count=42,
        trend={"alert": True, "reason": "spike", "pct_change": 1.5},
        env={"PFM_DEAD_SLUG_SLACK_WEBHOOK": "https://hooks.slack.example/T/B/X"},
        urlopen=fake_urlopen,
    )
    assert result["channel"] == "slack"
    assert result["sent"] is True
    assert result["url_host"] == "hooks.slack.example"
    assert len(calls) == 1
    body = json.loads(calls[0]["data"])
    assert "42 dead" in body["text"]
    assert calls[0]["method"] == "POST"


# ---------------------------------------------------------------------------
# 9. maybe_notify — no-alert + no-force = no-op
# ---------------------------------------------------------------------------


def test_maybe_notify_skips_when_no_alert_and_no_force():
    result = maybe_notify(
        report_date=date(2026, 5, 16),
        dead_count=5,
        trend={"alert": False, "reason": "within_threshold"},
        env={"PFM_DEAD_SLUG_SLACK_WEBHOOK": "https://hooks.slack.example/X"},
    )
    assert result["channel"] == "none"
    assert result["sent"] is False
    assert result["reason"] == "no_alert_and_not_forced"


# ---------------------------------------------------------------------------
# 10. maybe_notify — always-notify override + Slack transport error
# ---------------------------------------------------------------------------


def test_maybe_notify_always_force_and_transport_error():
    import urllib.error

    def boom(req, timeout=None):
        raise urllib.error.URLError("boom")

    result = maybe_notify(
        report_date=date(2026, 5, 16),
        dead_count=0,
        trend={"alert": False, "reason": "within_threshold"},
        env={
            "PFM_DEAD_SLUG_SLACK_WEBHOOK": "https://hooks.slack.example/X",
            "PFM_DEAD_SLUG_ALWAYS_NOTIFY": "1",
        },
        urlopen=boom,
    )
    assert result["channel"] == "slack"
    assert result["sent"] is False
    assert "boom" in result.get("error", "")


# ---------------------------------------------------------------------------
# 11. maybe_notify — email stub (logs only)
# ---------------------------------------------------------------------------


def test_maybe_notify_email_stub_records_target_but_does_not_send():
    result = maybe_notify(
        report_date=date(2026, 5, 16),
        dead_count=11,
        trend={"alert": True, "reason": "spike", "pct_change": 0.5},
        env={"PFM_DEAD_SLUG_EMAIL": "ops@example.test"},
    )
    assert result["channel"] == "email"
    assert result["sent"] is False
    assert result["to"] == "ops@example.test"
    assert result["reason"] == "smtp_not_configured"


# ---------------------------------------------------------------------------
# 12. render_markdown — notification block + sorting
# ---------------------------------------------------------------------------


def test_render_markdown_includes_notification_and_sorts_rows():
    dead = [
        _dead_record("z", source="polymarket"),
        _dead_record("a", source="kalshi"),
        _dead_record("m", source="polymarket"),
    ]
    md = render_markdown(
        report_date=date(2026, 5, 16),
        dead=dead,
        min_obs=30,
        since_days=90,
        trend={
            "alert": True,
            "reason": "spike",
            "baseline": 1.0,
            "delta": 2.0,
            "pct_change": 2.0,
            "samples": 3,
        },
        factors_yml=Path("/tmp/factors.yml"),
        factors_total=1228,
        notification={"channel": "slack", "sent": True},
    )
    # Sorted (source, slug): kalshi/slug-a, polymarket/slug-m, polymarket/slug-z
    idx_a = md.index("slug-a")
    idx_m = md.index("slug-m")
    idx_z = md.index("slug-z")
    assert idx_a < idx_m < idx_z
    assert "Total factors scanned: **1228**" in md
    assert "## Notification" in md
    assert "TREND_ALERT" in md
    assert "Channel: `slack`" in md


# ---------------------------------------------------------------------------
# 13. run_sweep — fetch_history is forwarded to detector
# ---------------------------------------------------------------------------


def test_run_sweep_forwards_fetch_history_to_detector(tmp_path):
    factors_yml = _fake_factors_yml(tmp_path)

    captured: dict = {}

    def fake_fetch(factor, cutoff):
        return [1] * 50

    def fake_detect(path, *, min_obs, since_days, fetch_history=None):
        captured["fetch_history"] = fetch_history
        return []

    run_sweep(
        factors_yml=factors_yml,
        report_dir=tmp_path / "out",
        history_dir=tmp_path / "history",
        today=date(2026, 5, 16),
        detect_fn=fake_detect,
        fetch_history=fake_fetch,
        env={},
    )
    assert captured["fetch_history"] is fake_fetch


# ---------------------------------------------------------------------------
# 14. run_sweep — does NOT modify factors.yml
# ---------------------------------------------------------------------------


def test_run_sweep_does_not_touch_factors_yml(tmp_path):
    factors_yml = _fake_factors_yml(tmp_path)
    before = factors_yml.read_text()

    def fake_detect(path, *, min_obs, since_days, fetch_history=None):
        return [_dead_record("a")]

    run_sweep(
        factors_yml=factors_yml,
        report_dir=tmp_path / "out",
        history_dir=tmp_path / "history",
        today=date(2026, 5, 16),
        detect_fn=fake_detect,
        env={},
    )
    after = factors_yml.read_text()
    assert before == after
    # And no backup file is created either.
    siblings = list(factors_yml.parent.glob("factors.yml*"))
    assert siblings == [factors_yml]


# ---------------------------------------------------------------------------
# 15. CLI smoke — main() parses args and writes outputs
# ---------------------------------------------------------------------------


def test_cli_main_runs_with_overrides(tmp_path, monkeypatch, capsys):
    factors_yml = _fake_factors_yml(tmp_path)

    def fake_detect(path, *, min_obs, since_days, fetch_history=None):
        return [_dead_record("a"), _dead_record("b")]

    monkeypatch.setattr(sweep_mod, "_resolve_detector", lambda: fake_detect)

    rc = sweep_mod.main(
        [
            "--factors-yml",
            str(factors_yml),
            "--report-dir",
            str(tmp_path / "out"),
            "--history-dir",
            str(tmp_path / "history"),
            "--date",
            "2026-05-16",
            "--min-obs",
            "15",
            "--since-days",
            "30",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "Nightly dead-slug sweep 2026-05-16" in out
    assert "2 dead" in out
    assert (tmp_path / "out" / "dead-slugs-2026-05-16.json").exists()
    assert (tmp_path / "history" / "dead-slugs-2026-05-16.md").exists()
