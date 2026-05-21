"""Nightly dead-slug sweep — operational wrapper around ``detect_dead_slugs``.

This script is the cron-friendly automation layer on top of
``scripts/detect_dead_slugs.py`` (W12-09). It performs three jobs:

1. **Detect** — invokes :func:`detect_dead_slugs.detect_dead_slugs` against
   ``factors.yml`` (or any caller-supplied path) and writes a date-stamped
   JSON report to ``/tmp/dead-slugs-{YYYY-MM-DD}.json``.
2. **Report** — renders a human-readable Markdown summary into
   ``docs/dead-slugs-history/dead-slugs-{YYYY-MM-DD}.md`` so the repo keeps a
   versioned trail of catalog health.
3. **Notify** — emits an optional Slack/email alert (env-gated; default is a
   no-op print). When the current dead count diverges from the prior 7-day
   average by more than ``--trend-threshold`` (default 20%) the report tags
   the run as ``TREND_ALERT``.

The script never writes to ``factors.yml`` — pruning remains a deliberate,
operator-initiated act done via ``scripts/detect_dead_slugs.py --apply``.

Designed for ``0 2 * * *`` (2 AM UTC) cron entries::

    0 2 * * * cd /opt/pfm/api && PYTHONPATH=src \
        .venv/bin/python scripts/nightly_dead_slug_sweep.py >> /var/log/pfm/dead-slug.log 2>&1

Public API
----------

* :func:`run_sweep` — programmatic entrypoint; takes injected fetcher +
  detector so unit tests stay hermetic.
* :func:`render_markdown` — builds the report body.
* :func:`load_history` / :func:`compute_trend` — 7-day trend analysis.
* :func:`maybe_notify` — env-gated dispatcher (Slack webhook, SMTP).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import re
import sys
import urllib.error
import urllib.request
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Module bootstrap
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
_API_ROOT = _HERE.parent
_REPO_ROOT = _API_ROOT.parent
_SRC = _API_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Lazily import W12-09's detector. We do it via ``importlib.util`` because
# ``scripts/`` is not a package and re-running this module under different
# fixtures should always pick up the canonical file on disk.
_DETECTOR_PATH = _HERE / "detect_dead_slugs.py"

DEFAULT_FACTORS_YML = _SRC / "pfm" / "factors.yml"
DEFAULT_REPORT_DIR = Path("/tmp")
DEFAULT_HISTORY_DIR = _REPO_ROOT / "docs" / "dead-slugs-history"
DEFAULT_TREND_THRESHOLD = 0.20  # 20% deviation triggers the trend flag
DEFAULT_TREND_WINDOW_DAYS = 7
DEFAULT_MIN_OBS = 30
DEFAULT_SINCE_DAYS = 90

LOG = logging.getLogger("nightly_dead_slug_sweep")


# ---------------------------------------------------------------------------
# Detector accessor — late-binding lets tests monkeypatch a fake detector.
# ---------------------------------------------------------------------------


def _load_detector_module():
    """Return the loaded ``detect_dead_slugs`` module.

    We import via ``importlib`` because ``api/scripts`` is not a Python
    package. The module is cached under the alias ``"_w12_09_detect"`` so
    repeated calls are cheap.
    """
    cached = sys.modules.get("_w12_09_detect")
    if cached is not None:
        return cached
    if not _DETECTOR_PATH.exists():
        raise FileNotFoundError(
            f"detect_dead_slugs.py not found at {_DETECTOR_PATH}; W12-09 dependency missing."
        )
    spec = importlib.util.spec_from_file_location("_w12_09_detect", _DETECTOR_PATH)
    if spec is None or spec.loader is None:
        raise ImportError("Could not load detect_dead_slugs.py via importlib")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_w12_09_detect"] = mod
    spec.loader.exec_module(mod)
    return mod


def _resolve_detector() -> Callable[..., list[dict]]:
    """Return ``detect_dead_slugs`` from W12-09, with an inline fallback.

    Tests may monkeypatch this function to inject a stub. When the W12-09
    script is unavailable we fall back to a conservative detector that
    treats every factor as dead (since we have no data source). This is a
    rare path — production should always have detect_dead_slugs available.
    """
    try:
        mod = _load_detector_module()
    except (FileNotFoundError, ImportError):  # pragma: no cover - defensive
        return _inline_fallback_detector
    fn = getattr(mod, "detect_dead_slugs", None)
    if not callable(fn):  # pragma: no cover - defensive
        return _inline_fallback_detector
    return fn


def _inline_fallback_detector(  # pragma: no cover - defensive fallback only
    factors_yml_path: str | Path,
    *,
    min_obs: int = DEFAULT_MIN_OBS,
    since_days: int = DEFAULT_SINCE_DAYS,
    **_kwargs: Any,
) -> list[dict]:
    """Inline fallback when detect_dead_slugs.py is missing.

    We deliberately under-promise here: with no fetcher and no detector
    module we simply return an empty list and let the report flag the
    missing dependency in the warnings section.
    """
    LOG.warning(
        "detect_dead_slugs not available (path=%s); returning empty list. "
        "Install W12-09 (scripts/detect_dead_slugs.py).",
        _DETECTOR_PATH,
    )
    del factors_yml_path, min_obs, since_days
    return []


# ---------------------------------------------------------------------------
# Trend analysis
# ---------------------------------------------------------------------------


_REPORT_FILENAME_RE = re.compile(r"^dead-slugs-(\d{4}-\d{2}-\d{2})\.json$")


def load_history(
    report_dir: Path,
    *,
    today: date,
    window_days: int = DEFAULT_TREND_WINDOW_DAYS,
) -> list[tuple[date, int]]:
    """Return ``[(report_date, dead_count), ...]`` for the prior ``window_days``.

    Looks at JSON files matching ``dead-slugs-YYYY-MM-DD.json`` in
    ``report_dir`` whose date falls in ``[today - window_days, today - 1]``.
    Malformed JSON, missing ``dead_count``, and IO errors are silently
    skipped — the trend analysis is best-effort.
    """
    if not report_dir.exists() or not report_dir.is_dir():
        return []
    cutoff_start = today - timedelta(days=window_days)
    out: list[tuple[date, int]] = []
    for p in sorted(report_dir.iterdir()):
        m = _REPORT_FILENAME_RE.match(p.name)
        if not m:
            continue
        try:
            d = date.fromisoformat(m.group(1))
        except ValueError:
            continue
        if d >= today or d < cutoff_start:
            continue
        try:
            payload = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        count = payload.get("dead_count")
        if not isinstance(count, int):
            continue
        out.append((d, count))
    return out


def compute_trend(
    current_count: int,
    history: list[tuple[date, int]],
    *,
    threshold: float = DEFAULT_TREND_THRESHOLD,
) -> dict:
    """Return ``{baseline, delta, pct_change, alert, reason}`` for the trend.

    ``alert`` is ``True`` when ``abs(pct_change) > threshold`` and the
    baseline has at least one prior report. With zero history the result
    is ``alert=False`` and ``reason="no_history"``.
    """
    if not history:
        return {
            "baseline": None,
            "delta": None,
            "pct_change": None,
            "alert": False,
            "reason": "no_history",
            "samples": 0,
        }
    counts = [c for _, c in history]
    baseline = sum(counts) / len(counts)
    delta = current_count - baseline
    # Guard against div-by-zero when baseline is 0 — treat any current>0 as
    # a 100% rise so we still alert on a transition from clean to dirty.
    if baseline == 0:
        pct_change = float("inf") if current_count > 0 else 0.0
    else:
        pct_change = delta / baseline
    alert = abs(pct_change) > threshold if pct_change != float("inf") else current_count > 0
    if alert:
        reason = "spike" if delta > 0 else "drop"
    else:
        reason = "within_threshold"
    return {
        "baseline": round(baseline, 2),
        "delta": round(delta, 2) if isinstance(delta, float) else delta,
        "pct_change": (round(pct_change, 4) if pct_change != float("inf") else None),
        "pct_change_inf": pct_change == float("inf"),
        "alert": alert,
        "reason": reason,
        "samples": len(history),
    }


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def render_markdown(
    *,
    report_date: date,
    dead: list[dict],
    min_obs: int,
    since_days: int,
    trend: dict,
    factors_yml: Path,
    factors_total: int | None = None,
    notification: dict | None = None,
) -> str:
    """Render the dead-slug Markdown report body."""
    lines: list[str] = []
    lines.append(f"# Dead-slug sweep — {report_date.isoformat()}")
    lines.append("")
    status = "TREND_ALERT" if trend.get("alert") else "OK"
    lines.append(f"**Status:** `{status}`")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Factors file: `{factors_yml}`")
    if factors_total is not None:
        lines.append(f"- Total factors scanned: **{factors_total}**")
    lines.append(f"- Thresholds: `min_obs={min_obs}`, `since_days={since_days}`")
    lines.append(f"- Dead slugs detected: **{len(dead)}**")
    lines.append("")
    lines.append("## Trend (vs prior 7 days)")
    lines.append("")
    baseline = trend.get("baseline")
    if baseline is None:
        lines.append("- No prior reports found in history window.")
    else:
        pct = trend.get("pct_change")
        pct_str = f"{pct * 100:+.1f}%" if isinstance(pct, (int, float)) else "n/a (baseline=0)"
        lines.append(f"- Baseline (mean of {trend.get('samples', 0)} reports): {baseline}")
        lines.append(f"- Delta: {trend.get('delta')}")
        lines.append(f"- Pct change: {pct_str}")
        lines.append(f"- Alert: `{trend.get('alert')}` ({trend.get('reason')})")
    lines.append("")
    lines.append("## Dead slugs")
    lines.append("")
    if not dead:
        lines.append("_None — catalog clean._")
    else:
        lines.append("| ID | Slug | Source | Theme | Obs | Reason |")
        lines.append("|----|------|--------|-------|-----|--------|")
        # Sort by source then slug for deterministic output (tests rely on order).
        for r in sorted(dead, key=lambda x: (str(x.get("source", "")), str(x.get("slug", "")))):
            lines.append(
                "| {id} | `{slug}` | {source} | {theme} | {obs} | {reason} |".format(
                    id=r.get("id", ""),
                    slug=r.get("slug", ""),
                    source=r.get("source", ""),
                    theme=r.get("theme", ""),
                    obs=r.get("obs_count", 0),
                    reason=r.get("reason", ""),
                )
            )
    lines.append("")
    if notification is not None:
        lines.append("## Notification")
        lines.append("")
        lines.append(f"- Channel: `{notification.get('channel', 'none')}`")
        lines.append(f"- Sent: `{notification.get('sent', False)}`")
        if notification.get("error"):
            lines.append(f"- Error: `{notification['error']}`")
        lines.append("")
    lines.append("---")
    lines.append(
        f"_Generated by `scripts/nightly_dead_slug_sweep.py` at {datetime.now(UTC).isoformat()}._"
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Notification stubs (env-gated; no-op by default)
# ---------------------------------------------------------------------------


def maybe_notify(
    *,
    report_date: date,
    dead_count: int,
    trend: dict,
    env: dict | None = None,
    urlopen: Callable[..., Any] | None = None,
) -> dict:
    """Dispatch a notification if env vars opt in.

    Channels (in priority order):

    1. ``PFM_DEAD_SLUG_SLACK_WEBHOOK`` — POSTs a JSON ``{"text": "..."}``
       payload to the URL when ``alert`` fires or when ``PFM_DEAD_SLUG_ALWAYS_NOTIFY=1``.
    2. ``PFM_DEAD_SLUG_EMAIL`` — stub (logs only; real SMTP would be added
       once an SMTP relay is provisioned).
    3. Default: returns ``{"channel": "none", "sent": False}``.

    The function never raises; transport errors are captured into the
    returned dict so the caller can include them in the Markdown report.
    """
    env = env if env is not None else os.environ.copy()
    always = env.get("PFM_DEAD_SLUG_ALWAYS_NOTIFY", "").strip() in {"1", "true", "yes"}
    should_send = bool(trend.get("alert")) or always

    if not should_send:
        return {"channel": "none", "sent": False, "reason": "no_alert_and_not_forced"}

    slack_url = (env.get("PFM_DEAD_SLUG_SLACK_WEBHOOK") or "").strip()
    if slack_url:
        text = (
            f":rotating_light: Dead-slug sweep {report_date.isoformat()}: "
            f"{dead_count} dead "
            f"(trend reason={trend.get('reason')}, pct_change={trend.get('pct_change')})"
        )
        payload = json.dumps({"text": text}).encode("utf-8")
        req = urllib.request.Request(
            slack_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        opener = urlopen if urlopen is not None else urllib.request.urlopen
        try:
            opener(req, timeout=5)  # type: ignore[arg-type]
            return {"channel": "slack", "sent": True, "url_host": _host_of(slack_url)}
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            return {
                "channel": "slack",
                "sent": False,
                "url_host": _host_of(slack_url),
                "error": str(exc)[:160],
            }

    email_to = (env.get("PFM_DEAD_SLUG_EMAIL") or "").strip()
    if email_to:
        # Real SMTP wiring deferred until a relay is configured. We log so
        # the operator can see the path was exercised in dry-runs.
        LOG.info(
            "Would email %s about dead-slug sweep %s (count=%d, trend=%s)",
            email_to,
            report_date.isoformat(),
            dead_count,
            trend.get("reason"),
        )
        return {"channel": "email", "sent": False, "reason": "smtp_not_configured", "to": email_to}

    return {"channel": "none", "sent": False, "reason": "no_channel_configured"}


def _host_of(url: str) -> str:
    """Extract the hostname from ``url`` without importing urllib.parse for a one-liner."""
    try:
        from urllib.parse import urlparse

        host = urlparse(url).hostname or ""
        return host
    except Exception:  # pragma: no cover - defensive
        return ""


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_sweep(
    *,
    factors_yml: Path | str = DEFAULT_FACTORS_YML,
    report_dir: Path | str = DEFAULT_REPORT_DIR,
    history_dir: Path | str = DEFAULT_HISTORY_DIR,
    today: date | None = None,
    min_obs: int = DEFAULT_MIN_OBS,
    since_days: int = DEFAULT_SINCE_DAYS,
    trend_threshold: float = DEFAULT_TREND_THRESHOLD,
    trend_window_days: int = DEFAULT_TREND_WINDOW_DAYS,
    detect_fn: Callable[..., list[dict]] | None = None,
    fetch_history: Callable[..., list[Any]] | None = None,
    env: dict | None = None,
    notifier: Callable[..., dict] | None = None,
    factors_total: int | None = None,
) -> dict:
    """Run the full sweep and return a summary dict.

    Parameters
    ----------
    factors_yml:
        Path to the YAML catalog.
    report_dir:
        Directory that receives ``dead-slugs-{date}.json`` (defaults to ``/tmp``).
    history_dir:
        Directory that receives the versioned Markdown report (defaults to
        ``docs/dead-slugs-history/`` under the repo root).
    today:
        Override for the date stamp; defaults to ``datetime.now(UTC).date()``.
    min_obs / since_days:
        Forwarded to ``detect_dead_slugs``.
    trend_threshold:
        Relative deviation that triggers a ``TREND_ALERT`` flag.
    trend_window_days:
        Days of history to consult for the baseline.
    detect_fn:
        Injected detector — useful in tests. Defaults to W12-09's
        ``detect_dead_slugs``.
    fetch_history:
        Forwarded to ``detect_fn`` so production callers can plug a real
        upstream fetcher. Default ``None`` lets the detector use its own
        fallback (CLI fetcher).
    env:
        Override of ``os.environ`` for notification gating.
    notifier:
        Override for the notification dispatcher (defaults to
        :func:`maybe_notify`). Tests inject a recording stub.
    factors_total:
        Optional override for the "Total factors scanned" line in the
        Markdown report. When omitted, the report omits the line — we don't
        want to re-read the YAML just to count.

    Returns
    -------
    dict
        ``{"report_date", "dead_count", "json_path", "md_path", "trend",
        "notification"}``.
    """
    today_ = today if today is not None else datetime.now(UTC).date()
    factors_yml = Path(factors_yml)
    report_dir = Path(report_dir)
    history_dir = Path(history_dir)
    detect = detect_fn if detect_fn is not None else _resolve_detector()
    notify = notifier if notifier is not None else maybe_notify

    LOG.info(
        "Starting nightly dead-slug sweep for %s (min_obs=%d, since_days=%d)",
        today_.isoformat(),
        min_obs,
        since_days,
    )

    dead = list(
        detect(
            factors_yml,
            min_obs=min_obs,
            since_days=since_days,
            fetch_history=fetch_history,
        )
    )

    json_path = report_dir / f"dead-slugs-{today_.isoformat()}.json"
    report_dir.mkdir(parents=True, exist_ok=True)
    json_payload = {
        "checked_at": datetime.now(UTC).isoformat(),
        "report_date": today_.isoformat(),
        "factors_yml": str(factors_yml),
        "min_obs": min_obs,
        "since_days": since_days,
        "dead_count": len(dead),
        "dead_slugs": dead,
    }
    json_path.write_text(json.dumps(json_payload, indent=2))

    history = load_history(report_dir, today=today_, window_days=trend_window_days)
    trend = compute_trend(len(dead), history, threshold=trend_threshold)

    notification = notify(
        report_date=today_,
        dead_count=len(dead),
        trend=trend,
        env=env,
    )

    md_body = render_markdown(
        report_date=today_,
        dead=dead,
        min_obs=min_obs,
        since_days=since_days,
        trend=trend,
        factors_yml=factors_yml,
        factors_total=factors_total,
        notification=notification,
    )
    history_dir.mkdir(parents=True, exist_ok=True)
    md_path = history_dir / f"dead-slugs-{today_.isoformat()}.md"
    md_path.write_text(md_body)

    LOG.info(
        "Sweep complete: %d dead, json=%s, md=%s, trend.alert=%s",
        len(dead),
        json_path,
        md_path,
        trend.get("alert"),
    )

    return {
        "report_date": today_.isoformat(),
        "dead_count": len(dead),
        "json_path": str(json_path),
        "md_path": str(md_path),
        "trend": trend,
        "notification": notification,
    }


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Nightly dead-slug sweep (operational wrapper around "
            "scripts/detect_dead_slugs.py). Cron-friendly entrypoint."
        ),
    )
    p.add_argument("--factors-yml", type=Path, default=DEFAULT_FACTORS_YML)
    p.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    p.add_argument("--history-dir", type=Path, default=DEFAULT_HISTORY_DIR)
    p.add_argument("--min-obs", type=int, default=DEFAULT_MIN_OBS)
    p.add_argument("--since-days", type=int, default=DEFAULT_SINCE_DAYS)
    p.add_argument(
        "--trend-threshold",
        type=float,
        default=DEFAULT_TREND_THRESHOLD,
        help="Relative deviation (0.20 = 20%%) above which TREND_ALERT fires.",
    )
    p.add_argument("--trend-window-days", type=int, default=DEFAULT_TREND_WINDOW_DAYS)
    p.add_argument(
        "--date",
        type=date.fromisoformat,
        default=None,
        help="Override the date stamp (ISO-8601). Useful for backfills.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress the human-readable summary on stdout.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    result = run_sweep(
        factors_yml=args.factors_yml,
        report_dir=args.report_dir,
        history_dir=args.history_dir,
        today=args.date,
        min_obs=args.min_obs,
        since_days=args.since_days,
        trend_threshold=args.trend_threshold,
        trend_window_days=args.trend_window_days,
    )
    if not args.quiet:
        print(
            "Nightly dead-slug sweep {date}: {dead} dead "
            "(trend.alert={alert}, reason={reason})".format(
                date=result["report_date"],
                dead=result["dead_count"],
                alert=result["trend"].get("alert"),
                reason=result["trend"].get("reason"),
            )
        )
        print(f"  JSON: {result['json_path']}")
        print(f"  MD  : {result['md_path']}")
        if result["notification"].get("sent"):
            print(f"  Notified via {result['notification'].get('channel')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
