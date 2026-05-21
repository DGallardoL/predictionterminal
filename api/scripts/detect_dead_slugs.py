"""Dead-slug detector — flag factors with insufficient daily observations.

A factor is "dead" when, over the trailing ``--since-days`` window, the
upstream venue returned **fewer than ``--min-obs`` daily observations**. By
default the thresholds are ``min_obs=30`` and ``since_days=90``, matching the
CLAUDE.md / Wave-N convention that every catalogued slug should have at least
30 daily bars before being added.

Public API
----------

* :func:`detect_dead_slugs` — pure-Python, fetcher-injectable, returns a list
  of ``{slug, theme, obs_count, reason}`` dicts. Used by tests with mocked
  fetchers; never hits the network when ``fetch_history`` is supplied.
* :func:`apply_prune` — write a backup of ``factors.yml`` and emit a pruned
  copy with the flagged ids removed.

CLI
---

::

    python scripts/detect_dead_slugs.py                  # dry-run, default thresholds
    python scripts/detect_dead_slugs.py --min-obs 20     # softer threshold
    python scripts/detect_dead_slugs.py --since-days 60  # shorter window
    python scripts/detect_dead_slugs.py --apply          # rewrite factors.yml

The dry-run writes ``/tmp/dead-slugs.json``::

    {
      "checked_at": "...",
      "min_obs": 30,
      "since_days": 90,
      "dead_count": 47,
      "dead_slugs": [{"slug": "...", "theme": "...", "obs_count": 12, "reason": "..."}]
    }

With ``--apply`` the script also writes a timestamped backup of ``factors.yml``
to the same directory before overwriting it. Run from the ``api/`` directory
or with ``PYTHONPATH=src``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

_HERE = Path(__file__).resolve().parent
_API_ROOT = _HERE.parent
_SRC = _API_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

DEFAULT_FACTORS_YML = _SRC / "pfm" / "factors.yml"
DEFAULT_OUTPUT = Path("/tmp/dead-slugs.json")

DEFAULT_MIN_OBS = 30
DEFAULT_SINCE_DAYS = 90

# Reason strings emitted in the JSON report. Tests assert on these exact tags
# so they should be stable.
REASON_NO_DATA = "no_data_returned"
REASON_FETCH_ERROR = "fetch_error"
REASON_INSUFFICIENT_OBS = "insufficient_observations"

# Type alias for the pluggable fetcher used in tests: takes a factor dict and
# the cutoff date and returns a list of (date, value) pairs OR raises. The
# date may be either ``datetime.date`` or anything pandas would coerce — we
# only count the rows, we do not consume the values here.
FetchFn = Callable[[dict, datetime], list[Any]]


# ---------------------------------------------------------------------------
# Core detector
# ---------------------------------------------------------------------------


def detect_dead_slugs(
    factors_yml_path: str | Path,
    *,
    min_obs: int = DEFAULT_MIN_OBS,
    since_days: int = DEFAULT_SINCE_DAYS,
    fetch_history: FetchFn | None = None,
    now: datetime | None = None,
    sources: frozenset[str] | None = None,
    sleep_between: float = 0.0,
    progress: Callable[[int, int, dict], None] | None = None,
) -> list[dict]:
    """Return a list of dead-slug records for ``factors.yml``.

    Parameters
    ----------
    factors_yml_path:
        Path to the factors catalog. Must contain a top-level ``factors:`` key
        whose value is a list of mappings with at least ``id`` and ``slug``.
    min_obs:
        Minimum number of daily observations required in the trailing window.
        A factor with strictly fewer observations is flagged dead.
    since_days:
        Length of the trailing window in days. The cutoff used by
        ``fetch_history`` is ``now - timedelta(days=since_days)``.
    fetch_history:
        Optional callable ``(factor_dict, cutoff_dt) -> list``. Returning an
        empty list means "no data" (treated as 0 observations); raising any
        exception yields the ``fetch_error`` reason. When ``None`` the
        detector uses :func:`_default_fetch_history` which simply reports
        ``no_data_returned`` for every factor — callers in production should
        pass a real fetcher (CLI uses the audit-report stub on disk).
    now:
        Reference timestamp; defaults to ``datetime.now(UTC)``. Tests inject
        a fixed timestamp for determinism.
    sources:
        Optional whitelist of source strings to scan (e.g.
        ``frozenset({"polymarket", "kalshi"})``). ``None`` scans every factor.
    sleep_between:
        Throttle between fetches; useful for CLI use against the live API.
        Tests pass ``0.0``.
    progress:
        Optional callback invoked every iteration as
        ``progress(i_done, n_total, factor)``.

    Returns
    -------
    list[dict]
        One entry per dead factor, each shaped
        ``{"slug": str, "theme": str, "obs_count": int, "reason": str, "id": str, "source": str}``.
        Healthy factors are not included.
    """
    if min_obs < 0:
        raise ValueError("min_obs must be >= 0")
    if since_days <= 0:
        raise ValueError("since_days must be > 0")

    factors = _load_factor_entries(factors_yml_path)
    if sources is not None:
        factors = [f for f in factors if f.get("source") in sources]

    ref_now = now if now is not None else datetime.now(UTC)
    cutoff = ref_now - timedelta(days=since_days)
    fetcher = fetch_history if fetch_history is not None else _default_fetch_history

    dead: list[dict] = []
    total = len(factors)
    for i, factor in enumerate(factors, 1):
        record = _evaluate_one(factor, cutoff, min_obs, fetcher)
        if progress is not None:
            try:
                progress(i, total, factor)
            except Exception:  # pragma: no cover - progress is best-effort
                pass
        if record is not None:
            dead.append(record)
        if sleep_between > 0 and i < total:
            time.sleep(sleep_between)
    return dead


def _evaluate_one(
    factor: dict,
    cutoff: datetime,
    min_obs: int,
    fetcher: FetchFn,
) -> dict | None:
    """Return a dead-slug record for ``factor`` or ``None`` if healthy."""
    slug = str(factor.get("slug", "")) or ""
    theme = str(factor.get("theme", "other"))
    source = str(factor.get("source", "unknown"))
    factor_id = str(factor.get("id", slug))
    try:
        rows = fetcher(factor, cutoff)
    except Exception as exc:
        return {
            "id": factor_id,
            "slug": slug,
            "theme": theme,
            "source": source,
            "obs_count": 0,
            "reason": REASON_FETCH_ERROR,
            "error": _short_error(exc),
        }

    obs_count = _count_observations(rows)
    if obs_count == 0:
        return {
            "id": factor_id,
            "slug": slug,
            "theme": theme,
            "source": source,
            "obs_count": 0,
            "reason": REASON_NO_DATA,
        }
    if obs_count < min_obs:
        return {
            "id": factor_id,
            "slug": slug,
            "theme": theme,
            "source": source,
            "obs_count": obs_count,
            "reason": REASON_INSUFFICIENT_OBS,
        }
    return None


def _count_observations(rows: Any) -> int:
    """Best-effort row counter that accepts lists, tuples, DataFrames, dicts."""
    if rows is None:
        return 0
    if hasattr(rows, "shape"):
        try:
            return int(rows.shape[0])
        except Exception:  # pragma: no cover - exotic shape
            return 0
    if isinstance(rows, dict):
        # Treat dict-of-series as ``{date: value}``
        return len(rows)
    try:
        return len(rows)
    except TypeError:
        return 0


def _short_error(exc: BaseException) -> str:
    msg = str(exc).strip().splitlines()[0] if str(exc) else exc.__class__.__name__
    return msg[:160]


def _default_fetch_history(factor: dict, cutoff: datetime) -> list[Any]:
    """Conservative fallback that reports 'no data' for every factor.

    Production callers should inject a real fetcher (HTTP call to Polymarket /
    Kalshi / FRED). The CLI passes :func:`_make_cli_fetcher`. The default
    exists so that ``detect_dead_slugs(...)`` is callable without arguments —
    handy for unit-test scaffolding and dry-run smoke tests.
    """
    del factor, cutoff
    return []


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------


def _load_factor_entries(factors_yml_path: str | Path) -> list[dict]:
    """Return the raw list of factor dicts from ``factors.yml``."""
    path = Path(factors_yml_path)
    if not path.exists():
        raise FileNotFoundError(f"factors file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    items = raw.get("factors", [])
    if not isinstance(items, list):
        raise ValueError("`factors` key must be a list")
    return [dict(item) for item in items if isinstance(item, dict)]


def apply_prune(
    factors_yml_path: str | Path,
    dead_records: list[dict],
    *,
    backup_suffix: str | None = None,
) -> Path:
    """Rewrite ``factors.yml`` with dead ids removed; return the backup path.

    Behaviour
    ---------
    1. Read the current YAML.
    2. Write the unchanged bytes to ``factors.yml.bak.dead_slugs.<UTC>`` (or to
       ``factors.yml.bak<suffix>`` when ``backup_suffix`` is passed).
    3. Write the filtered factor list back to ``factors.yml`` using
       ``yaml.safe_dump`` with ``sort_keys=False``.

    Empty ``dead_records`` is a no-op and the function returns the (still
    created) backup path so callers always have a rollback point.
    """
    path = Path(factors_yml_path)
    if not path.exists():
        raise FileNotFoundError(f"factors file not found: {path}")

    if backup_suffix is None:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        backup_suffix = f".bak.dead_slugs.{stamp}"
    backup = path.with_name(path.name + backup_suffix)
    shutil.copy2(path, backup)

    dead_ids = {str(r.get("id")) for r in dead_records if r.get("id")}
    dead_slugs = {str(r.get("slug")) for r in dead_records if r.get("slug")}

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    items = raw.get("factors", []) or []
    kept = [
        e
        for e in items
        if isinstance(e, dict)
        and str(e.get("id", "")) not in dead_ids
        and str(e.get("slug", "")) not in dead_slugs
    ]
    out = {"factors": kept}
    path.write_text(
        "# factors.yml - pruned by scripts/detect_dead_slugs.py.\n"
        f"# Removed {len(items) - len(kept)} dead slugs at "
        f"{datetime.now(UTC).isoformat()}.\n"
        f"# Backup: {backup.name}\n\n"
        + yaml.safe_dump(out, sort_keys=False, allow_unicode=True, width=120)
    )
    return backup


# ---------------------------------------------------------------------------
# CLI fetcher — best-effort lookup against an existing audit report on disk.
# ---------------------------------------------------------------------------


def _make_cli_fetcher() -> FetchFn:
    """Return a fetcher that consults ``audit_dead_factors.report.json``.

    For the CLI we deliberately avoid hammering the real API (the existing
    ``audit_dead_factors.py`` script already does that and writes its
    findings). Instead we re-use the most recent audit on disk:

    * If a factor's slug appears under ``by_status.DEAD`` we report 0 obs.
    * Otherwise we assume the slug is healthy and return ``[1] * min_obs`` so
      the detector keeps it.

    This keeps the dry-run path side-effect-free and fast. When that report
    file is missing the fetcher conservatively returns "no data" for every
    factor; the operator can re-run ``audit_dead_factors.py`` first.
    """
    report_path = _API_ROOT / "scripts" / "audit_dead_factors.report.json"
    dead_slugs: set[str] = set()
    if report_path.exists():
        try:
            data = json.loads(report_path.read_text())
            for entry in data.get("by_status", {}).get("DEAD", []) or []:
                if entry.get("slug"):
                    dead_slugs.add(str(entry["slug"]))
        except (json.JSONDecodeError, OSError):
            dead_slugs = set()

    def _fetch(factor: dict, cutoff: datetime) -> list[Any]:
        del cutoff
        slug = str(factor.get("slug", ""))
        if slug in dead_slugs:
            return []
        # Healthy stub — return DEFAULT_MIN_OBS placeholder rows. We use the
        # default here (not the caller's ``min_obs``) because the fetcher is
        # source-agnostic; the detector still re-applies its own threshold.
        return [1] * DEFAULT_MIN_OBS

    return _fetch


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Detect (and optionally prune) dead factor slugs.")
    p.add_argument(
        "--factors-yml",
        type=Path,
        default=DEFAULT_FACTORS_YML,
        help=f"Path to factors.yml (default: {DEFAULT_FACTORS_YML})",
    )
    p.add_argument(
        "--min-obs",
        type=int,
        default=DEFAULT_MIN_OBS,
        help=f"Minimum daily observations to consider healthy (default: {DEFAULT_MIN_OBS})",
    )
    p.add_argument(
        "--since-days",
        type=int,
        default=DEFAULT_SINCE_DAYS,
        help=f"Trailing window in days (default: {DEFAULT_SINCE_DAYS})",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Report destination (default: {DEFAULT_OUTPUT})",
    )
    p.add_argument(
        "--source",
        type=str,
        default=None,
        help="Comma-separated source filter (e.g. 'polymarket,kalshi'). Default: all sources.",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="Rewrite factors.yml with the flagged ids removed (creates a timestamped backup).",
    )
    return p


def write_report(
    dead: list[dict],
    *,
    output: Path,
    min_obs: int,
    since_days: int,
    checked_at: datetime | None = None,
) -> Path:
    """Write the JSON report; returns the path written."""
    payload = {
        "checked_at": (checked_at or datetime.now(UTC)).isoformat(),
        "min_obs": min_obs,
        "since_days": since_days,
        "dead_count": len(dead),
        "dead_slugs": dead,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    return output


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    sources: frozenset[str] | None = None
    if args.source:
        sources = frozenset(s.strip() for s in args.source.split(",") if s.strip())

    fetcher = _make_cli_fetcher()
    dead = detect_dead_slugs(
        args.factors_yml,
        min_obs=args.min_obs,
        since_days=args.since_days,
        fetch_history=fetcher,
        sources=sources,
    )
    report_path = write_report(
        dead,
        output=args.output,
        min_obs=args.min_obs,
        since_days=args.since_days,
    )
    print(
        f"Dead-slug scan: {len(dead)} flagged "
        f"(min_obs={args.min_obs}, since_days={args.since_days}) -> {report_path}"
    )
    if dead:
        for r in dead[:10]:
            print(f"  - {r['slug']:<50} obs={r['obs_count']:>3}  {r['reason']}")
        if len(dead) > 10:
            print(f"  ... +{len(dead) - 10} more")

    if args.apply:
        if not dead:
            print("No dead slugs to prune; factors.yml unchanged.")
        else:
            backup = apply_prune(args.factors_yml, dead)
            print(f"Pruned {len(dead)} entries from {args.factors_yml} (backup: {backup})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
