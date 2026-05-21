"""Per-theme coverage statistics for ``factors.yml``.

For every factor declared in ``factors.yml`` we attempt to fetch its
recent daily history and record how many observations come back. We
group those results by ``theme`` and emit a JSON report shaped like::

    {
      "checked_at": "2026-05-16T03:14:15Z",
      "themes": [
        {
          "theme": "politics",
          "factor_count": 412,
          "with_data": 380,
          "median_obs": 256,
          "min_obs": 5
        },
        ...
      ],
      "totals": {
        "factors": 1228,
        "themes": 19,
        "with_data": 1145,
        "stale": 83
      }
    }

A factor counts as "with data" when ``fetch_factor_history_dispatch``
returns at least ``--min-obs`` rows (default: 1). A factor is "stale"
when it has fewer than that many rows OR the fetcher raised.

CLI::

    python scripts/factor_coverage_stats.py [--output /tmp/foo.json]
                                            [--factors-yml PATH]
                                            [--lookback-days N]
                                            [--workers N]
                                            [--min-obs N]
                                            [--limit N]

Run from the ``api/`` directory. The script is the worker behind a
weekly catalog health check; tests cover the math with mocked
fetchers — we never hit the live venues from ``pytest``.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

# Make ``pfm`` importable when invoked as ``python scripts/factor_coverage_stats.py``
# from the ``api/`` directory.
_HERE = Path(__file__).resolve().parent
_API_ROOT = _HERE.parent
_SRC = _API_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pfm.factors import (
    FactorConfig,
    fetch_factor_history_dispatch,
    load_factors,
)

FACTORS_YML = _SRC / "pfm" / "factors.yml"

DEFAULT_LOOKBACK_DAYS = 365
DEFAULT_WORKERS = 16
DEFAULT_MIN_OBS = 1


@dataclass(frozen=True)
class FactorObservation:
    """How many rows we recovered for one factor."""

    factor_id: str
    theme: str
    n_obs: int
    error: str = ""

    @property
    def has_data(self) -> bool:
        return self.error == "" and self.n_obs >= 1


@dataclass
class ThemeStats:
    """Per-theme aggregation."""

    theme: str
    factor_count: int = 0
    with_data: int = 0
    obs_samples: list[int] = field(default_factory=list)

    def add(self, obs: FactorObservation, *, min_obs: int) -> None:
        self.factor_count += 1
        if obs.error == "" and obs.n_obs >= min_obs:
            self.with_data += 1
        # We always record the observation count (including zeros) so
        # the median/min reflect every factor in the theme — not just
        # the ones that crossed the freshness threshold.
        self.obs_samples.append(max(0, obs.n_obs))

    @property
    def median_obs(self) -> int:
        if not self.obs_samples:
            return 0
        return int(statistics.median(self.obs_samples))

    @property
    def min_obs(self) -> int:
        if not self.obs_samples:
            return 0
        return int(min(self.obs_samples))

    def to_dict(self) -> dict[str, Any]:
        return {
            "theme": self.theme,
            "factor_count": self.factor_count,
            "with_data": self.with_data,
            "median_obs": self.median_obs,
            "min_obs": self.min_obs,
        }


# Type alias for the fetcher injection seam — tests override this with a
# lightweight stub that returns synthetic DataFrames.
FetcherFn = Callable[[FactorConfig, pd.Timestamp, pd.Timestamp], pd.DataFrame]


def _default_fetcher(fc: FactorConfig, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Production fetcher — hits the real source dispatchers."""
    return fetch_factor_history_dispatch(fc, start, end)


def count_observations(
    fc: FactorConfig,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    fetcher: FetcherFn,
) -> FactorObservation:
    """Run ``fetcher`` and return a structured observation count.

    Any exception is swallowed and recorded in ``error``. We refuse to
    let one bad slug poison the whole report.
    """
    try:
        df = fetcher(fc, start, end)
    except Exception as exc:  # pragma: no cover - exercised via tests
        return FactorObservation(
            factor_id=fc.id,
            theme=fc.theme,
            n_obs=0,
            error=f"{type(exc).__name__}: {exc}",
        )
    if df is None:
        return FactorObservation(factor_id=fc.id, theme=fc.theme, n_obs=0)
    try:
        n_obs = int(len(df))
    except TypeError:
        n_obs = 0
    return FactorObservation(factor_id=fc.id, theme=fc.theme, n_obs=n_obs)


def aggregate_by_theme(
    observations: list[FactorObservation],
    *,
    min_obs: int = DEFAULT_MIN_OBS,
) -> list[ThemeStats]:
    """Group ``observations`` by theme and compute summary stats."""
    by_theme: dict[str, ThemeStats] = {}
    for obs in observations:
        theme = obs.theme or "other"
        bucket = by_theme.setdefault(theme, ThemeStats(theme=theme))
        bucket.add(obs, min_obs=min_obs)
    return sorted(by_theme.values(), key=lambda t: t.theme)


def build_report(
    factors: dict[str, FactorConfig],
    *,
    fetcher: FetcherFn,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
    workers: int = DEFAULT_WORKERS,
    min_obs: int = DEFAULT_MIN_OBS,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Build the report dict by fanning out ``fetcher`` across ``factors``."""
    moment = now or datetime.now(UTC)
    end_ts = pd.Timestamp(moment).normalize()
    start_ts = end_ts - pd.Timedelta(days=lookback_days)

    targets = sorted(factors.values(), key=lambda fc: fc.id)
    results: list[FactorObservation] = []

    if not targets:
        return {
            "checked_at": moment.isoformat().replace("+00:00", "Z"),
            "themes": [],
            "totals": {"factors": 0, "themes": 0, "with_data": 0, "stale": 0},
        }

    if workers <= 1:
        # Sequential path — simpler for tests, deterministic order.
        for fc in targets:
            results.append(count_observations(fc, start_ts, end_ts, fetcher=fetcher))
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            fut_to_fc = {
                ex.submit(count_observations, fc, start_ts, end_ts, fetcher=fetcher): fc
                for fc in targets
            }
            for fut in as_completed(fut_to_fc):
                results.append(fut.result())

    theme_stats = aggregate_by_theme(results, min_obs=min_obs)
    totals_with_data = sum(t.with_data for t in theme_stats)
    return {
        "checked_at": moment.isoformat().replace("+00:00", "Z"),
        "themes": [t.to_dict() for t in theme_stats],
        "totals": {
            "factors": len(results),
            "themes": len(theme_stats),
            "with_data": totals_with_data,
            "stale": len(results) - totals_with_data,
        },
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compute per-theme coverage statistics for factors.yml. Emits a "
            "JSON document describing factor_count / with_data / median_obs / "
            "min_obs per theme plus totals."
        )
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output path for the JSON report. Default: print to stdout.",
    )
    p.add_argument(
        "--factors-yml",
        default=str(FACTORS_YML),
        help="Override path to factors.yml (mostly for tests).",
    )
    p.add_argument(
        "--lookback-days",
        type=int,
        default=DEFAULT_LOOKBACK_DAYS,
        help=f"How far back to request history (default: {DEFAULT_LOOKBACK_DAYS}).",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"ThreadPoolExecutor max_workers (default: {DEFAULT_WORKERS}).",
    )
    p.add_argument(
        "--min-obs",
        type=int,
        default=DEFAULT_MIN_OBS,
        help=(
            "Minimum observation count for a factor to be counted as "
            f"with_data (default: {DEFAULT_MIN_OBS})."
        ),
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N factors after sorting (handy for smoke runs).",
    )
    return p.parse_args(argv)


def run(args: argparse.Namespace, *, fetcher: FetcherFn | None = None) -> int:
    """Execute coverage stats; return the process exit code (always 0)."""
    factors_yml = Path(args.factors_yml)
    factors = load_factors(factors_yml)
    if args.limit is not None and args.limit >= 0:
        keep = dict(sorted(factors.items())[: args.limit])
        factors = keep

    report = build_report(
        factors,
        fetcher=fetcher or _default_fetcher,
        lookback_days=args.lookback_days,
        workers=max(1, int(args.workers)),
        min_obs=max(0, int(args.min_obs)),
    )

    payload = json.dumps(report, indent=2)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload)
        print(
            f"wrote {out_path} ({len(report['themes'])} themes, "
            f"{report['totals']['factors']} factors)"
        )
    else:
        print(payload)
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(_parse_args(argv))


# Re-export for tests + downstream callers that prefer the convenient name.
__all__ = [
    "DEFAULT_LOOKBACK_DAYS",
    "DEFAULT_MIN_OBS",
    "DEFAULT_WORKERS",
    "FactorObservation",
    "ThemeStats",
    "aggregate_by_theme",
    "build_report",
    "count_observations",
    "main",
    "run",
]


# Suppress unused-import lint when run-as-script and silence the "timedelta
# only used for type narrowing" warning in mypy strict mode.
_ = timedelta


if __name__ == "__main__":
    sys.exit(main())
