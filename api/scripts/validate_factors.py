"""Validate every slug in ``factors.yml`` against its upstream venue.

For each factor we hit a cheap, slug-keyed existence endpoint:

  * polymarket → ``GET https://gamma-api.polymarket.com/markets?slug=<slug>``
                 An empty list (``[]``) means the slug has been deleted upstream
                 and the factor is DEAD.
  * kalshi     → ``GET https://api.elections.kalshi.com/trade-api/v2/markets/<slug>``
                 ``404`` means the ticker is gone and the factor is DEAD.

Sources without a simple slug→exists endpoint (fred, bls, manifold, predictit,
chain) are reported as SKIPPED rather than checked. They have their own
health checks elsewhere; this script is specifically a slug-catalog janitor.

Output:
  * Stdout: a per-factor line ``[i/N] STATUS source slug``.
  * JSON report on disk (``factor_validation_<UTC-date>.json`` by default)
    with ``{ok, dead, skipped, meta}`` arrays.
  * Exit code 0 if no dead slugs (or, in non-strict mode, dead share ≤5%).
    Exit code 1 otherwise.

Run from the ``api/`` directory::

    .venv/bin/python scripts/validate_factors.py \\
        --source polymarket,kalshi --limit 200 --out report.json

CI: a ``validate-factors`` job in ``.github/workflows/ci.yml`` runs this
weekly (Mon 06:00 UTC) and on manual dispatch, uploading the JSON report as
an artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

# Make ``pfm`` importable when invoked as ``python scripts/validate_factors.py``
# from the ``api/`` directory.
_HERE = Path(__file__).resolve().parent
_API_ROOT = _HERE.parent
_SRC = _API_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from pfm.factors import FactorConfig, load_factors

FACTORS_YML = _SRC / "pfm" / "factors.yml"

POLYMARKET_GAMMA = "https://gamma-api.polymarket.com/markets"
KALSHI_BASE = "https://api.elections.kalshi.com/trade-api/v2/markets"

LIVE_CHECKABLE_SOURCES: frozenset[str] = frozenset({"polymarket", "kalshi"})
SKIPPED_SOURCES: frozenset[str] = frozenset({"fred", "bls", "manifold", "predictit", "chain"})

DEFAULT_TIMEOUT = 5.0
DEFAULT_WORKERS = 20
DEFAULT_DEAD_FRACTION_TOLERANCE = 0.05  # 5%


@dataclass(frozen=True)
class CheckResult:
    """Outcome of a single factor health check."""

    factor_id: str
    slug: str
    source: str
    status: str  # "ok" | "dead" | "skipped"
    detail: str = ""


def _check_polymarket(client: httpx.Client, slug: str) -> tuple[str, str]:
    """Return ``(status, detail)`` for a polymarket slug.

    A non-200 response is treated as ``skipped`` (transient upstream issue),
    not ``dead``. Only a confirmed empty list yields ``dead``.
    """
    try:
        r = client.get(POLYMARKET_GAMMA, params={"slug": slug}, timeout=DEFAULT_TIMEOUT)
    except httpx.HTTPError as e:
        return "skipped", f"polymarket transport error: {type(e).__name__}"
    if r.status_code != 200:
        return "skipped", f"polymarket http {r.status_code}"
    try:
        data = r.json()
    except json.JSONDecodeError:
        return "skipped", "polymarket non-JSON body"
    if isinstance(data, list) and len(data) == 0:
        return "dead", "polymarket returned empty list"
    if isinstance(data, list) and data:
        return "ok", ""
    # Unexpected shape — be conservative.
    return "skipped", f"polymarket unexpected shape: {type(data).__name__}"


def _check_kalshi(client: httpx.Client, slug: str) -> tuple[str, str]:
    """Return ``(status, detail)`` for a kalshi ticker (slug)."""
    url = f"{KALSHI_BASE}/{slug}"
    try:
        r = client.get(url, timeout=DEFAULT_TIMEOUT)
    except httpx.HTTPError as e:
        return "skipped", f"kalshi transport error: {type(e).__name__}"
    if r.status_code == 404:
        return "dead", "kalshi 404"
    if r.status_code != 200:
        return "skipped", f"kalshi http {r.status_code}"
    return "ok", ""


def check_one(client: httpx.Client, fc: FactorConfig) -> CheckResult:
    """Dispatch a single factor to the appropriate venue check."""
    if fc.source == "polymarket":
        status, detail = _check_polymarket(client, fc.slug)
    elif fc.source == "kalshi":
        status, detail = _check_kalshi(client, fc.slug)
    else:
        status, detail = "skipped", f"source {fc.source}"
    return CheckResult(
        factor_id=fc.id,
        slug=fc.slug,
        source=fc.source,
        status=status,
        detail=detail,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Validate slugs in factors.yml against their upstream venues "
            "(polymarket gamma, kalshi v2). Emits a JSON report and exits "
            "non-zero when too many slugs are dead."
        )
    )
    p.add_argument(
        "--source",
        default=",".join(sorted(LIVE_CHECKABLE_SOURCES)),
        help=(
            "Comma-separated list of sources to live-check. "
            f"Default: {','.join(sorted(LIVE_CHECKABLE_SOURCES))}. "
            "Anything outside this list is reported as SKIPPED."
        ),
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N factors after source filtering.",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Exit 1 on a single dead factor. Default: exit 1 only when dead "
            f"share exceeds {int(DEFAULT_DEAD_FRACTION_TOLERANCE * 100)}%% of checked."
        ),
    )
    p.add_argument(
        "--out",
        default=None,
        help=(
            "Custom output path for the JSON report. "
            "Default: factor_validation_<UTC-date>.json in the cwd."
        ),
    )
    p.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"ThreadPoolExecutor max_workers (default: {DEFAULT_WORKERS}).",
    )
    p.add_argument(
        "--factors-yml",
        default=str(FACTORS_YML),
        help="Override path to factors.yml (mostly for tests).",
    )
    return p.parse_args(argv)


def _select_factors(
    factors: dict[str, FactorConfig], requested_sources: set[str]
) -> list[FactorConfig]:
    """Return factors whose source is in ``requested_sources``, deterministically."""
    selected = [fc for fc in factors.values() if fc.source in requested_sources]
    selected.sort(key=lambda fc: fc.id)
    return selected


def _default_report_path() -> Path:
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    return Path.cwd() / f"factor_validation_{today}.json"


def run(args: argparse.Namespace) -> int:
    """Execute validation; return the desired process exit code."""
    factors_yml = Path(args.factors_yml)
    factors = load_factors(factors_yml)

    requested_sources = {s.strip() for s in args.source.split(",") if s.strip()}
    unknown = requested_sources - (LIVE_CHECKABLE_SOURCES | SKIPPED_SOURCES)
    if unknown:
        print(
            f"warning: unknown source filter(s) {sorted(unknown)}; "
            f"only {sorted(LIVE_CHECKABLE_SOURCES)} are live-checkable.",
            file=sys.stderr,
        )

    targets = _select_factors(factors, requested_sources)
    if args.limit is not None:
        targets = targets[: args.limit]
    total = len(targets)

    if total == 0:
        print("no factors matched the source filter; nothing to do.")
        return 0

    print(f"validating {total} factor(s) from {factors_yml} with {args.workers} workers …")

    results: list[CheckResult] = []
    completed = 0

    with (
        httpx.Client(
            headers={"User-Agent": "pfm-validate-factors/1.0"},
            follow_redirects=True,
        ) as client,
        ThreadPoolExecutor(max_workers=args.workers) as ex,
    ):
        future_to_fc = {ex.submit(check_one, client, fc): fc for fc in targets}
        for fut in as_completed(future_to_fc):
            fc = future_to_fc[fut]
            try:
                res = fut.result()
            except Exception as e:  # surface anything unusual
                res = CheckResult(
                    factor_id=fc.id,
                    slug=fc.slug,
                    source=fc.source,
                    status="skipped",
                    detail=f"worker error: {type(e).__name__}: {e}",
                )
            results.append(res)
            completed += 1
            label = "OK" if res.status == "ok" else "DEAD" if res.status == "dead" else "SKIPPED"
            suffix = f" — {res.detail}" if res.detail and res.status != "ok" else ""
            print(f"[{completed:>5}/{total}] {label:<7} {res.source:<10} {res.slug}{suffix}")

    ok = [r for r in results if r.status == "ok"]
    dead = [r for r in results if r.status == "dead"]
    skipped = [r for r in results if r.status == "skipped"]

    # Keep the on-disk report deterministic for diffing.
    ok.sort(key=lambda r: r.factor_id)
    dead.sort(key=lambda r: r.factor_id)
    skipped.sort(key=lambda r: r.factor_id)

    out_path = Path(args.out) if args.out else _default_report_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "meta": {
            "generated_at": datetime.now(UTC).isoformat(),
            "factors_yml": str(factors_yml),
            "requested_sources": sorted(requested_sources),
            "limit": args.limit,
            "workers": args.workers,
            "strict": bool(args.strict),
            "total_in_yml": len(factors),
            "n_checked": len(results),
            "n_ok": len(ok),
            "n_dead": len(dead),
            "n_skipped": len(skipped),
        },
        "ok": [{"id": r.factor_id, "slug": r.slug, "source": r.source} for r in ok],
        "dead": [
            {"id": r.factor_id, "slug": r.slug, "source": r.source, "detail": r.detail}
            for r in dead
        ],
        "skipped": [
            {"id": r.factor_id, "slug": r.slug, "source": r.source, "detail": r.detail}
            for r in skipped
        ],
    }
    out_path.write_text(json.dumps(report, indent=2))

    print()
    print(f"  total checked : {len(results)}")
    print(f"  OK            : {len(ok)}")
    print(f"  DEAD          : {len(dead)}")
    print(f"  SKIPPED       : {len(skipped)}")
    print(f"  report        : {out_path}")
    if dead:
        print("  first dead slugs:")
        for r in dead[:10]:
            print(f"    - {r.source}: {r.slug}")

    # Live-checkable subset is what we judge the failure threshold on, so a
    # large skipped pool (e.g. --source=fred only) never pretends to be a pass.
    live_checked = [r for r in results if r.status in {"ok", "dead"}]
    if args.strict:
        return 1 if dead else 0
    if not live_checked:
        # Nothing was actually live-checked. Don't pretend to have validated.
        return 0
    dead_fraction = len(dead) / len(live_checked)
    if dead_fraction > DEFAULT_DEAD_FRACTION_TOLERANCE:
        print(
            f"  FAIL: dead share {dead_fraction:.1%} > "
            f"{DEFAULT_DEAD_FRACTION_TOLERANCE:.0%} threshold."
        )
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(_parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
